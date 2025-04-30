from src.arguments import get_training_args
from src.vec_env import PromptEnv
from src.data_utils import custom_load_dataset
from src.utils import get_llama_embeddings, get_embedding, get_ic_sentences_faiss, select_k_samples_per_class, generate_vllm,  get_senemb_embbeddings, process_data_multiple_fields, convert_to_dicts
from src.a2c_ppo_acktr import utils
from src.a2c_ppo_acktr.model import Policy
from src.a2c_ppo_acktr.storage import RolloutStorage
from src.a2c_ppo_acktr.algo.ppo import PPO
from src.a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_eval, make_vec_envs_fseval
from src.reward_llm import setup_reward_llm, setup_qa_llm
from src.sample_memory import Memory, VerbalizerMemory, VerbalizerAllMemory
from src.evaluation import evaluate_actor_critic, evaluate
from sentence_transformers import SentenceTransformer
# export pythonpath if problem arises
from angle_emb import AnglE

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from pathlib import Path
from promptsource.templates import DatasetTemplates
import random
import transformers

import copy
import os
from collections import deque
import time
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
from datasets import load_dataset

import torch
import torch.multiprocessing as mp
import math

import pickle
import sys

class Normalizer:
    _STATS_FNAME = "env_stats.pickle"

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, in_size, num_process, device='cpu', dtype=torch.float):
        device='cpu'
        self.mean = torch.zeros((num_process, in_size), device=device, dtype=dtype)
        self.std = torch.ones((num_process, in_size), device=device, dtype=dtype)
        self.num_process = num_process
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device
        self.count = self.eps + torch.zeros((num_process, in_size), device=device, dtype=dtype)

    def update_stats(self, batch_data, batch_indices):
        if isinstance(batch_data, np.ndarray):
            batch_data = torch.from_numpy(batch_data).float().to(data.device)
        batch_data = batch_data.to('cpu')
        if isinstance(batch_indices, np.ndarray):
            batch_indices = torch.from_numpy(batch_indices).to('cpu')
        for i in range(self.num_process):
            index = (batch_indices == i).nonzero()
            data = torch.gather(batch_data, dim=0, index=index)
            if data.shape[0] > 1:
                batch_mean = data.mean(0, keepdim=True)
                batch_var = data.var(0, keepdim=True)
                batch_count = data.shape[0]
                self.update_from_moments(batch_mean, batch_var, batch_count, i)

    def update_from_moments(self, batch_mean, batch_var, batch_count, index):
        delta = batch_mean - self.mean[[index]]
        tot_count = self.count[[index]] + batch_count

        new_mean = self.mean[[index]] + delta * batch_count / tot_count
        m_a = torch.square(self.std[[index]]) * (self.count[[index]])
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count[[index]] * batch_count / (self.count[[index]] + batch_count)
        new_var = M2 / (self.count[[index]] + batch_count)

        new_count = batch_count + self.count[[index]]

        self.mean[[index]] = new_mean
        self.std[[index]] = torch.sqrt(new_var)
        self.count[[index]] = new_count

    def normalize(self, val, index):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        mean = self.mean[index]
        std = std[index]
        return (val - mean.to(val.device)) / std.to(val.device)

    def denormalize(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        return std * val.to(val.device) + self.mean.to(val.device)
 

def random_sampling(sentences, labels, num):
    from copy import deepcopy
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)


def main():
    args = get_training_args()
    print("Baseline: ", args.baseline)
    if args.baseline == 'all':
        if args.n_actions == 360:
            print("Memory for 6 examples")
        elif args.n_actions == 1680:
            print("Memory for 8 examples")
    print("Dataset: ", args.dataset)
    print("====")
    print("Seed: ", args.seed)
    print("Distance metric: ", args.distance_metric)
    print("torch.device: ", torch.cuda.current_device())

    if args.type_embd == 'senemb':
        print("Sentence embedding model: ", args.senemb_model_name)
    elif args.type_embd == 'llama':
        print("LLAMA")
    if args.wandb:
        wandb.init(project="RL for LLM")   
    ctx = mp.get_context('spawn')
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.use_attention:
        print("Use attention")
    if args.reward_type == 'prob':
        print("Reward type PROB")
    elif args.reward_type == 'acc':
        print("Reward type ACC")

    # NOTE: the current max for few_shot dataset (32 samples)
    cur_max = 0
    n_slots = args.n_slots

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    device = torch.device("cuda:0")

    # Construct params for training
    params = {
        'dataset': args.dataset,
        'n_candidates': args.n_candidates,
        'action_instruction_prefix': f"You are given a prompt with instruction and in-context examples and a test example, each is separated by 2 newlines. Your job is to generate {args.n_candidates} better instructions. Each of the the generated instructions should be a completed instruction. Start each instruction with 'Instruction'. There are two classes that should not be changed for the instructions, which are 'great' and 'terrible'. Do not use other words for classification. Do not make any biased instructions. Do not give redundant sentences in your response. After having generated all the instructions, do not generate any more sentences.",
        # 'action_instruction_prefix': f"You are a helpful assistant. You are given a prompt with instruction and in-context examples below. Your job is to improve and response {args.n_candidates} prompts, each starts with 'Instruction'. Do not change the classes of the instructions, which are 'great' and 'terrible'. Do not generate anythingelse beside the 3 improved instructions.",
        'n_examples': 2,
        'n_ic_examples': 4,
        'lambda1': 2.0,
        'lambda2': 1.8,
        'verbalize': args.verbalize,
        'dataset_name': args.dataset,
        'task_format': args.task_format,
        'reward_model_name': args.reward_model_name,
        'prompt_format': args.prompt_format,
        'debug_mode': args.debug_mode,
        'extra_exs': args.extra_exs,
        'n_pool_exs': args.n_pool_exs
    }


    # Initialize reward LLM and tokenizer LLM
    if args.task_format == 'classification':
        reward_llm, tokenizer_llm = setup_reward_llm(args.reward_model_name, gpu_id=0)

    # Load dataset
    train_sentences_, train_labels_, valid_sentences_, valid_labels_, test_sentences_, test_labels_ = custom_load_dataset(params, change_params=True)

    
    # NOTE: We now load the training data from file, same as benchmark
    train_file_dict = {'train': f'/home/s223540177/dai/RLforLLM/src/data_benchmark/{args.dataset}/benchmark/train.tsv'}
    train_sentences = load_dataset('csv', data_files=train_file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']

    # get the in-context exemplar pool
    if args.dataset in ['glue/sst2', 'glue/cola', 'ag_news', 'cr']:
        if args.extra_exs:
            # load the extra ic_examples
            print(f"EXTRA POOL OF {args.n_pool_exs} EXAMPLES")
            file_dict = {'train': f'src/data_benchmark/{args.dataset}/benchmark/pool_{args.n_pool_exs}_examples.tsv'}
            in_context_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
            in_context_labels = in_context_sentences['label']
        else:
            file_dict = {'train': f'src/data_benchmark/{args.dataset}/benchmark/in_context.tsv'}
            in_context_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
            in_context_labels = in_context_sentences['label']
    else:
        file_dict = {'train': f'src/data_benchmark/{args.dataset}/benchmark/in_context.tsv'}
        in_context_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
        in_context_labels = in_context_sentences['label']

    # get the test sentences
    file_dict = {'train': f'src/data_benchmark/{args.dataset}/benchmark/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']


    # Process the dataset
    if params['dataset'] in ['yelp_polarity', 'trec' 'emotion', 'ag_news', 'imdb', 'subj', 'mteb/twitter', 'poem_sentiment', 'SetFit/sst5',\
                            'trec', 'hate_speech18']:
        train_sentences = train_sentences['text']
        in_context_sentences = in_context_sentences['text']
        valid_sentences, valid_labels = test_sentences['text'], test_labels
    elif params['dataset'] in ['glue/mrpc', 'glue/qnli', 'glue/mnli', 'snli', 'super_glue/boolq', 'glue/qqp']:
        valid_sentences, valid_labels = convert_to_dicts(args, test_sentences)
    elif params['dataset'] in ['super_glue/rte']:
        valid_sentences, valid_labels = test_sentences, test_labels # this dataset has more than just train sentence
        # NOTE: Further process the data here
        valid_sentences, valid_labels = process_data_multiple_fields(params, valid_sentences)
    else:
        train_sentences = train_sentences['sentence']
        in_context_sentences = in_context_sentences['sentence']
        valid_sentences, valid_labels = test_sentences['sentence'], test_labels
    

    # NOTE: fs_sentences are used as checkpoints, to quickly evaluate the memory
    if args.dataset == 'glue/sst2':
        fs_sentences, fs_labels = select_k_samples_per_class(args, valid_sentences, valid_labels, 16, args.seed)
    else:
        fs_sentences, fs_labels = select_k_samples_per_class(args, valid_sentences, valid_labels, 128, args.seed)


    if args.reward_model_name == 'gpt2-xl':
        obs_size = 1600
    elif args.reward_model_name == 'gpt2-large':
        obs_size = 1280
    elif args.reward_model_name == 'gpt2-medium':
        obs_size = 1024
    elif args.reward_model_name == 'roberta-large':
        obs_size = 1024
    elif args.reward_model_name == 't5-large':
        obs_size = 1024
    elif args.reward_model_name == 't5-11b':
        obs_size = 1024
    elif args.reward_model_name == 't5-3b':
        obs_size = 1024
    elif args.reward_model_name in ['llama-v2-7B-chat', 'mistralai/Mistral-7B-Instruct-v0.2']:
        obs_size = 4096
    else:
        assert False

    # Initialize emb model
    if args.type_embd == 'llama':
        model = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
        tokenizer = None
    elif args.type_embd == 'mistral':
        model = SentenceTransformer('Salesforce/SFR-Embedding-Mistral')
        tokenizer = None
    else:
        model, tokenizer = SentenceTransformer(args.senemb_model_name), None
        if args.senemb_model_name == "mixedbread-ai/mxbai-embed-large-v1":
            args.obs_size = 1024
        elif args.senemb_model_name == "Alibaba-NLP/gte-large-en-v1.5":
            args.obs_size = 1024


    # NOTE: Initialize memory
    n_actions_memory = args.n_actions
    memory_instance = Memory(args, len(train_sentences), n_actions_memory)
    memory_instance_random = Memory(args, len(train_sentences), n_actions_memory) # this is for random example, memory permutation baseline

    # NOTE: Initalize memory extended - remember: the number of memory slots is still the same
    memory_extended_instance = Memory(args, len(train_sentences), n_actions_memory)

    if args.dataset in ['cr', 'movie_review']:
        prompt_template = DatasetTemplates('glue/sst2')
    elif args.dataset in ['mteb/twitter']:
        prompt_template = DatasetTemplates('tweet_eval/sentiment')
    else:
        prompt_template = DatasetTemplates(args.dataset)

    n_actions_memory_verbalizer = len(prompt_template.all_template_names)
    # NOTE: Initialize verbalizer memory
    if args.baseline in ['memory', 'memory_seperate']:
        memory_verbalizer_instance = VerbalizerMemory(args, len(train_sentences), n_actions_memory_verbalizer)
        memory_verbalizer_instance_random = VerbalizerMemory(args, len(train_sentences), n_actions_memory_verbalizer)
        # NOTE: Initialize verbalizer memory extended
        memory_verbalizer_instance_extended = VerbalizerMemory(args, len(train_sentences), n_actions_memory_verbalizer)
    elif args.baseline in ['all']:
        memory_verbalizer_instance = VerbalizerAllMemory(args, len(train_sentences), n_actions_memory_verbalizer, n_actions_memory)
        memory_verbalizer_instance_random = VerbalizerAllMemory(args, len(train_sentences), n_actions_memory_verbalizer, n_actions_memory)
        memory_verbalizer_instance_extended = VerbalizerAllMemory(args, len(train_sentences), n_actions_memory_verbalizer, n_actions_memory)
    else:
        memory_verbalizer_instance = Memory(args, len(train_sentences), n_actions_memory_verbalizer)
        memory_verbalizer_instance_random = Memory(args, len(train_sentences), n_actions_memory_verbalizer)
        # NOTE: Initialize verbalizer memory extended
        memory_verbalizer_instance_extended = Memory(args, len(train_sentences), n_actions_memory_verbalizer)


    envs = make_vec_envs(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature=0.7, top_p=0.95, n_candidates=params['n_candidates'], n_slots=n_slots, epsilon=0.005, \
                        num_processes=args.num_processes, i=0, gpu_id=0, evaluate=False, external_memory=memory_instance, \
                        external_memory_random=memory_instance_random, external_memory_extended=memory_extended_instance, \
                        memory_verbalizer=memory_verbalizer_instance, memory_verbalizer_random=memory_verbalizer_instance_random,\
                        memory_verbalizer_extended=memory_verbalizer_instance_extended, ic_examples=in_context_sentences, \
                        ic_labels=in_context_labels, model=model, tokenizer=tokenizer)

    num_test_samples_per_gpu = int(len(valid_sentences)/args.num_actors) - 1
    total_test_samples = int(len(valid_sentences))
    print("Total train samples:", len(train_sentences))
    print("Total test samples: ", total_test_samples)
    print("Num test samples per GPU: ", num_test_samples_per_gpu)
    

    eval_envs = []
    fs_env = []

    fs_env.append(make_vec_envs_fseval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, fs_sentences, fs_labels, \
                                        temperature=0.7, top_p=0.95, n_candidates=params['n_candidates'], n_slots=n_slots, epsilon=0.005, \
                                        num_processes=args.num_processes, i=0, gpu_id=0, evaluate=True, external_memory=memory_instance, \
                                        external_memory_random=memory_instance_random, external_memory_extended=memory_extended_instance, \
                                        memory_verbalizer=memory_verbalizer_instance, memory_verbalizer_random=memory_verbalizer_instance_random,\
                                        memory_verbalizer_extended=memory_verbalizer_instance_extended, ic_examples=in_context_sentences, \
                                        ic_labels=in_context_labels, model=model, tokenizer=tokenizer))
    
    for i in range(args.num_actors):
        # TODO: when using multiple gpus, change to i%torch.cuda.device_count()
        reward_llm_eval, tokenizer_llm_eval = setup_reward_llm(args.reward_model_name, gpu_id=i%torch.cuda.device_count())
        eval_env = make_vec_envs_eval(params, args, reward_llm_eval, tokenizer_llm_eval, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature=0.7, top_p=0.95, n_candidates=params['n_candidates'], n_slots=args.n_slots, epsilon=0.005, \
                    num_processes=args.num_processes, i=i, gpu_id=i%torch.cuda.device_count(), evaluate=True, external_memory=memory_instance, external_memory_random=memory_instance_random,\
                    external_memory_extended=memory_extended_instance, memory_verbalizer=memory_verbalizer_instance, memory_verbalizer_random=memory_verbalizer_instance_random,\
                    memory_verbalizer_extended=memory_verbalizer_instance_extended, ic_examples=in_context_sentences, ic_labels=in_context_labels, model=model, tokenizer=tokenizer)
        eval_envs.append(eval_env)
    # Initialize actor-critic model
    # NOTE: be mindful of the Reward model & obs_size. Fix in arguments file

    # num_blocks is the number of tokens in the input sequence
    num_blocks = int(envs.observation_space.shape[0]/ args.obs_size)
    print('num_blocks: ', num_blocks)
    actor_critic = Policy(
        args,
        envs.observation_space.shape,
        envs.action_space,
        use_attention=args.use_attention,
        device=device,
        block_size=num_blocks,
        base_kwargs={'recurrent': args.recurrent_policy,
        'hidden_size': 1024})
    actor_critic.to(device)

    """
    Policy includes GPT base and MLP base, followed by a distribution to sample action
    GPT base has encoders only architecture, receives 5 elements:
        1. (wte): word token embedding - Linear: in: 1024 -> out: 128
        2. (wpe): word position embedding - Embeddings (n_tokens, 128)
        3. (wae): action embedding - Embeddings (n_actions, 128)
        4. (drop): dropout
        5. (h): Encoders only architecture (x4)
    
    MLP base has actor and critic, which share the same architecture. There is also critic_linear that maps:
        1. (fc1): Linear: in: n_tokens * 1024 -> out: 1024
        2. (fc2): Linear: in: 1024 -> out: 1024
        (For critic) (critic_linear): Linear: in: 1024 -> out: 1 - estimate the value function (might be q or v)
    
    Distribution is a Categorical distribution, which receives the final state representation and returns the probabilities over actions
    """


    if args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm
        )
    
    # print("Env obs shape", env.observation_space.shape)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # This is to normalize rewards after every args.num_steps steps
    all_rews = []

    # Episode rewards only accept the last 10 rewards
    episode_rewards = deque(maxlen=10)

    # Calculate number of updates
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    all_indexs = []

    if args.normalize_rew:
        rew_normalizer = Normalizer(1, 16 * len(params['label_dict'].keys()))
    else:
        rew_normalizer = None


    # START TRAINING
    # Make sure we dont train random baselines
    if args.baseline == 'random_random' or args.baseline == 'random_topk' or args.baseline == 'heuristic':
        print(f"Evaluate {args.baseline} baseline")
        evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs=False)
    else:
        print(f"Training {args.baseline} baseline")
        for j in tqdm(range(num_updates), desc="Updates"):
            time1 = time.time()
            if args.use_linear_lr_decay:
                # Decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                # Sample action 
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    
                # Observation reward and next obs
                subset_idxs = envs.venv.envs[0].subset_idxs
                all_indexs.append(copy.deepcopy(subset_idxs))
                obs, reward, done, infos = envs.step(action)

                # Normalize observation here
                if args.normalize_obs:
                    actor_critic.base.normalizer.update_stats(obs)
                all_rews.append(reward)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode_r'])
                if done[0]:
                    episode_rewards.append(info['episode_r'])


                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action, 
                                action_log_prob, torch.Tensor(subset_idxs).unsqueeze(-1), value, reward, masks, bad_masks)


            # Normalize reward here
            if args.normalize_rew:
                rew_normalizer.update_stats(torch.cat(all_rews, dim=0), torch.from_numpy(np.concatenate(all_indexs, axis=0)))
                rollouts.update_rew(rew_normalizer)
                all_indexs = []
                all_rews = []

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, 
                                        args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)                

            if args.wandb:
                wandb.log({"Value loss": value_loss})
                wandb.log({"Action loss": action_loss})
                wandb.log({"Entropy loss": dist_entropy})

            rollouts.after_update()

            time2 = time.time()
            

            if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)

                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                torch.save(
                    actor_critic.state_dict(),
                    os.path.join(args.save_dir + f"{args.env_name}{dt}.pt")
                )

            if args.inspect:
                acc = evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs=False)
                print(f"Test accuracy is {acc}")
                if args.debug_mode:
                    quit()
            else:
                if (args.eval_interval is not None and j % args.eval_interval == 0 and j > 0) or j == num_updates - 1:
                    if j == num_updates - 1:
                        print("Last memory evaluation")
                        fs_acc = evaluate_actor_critic(actor_critic, fs_env, args, ctx, total_fs_test_samples, num_test_samples_per_gpu_fs, params, obs_size, is_fs=True)
                        print(f"Last few-shot accuracy is {fs_acc}")
                        acc = evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs=False)
                        print(f"Last test accuracy is {acc}")
                    else:
                        if j == 0:
                            print("First memory evaluation")
                        total_fs_test_samples = int(len(fs_sentences))
                        num_test_samples_per_gpu_fs = total_fs_test_samples # as the fseval env is put on GPU 0 only


                        
                        fs_acc = evaluate_actor_critic(actor_critic, fs_env, args, ctx, total_fs_test_samples, num_test_samples_per_gpu_fs, params, obs_size, is_fs=True)
                        if args.dataset in ['ag_news']:
                            if fs_acc > cur_max:
                                print(f"Few-shot accuracy is {fs_acc}, which greater than the threshold {cur_max}")
                                print("Start evaluating on the test set")
                                cur_max = fs_acc
                                acc = evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs=False)
                                print("Done evaluating on the test set")
                                print(f"Test accuracy is {acc}")
                                print("=============================================================================")
                            else:
                                print(f"Few-shot accuracy is {fs_acc}, which is below the threshold {cur_max}")
                                print("Not evaluating on the test set")
                        else:
                            if fs_acc >= cur_max:
                                print(f"Few-shot accuracy is {fs_acc}, which greater than the threshold {cur_max}")
                                print("Start evaluating on the test set")
                                cur_max = fs_acc
                                acc = evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs=False)
                                print("Done evaluating on the test set")
                                print(f"Test accuracy is {acc}")
                                print("=============================================================================")
                            else:
                                print(f"Few-shot accuracy is {fs_acc}, which is below the threshold {cur_max}")
                                print("Not evaluating on the test set")

if __name__=="__main__":
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()