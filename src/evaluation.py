import numpy as np
import copy
import wandb
from tqdm import tqdm
import gym
from collections import deque

from .init_prompt import construct_initial_prompt, generate_template_prompts
from .utils import eval_accuracy, select_k_samples_per_class, select_n_random_samples
from .reward_llm import get_score, get_embeddings

from .a2c_ppo_acktr.storage import RolloutStorage

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
logging.getLogger("Python").setLevel(logging.WARNING)

device = torch.device("cuda:0")

def evaluate(params, args, actor_critic, seed, num_processes, num_start, num_test, eval_envs):
    print("Evaluate begins")

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    current_acc = []

    eval_envs = eval_envs[0]

    obs = eval_envs.reset()

    for idx in tqdm(range(num_start, num_test, num_processes), desc="Evaluating"):
        if idx + num_processes > num_test:
            idxs = np.arange(idx, num_test)
        else:
            idxs = np.arange(idx, idx + num_processes)
        # TODO: auto this
        # eval_envs.venv.envs[0].idxs = idxs
        eval_envs.envs[0].idxs = idxs
        obs = eval_envs.reset()
        _done = False
        # at test time, agent allowed to edit for maximum of args.num_steps steps
        while not _done:
            with torch.no_grad():
                value, action, action_log_prob, eval_recurrent_hidden_states = actor_critic.act(
                    # obs.cpu().cuda(),
                    torch.tensor(obs).float().cuda(),
                    # eval_recurrent_hidden_states.cpu().cuda(),
                    # eval_masks.cpu().cuda(),
                    None, 
                    None,
                    deterministic=True)
                    # deterministic=False)
            
            obs, reward, done, info = eval_envs.step(action)
            _done = done[0]

            if args.baseline == 'memory_qa':
                current_acc.append(info[0]['memory_qa_acc'])
            print("Current accuracy: ", np.mean(current_acc))

    return np.mean(current_acc)


def evaluate_lm(i, actor_critic, obs_rms, eval_envs, seed, num_processes, total_samples, num_start,\
             num_test, params, args, obs_size, initial_acc_list, current_acc_list, random_topk_acc_list,\
             random_random_acc_list, random_permutation_acc_list, extended_acc_list, heuristic_acc_list, \
             memory_seperate_list, memory_qa_list):
    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.obs_rms = obs_rms
    import time
    tik = time.time()
    print(f'Evaluation {i} started at {tik}', flush=True)
    if args.load_ckpt: 
        file_path = 'checkpoints/'+str(args.models)+'_'+str(args.datasets)+'_'+str(args.seed)+'/'
        eval_envs.envs[0].load_ckpt(file_path, i, num_test)
    else:
        pass

    obs = eval_envs.reset()
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size).cuda()
    # eval_masks = torch.zeros(num_processes, 1).cuda()

    initial_acc = []
    current_acc = []
    random_topk_acc = []
    random_random_acc = []
    random_permutation_acc = []
    extended_acc = []
    heuristic_acc = []
    memory_seperate_acc = []
    memory_qa_acc = []
    

    for idx in tqdm(range(num_start, num_test, num_processes), desc="Evaluating"):
        if idx + num_processes > num_test:
            idxs = np.arange(idx, num_test)
        else:
            idxs = np.arange(idx, idx + num_processes)
        # TODO: auto this
        # eval_envs.venv.envs[0].idxs = idxs
        eval_envs.envs[0].idxs = idxs
        obs = eval_envs.reset()
        _done = False
        while not _done:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    # obs.cpu().cuda(),
                    torch.tensor(obs).float(),
                    # eval_recurrent_hidden_states.cpu().cuda(),
                    # eval_masks.cpu().cuda(),
                    None, 
                    None,
                    deterministic=True)
                    # deterministic=False)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            _done = done[0]

            # eval_masks = torch.tensor(
            #     [[0.0] if done_ else [1.0] for done_ in done],
            #     dtype=torch.float32,
            #     device=device)
        # total_correct += infos[0]['correct']
        # total_orig_correct += infos[0]['orig_correct']
        # total_samples += infos[0]['total']
        # _tp += infos[0]['tp']
        # _fp += infos[0]['fp']
        # _fn += infos[0]['fn']
        initial_acc.append(infos[0]['initial_acc'])

        if args.baseline == 'memory':
            current_acc.append(infos[0]['current_acc'])
        elif args.baseline == 'random_topk':
            random_topk_acc.append(infos[0]['random_topk_acc'])
            print("Random topk accuracy: ", np.mean(random_topk_acc))
        elif args.baseline == 'random_random':
            random_random_acc.append(infos[0]['random_random_acc'])
            print("Random random accuracy: ", np.mean(random_random_acc))
        elif args.baseline == 'memory_random':
            random_permutation_acc.append(infos[0]['random_permutation_acc'])
        elif args.baseline == 'all':
            extended_acc.append(infos[0]['extended_acc'])
        elif args.baseline == 'heuristic':
            heuristic_acc.append(infos[0]['heuristic_acc'])
        elif args.baseline == 'memory_seperate':
            memory_seperate_acc.append(infos[0]['memory_seperate_acc'])
        elif args.baseline == 'memory_qa':
            memory_qa_acc.append(infos[0]['memory_qa_acc'])
        


    print(f"Process {i} has finished evaluation", flush=True)

    initial_acc_list.put(np.mean(initial_acc))
    if args.baseline == 'memory':
        current_acc_list.put(np.mean(current_acc))
    elif args.baseline == 'random_topk':
        random_topk_acc_list.put(np.mean(random_topk_acc))
    elif args.baseline == 'random_random':
        random_random_acc_list.put(np.mean(random_random_acc))
    elif args.baseline == 'memory_random':
        random_permutation_acc_list.put(np.mean(random_permutation_acc))
    elif args.baseline == 'all':
        extended_acc_list.put(np.mean(extended_acc))
    elif args.baseline == 'heuristic':
        heuristic_acc_list.put(np.mean(heuristic_acc))
    elif args.baseline == 'memory_seperate':
        memory_seperate_list.put(np.mean(memory_seperate_acc))
    elif args.baseline == 'memory_qa':
        memory_qa_list.put(np.mean(memory_qa_acc))

    # print("TOTAL SAMPLES: ", total_samples)
    # print(" Evaluation using {} episodes: mean reward {:.5f}, original mean reward {:.5f}".format(
    #     len(eval_episode_rewards), total_correct/total_samples, total_orig_correct/total_samples), flush=True)



def evaluate_actor_critic(actor_critic, eval_envs, args, ctx, total_test_samples, num_test_samples_per_gpu, params, obs_size, is_fs):
    actor_critic.to('cpu')
    initial_acc = ctx.Queue()
    current_acc = ctx.Queue()
    random_topk_acc = ctx.Queue()
    random_random_acc = ctx.Queue()
    random_permutation_acc = ctx.Queue()
    extended_acc = ctx.Queue()
    heuristic_acc = ctx.Queue()
    memory_qa_acc = ctx.Queue()
    memory_seperate_acc = ctx.Queue()
    evaluate_processes = []
    for i in range(args.num_actors):
        eval_proc = ctx.Process(
            target=evaluate_lm,
            args=(i, actor_critic, None, eval_envs[i], args.seed, 
                  args.num_processes, total_test_samples, num_test_samples_per_gpu * i, 
                  num_test_samples_per_gpu, params, args, obs_size, 
                  initial_acc, current_acc, random_topk_acc, random_random_acc, 
                  random_permutation_acc, extended_acc, heuristic_acc, memory_seperate_acc,
                  memory_qa_acc))
        eval_proc.start()
        evaluate_processes.append(eval_proc)
    for eval_proc in evaluate_processes:
        eval_proc.join()
    initial_acc_list = []
    current_acc_list = []
    random_topk_acc_list = []
    random_random_acc_list = []
    random_permutation_acc_list = []
    extended_acc_list = []
    heuristic_acc_list = []
    memory_seperate_list = []
    memory_qa_list = []

    for i in range(initial_acc.qsize()):
        initial_acc_list.append(initial_acc.get())
        if args.baseline == 'memory':
            current_acc_list.append(current_acc.get())
        elif args.baseline == 'random_topk':
            random_topk_acc_list.append(random_topk_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(random_topk_acc_list)/len(random_topk_acc_list)}")
        elif args.baseline == 'random_random':
            random_random_acc_list.append(random_random_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(random_random_acc_list)/len(random_random_acc_list)}")
        elif args.baseline == 'memory_random':
            random_permutation_acc_list.append(random_permutation_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(random_permutation_acc_list)/len(random_permutation_acc_list)}")
        elif args.baseline == 'all':
            extended_acc_list.append(extended_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(extended_acc_list)/len(extended_acc_list)}")
        elif args.baseline == 'heuristic':
            heuristic_acc_list.append(heuristic_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(heuristic_acc_list)/len(heuristic_acc_list)}")
        elif args.baseline == 'memory_seperate':
            memory_seperate_list.append(memory_seperate_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(memory_seperate_list)/len(memory_seperate_list)}")
        elif args.baseline == 'memory_qa':
            memory_qa_list.append(memory_qa_acc.get())
            print(f"Evaluation {args.baseline} accuracy: {sum(memory_qa_list)/len(memory_qa_list)}")



    if args.wandb:
        if not is_fs:
            wandb.log({"Evaluation original accuracy": sum(initial_acc_list)/len(initial_acc_list)})
            if args.baseline == 'memory':
                wandb.log({"Evaluation accuracy": sum(current_acc_list)/len(current_acc_list)})
            elif args.baseline == 'random_topk':
                wandb.log({"Evaluation random_topk accuracy": sum(random_topk_acc_list)/len(random_topk_acc_list)})
            elif args.baseline == 'random_random':
                wandb.log({"Evaluation random_random accuracy": sum(random_random_acc_list)/len(random_random_acc_list)})
            elif args.baseline == 'memory_random':
                wandb.log({"Evaluation random_permutation accuracy": sum(random_permutation_acc_list)/len(random_permutation_acc_list)})
            elif args.baseline == 'all':
                wandb.log({"Evaluation extended accuracy": sum(extended_acc_list)/len(extended_acc_list)})
            elif args.baseline == 'heuristic':
                wandb.log({"Evaluation heuristic accuracy": sum(heuristic_acc_list)/len(heuristic_acc_list)})
            elif args.baseline == 'memory_seperate':
                wandb.log({"Evaluation memory_seperate accuracy": sum(memory_seperate_list)/len(memory_seperate_list)})
            elif args.baseline == 'memory_qa':
                wandb.log({"Evaluation memory_qa accuracy": sum(memory_qa_list)/len(memory_qa_list)})

    actor_critic.to('cuda:0')

    if args.baseline == 'memory':
        return sum(current_acc_list)/len(current_acc_list)
    elif args.baseline == 'random_topk':
        return sum(random_topk_acc_list)/len(random_topk_acc_list)
    elif args.baseline == 'random_random':
        return sum(random_random_acc_list)/len(random_random_acc_list)
    elif args.baseline == 'memory_random':
        return sum(random_permutation_acc_list)/len(random_permutation_acc_list)
    elif args.baseline == 'all':
        return sum(extended_acc_list)/len(extended_acc_list)
    elif args.baseline == 'heuristic':
        return sum(heuristic_acc_list)/len(heuristic_acc_list)
    elif args.baseline == 'memory_seperate':
        return sum(memory_seperate_list)/len(memory_seperate_list)
    elif args.baseline == 'memory_qa':
        return sum(memory_qa_list)/len(memory_qa_list)

