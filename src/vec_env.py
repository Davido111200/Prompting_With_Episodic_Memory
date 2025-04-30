import numpy as np
import gym
from gym import spaces
from typing import List
import random
import torch
import warnings
import time
from tqdm import tqdm
import sys
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import wandb
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import pandas as pd
from rank_bm25 import BM25Okapi

from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from scipy.spatial import distance
from datasets import Dataset

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


from .utils import generate_instructions, \
        get_senemb_embbeddings, parse_instructions, \
            memory_action_epsilon_greedy, get_state_embds, get_permutations_dict, \
            construct_prompt, eval_accuracy, fill_template, get_llama_embeddings, \
            memory_verbalizer_epsilon_greedy, get_embedding, get_ic_examples, get_random_examples,\
            get_mistral_embeddings, process_data_multiple_fields, memory_all_verbalizer_epsilon_greedy, \
            get_ic_examples_imbalance, get_ic_examples_furthest, get_ic_examples_bm25, get_balanced_examples
from .reward_llm import get_score, get_embeddings, get_score_qa
from .init_prompt import generate_template_prompts
from .prompt_template import ZeroshotActionPromptTemplate
from .sample_memory import Memory
from promptsource.templates import DatasetTemplates

warnings.filterwarnings("ignore")

class PromptEnv(gym.Env):
    def __init__(
        self, 
        params, 
        args, 
        reward_llm, 
        reward_tokenizer, 
        train_examples, 
        train_labels, 
        valid_examples, 
        valid_labels,
        temperature, 
        top_p, 
        n_candidates, 
        n_slots, 
        epsilon, 
        num_processes, 
        i, 
        gpu_id, 
        evaluate, 
        external_memory, 
        external_memory_random,
        external_memory_extended, 
        memory_verbalizer, 
        memory_verbalizer_random, 
        memory_verbalizer_extended, 
        ic_examples, 
        ic_labels, 
        model, 
        tokenizer
    ):
        super(PromptEnv, self).__init__()

        self.params = params
        self.args = args
        self.batch_size = args.batch_size
        self.n_candidates = n_candidates
        self.n_slots = n_slots
        self.obs_size = args.obs_size
        self.num_examples = args.num_examples
        self.max_num_edits = args.max_num_edits
        self.epsilon = epsilon
        self.epsilon_order = 0.2  # especially for the order memory
        self.temperature = temperature
        self.top_p = top_p
        self.observation_space = spaces.Box(-np.inf, np.inf, ((self.obs_size,)))  # (1024 * (3 + 1))
        self.num_processes = num_processes
        self.i = i
        self.gpu_id = gpu_id
        self.evaluate = evaluate
        self.action_space = spaces.Discrete(n_candidates + 1)  # number of new candidates + 1 for no edit
        self.n_actions_ins = n_candidates + 1
        self.reward_llm, self.reward_tokenizer = reward_llm, reward_tokenizer
        self.steps = 0
        self.rew_scale = 100
        self.idxs = None
        self.episodes = 0
        self.init_eps = 1.0
        self.final_eps = 0.0001
        self.num_predict_tokens = 25
        self.model = model
        self.tokenizer = tokenizer
        self.n_pred_debug = 0


        if args.dataset in ['cr']:
            self.all_prompt_templates = DatasetTemplates('glue/sst2')
        elif args.dataset in ['mteb/twitter']:
            self.all_prompt_templates = DatasetTemplates('tweet_eval/sentiment')
        else:
            self.all_prompt_templates = DatasetTemplates(args.dataset)
        self.prompt_template_keys = self.all_prompt_templates.all_template_names

        for key in self.prompt_template_keys:
            answer_lists = self.all_prompt_templates[key].answer_choices.split("|||")
            for promt_answer, correct_answer in zip(answer_lists, params['inv_label_dict'].keys()):
                self.all_prompt_templates[key].jinja = self.all_prompt_templates[key].jinja.replace(promt_answer.strip(), correct_answer)

        if not evaluate:
            self.train_sentences = train_examples
            self.train_labels = train_labels
        else:
            self.train_sentences = valid_examples
            self.train_labels = valid_labels

        # If baseline is 'all', check the number of actions to get balanced ic_example pool set
        if args.baseline == 'all':
            if args.n_actions == 360:
                # get 6 examples from ic_examples, with 3 for each label
                self.ic_examples_, self.ic_labels_ = get_balanced_examples(ic_examples, ic_labels, n_examples=6)
            elif args.n_actions == 1680:
                self.ic_examples_, self.ic_labels_ = get_balanced_examples(ic_examples, ic_labels, n_examples=8)
            self.n_pool = len(self.ic_examples_)
        else:
            self.ic_examples_ = ic_examples
            self.ic_labels_ = ic_labels
            self.n_pool = len(self.ic_examples_)


        n_labels = len(set(self.ic_labels_))
        self.data_by_label = {i: [] for i in range(n_labels)}
        for data, label in zip(self.ic_examples_, self.ic_labels_):
            self.data_by_label[label].append(data)


        # NOTE: this is the part where we process datasets
        # might need a better way to do this 
        if params['dataset'] in ['super_glue/rte', 'snli']:
            ic_concat_list = [d['premise'] + d['hypothesis'] for d in self.ic_examples_]
            self.train_concat_list = [d['premise'] + d['hypothesis'] for d in self.train_sentences]
        elif params['dataset'] in ['glue/mrpc']:
            ic_concat_list = [d['text1'] + d['text2'] for d in self.ic_examples_]
            self.train_concat_list = [d['text1'] + d['text2'] for d in self.train_sentences]
        elif params['dataset'] in ['glue/qnli']:
            ic_concat_list = [d['question'] + d['sentence'] for d in self.ic_examples_]
            self.train_concat_list = [d['question'] + d['sentence'] for d in self.train_sentences]
        elif params['dataset'] in ['glue/mnli']:
            ic_concat_list = [d['premise'] + d['hypothesis'] for d in self.ic_examples_]
            self.train_concat_list = [d['premise'] + d['hypothesis'] for d in self.train_sentences]
        elif params['dataset'] == 'super_glue/boolq':
            ic_concat_list = [d['question'] + d['passage'] for d in self.ic_examples_]
            self.train_concat_list = [d['question'] + d['passage'] for d in self.train_sentences]
        elif params['dataset'] == 'glue/qqp':
            ic_concat_list = [d['question1'] + d['question2'] for d in self.ic_examples_]
            self.train_concat_list = [d['question1'] + d['question2'] for d in self.train_sentences]
        else:
            ic_concat_list = self.ic_examples_
            self.train_concat_list = self.train_sentences


        if args.type_embd == 'llama':    
            self.pool_sentences_embeddings = [list(get_llama_embeddings(ts, model, tokenizer)) for ts in ic_concat_list]
            self.train_sentences_embds = [list(get_llama_embeddings(ts, model, tokenizer)) for ts in self.train_concat_list]
        else:
            self.pool_sentences_embeddings = get_senemb_embbeddings(self.ic_examples_, model=model)
            self.train_sentences_embds = get_senemb_embbeddings(self.train_sentences, model=model)


        if self.args.use_bm25:
            print("Using BM25")
            tokenized_ic_corpus = [doc.split(" ") for doc in ic_concat_list]
            bm25_ic = BM25Okapi(tokenized_ic_corpus)
            
            self.ic_examples_indicies_training = [get_ic_examples_bm25(self.args, bm25_ic, sentence, self.ic_labels_) for sentence in self.train_sentences]
            if self.evaluate:
                self.ic_examples_indicies_testing = [get_ic_examples_bm25(self.args, bm25_ic, sentence, self.ic_labels_) for sentence in self.train_sentences]
        else:
            if self.args.imbalanced_labels:
                print("IMBALANCED LABELS")
                self.ic_examples_indicies_training = [get_ic_examples_imbalance(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_) for i in range(len(self.train_sentences_embds))]
                if self.evaluate:
                    self.ic_examples_indicies_testing = [get_ic_examples_imbalance(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_) for i in range(len(self.train_sentences_embds))]
            else:
                self.ic_examples_indicies_training_reversed = [get_ic_examples_furthest(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_) for i in range(len(self.train_sentences_embds))]
                if self.evaluate:
                    self.ic_examples_indicies_testing_reversed = [get_ic_examples_furthest(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_) for i in range(len(self.train_sentences_embds))]

                # NOTE: for now, just pretend that we select the furthest neighbor for ALL sentences
                # the reason for these lines is we might want a query-specific selection
                if self.args.furthest_neighbor:
                    print("FURTHEST NEIGHBOR")
                    self.ic_examples_indicies_training = self.ic_examples_indicies_training_reversed
                    if self.evaluate:
                        self.ic_examples_indicies_testing = self.ic_examples_indicies_testing_reversed
                else:
                    self.ic_examples_indicies_training = [get_ic_examples(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_, n_examples=self.args.n_slots) for i in range(len(self.train_sentences_embds))]
                    if self.evaluate:
                        self.ic_examples_indicies_testing = [get_ic_examples(self.args, self.train_sentences_embds[i], self.pool_sentences_embeddings, self.ic_labels_, n_examples=self.args.n_slots) for i in range(len(self.train_sentences_embds))]


        self.n_actions_memory = 1
        for i in range(0, self.n_slots):
            self.n_actions_memory *= (self.n_slots - i)

        self.instruction_pool = self.load_instructions(data_dir="/home/s223540177/dai/RLforLLM/src/instructions.csv")
        self.instructions = [self.params['prompt_prefix'] for _ in range(self.n_actions_ins)]

        self.memory_instance = external_memory
        self.memory_instance_random = external_memory_random
        self.memory_instance_extended = external_memory_extended
        self.memory_verbalizer_instance = memory_verbalizer
        self.memory_verbalizer_instance_random = memory_verbalizer_random
        self.memory_verbalizer_instance_extended = memory_verbalizer_extended

        if self.args.n_slots == 4:
            self.all_permutations = get_permutations_dict([0, 1, 2, 3], self.n_slots)
        elif self.args.n_slots == 2:
            self.all_permutations = get_permutations_dict([0, 1], 2)
        elif self.args.n_slots == 6:
            self.all_permutations = get_permutations_dict([0, 1, 2, 3, 4, 5], 6)
        elif self.args.n_slots == 8:
            self.all_permutations = get_permutations_dict([0, 1, 2, 3, 4, 5, 6, 7], 8)
        
        # base on the number of actions in memory, we can get the corresponding permutation
        if self.args.n_actions == 360:
            # 6*5*4*3 -> pool = 6
            self.all_permutations_extended = get_permutations_dict([_ for _ in range(6)], self.n_slots)
        elif self.args.n_actions == 1680:
            # 8*7*6*5 -> pool = 8
            self.all_permutations_extended = get_permutations_dict([_ for _ in range(8)], self.n_slots)


        self.embedding_prepared = torch.tensor(np.array([False])).share_memory_()

        if args.type_embd == 'senemb':
            self.senemb_model = model
        elif args.type_embd == 'llama':
            self.model = model
            self.tokenizer = tokenizer


        if self.args.visualize:
            self.iteration = 0
            self.df = {}
            self.permutation_counts = {i: 0 for i in range(self.n_actions_memory)}
            self.visual_text = []
            self.visual_embd = []
            self.df['text'] = self.train_concat_list
            self.df['embds'] = self.train_sentences_embds
            self.df['permutation'] = []
            self.df['permutation_index'] = []
            self.df['label'] = []


    def reset(self):
        """
        Return initial observation
        # TODO: this is for training purpose only. For evaluation, we need to reset the environment differently
        """
        self.steps = 0
        self.preds, self.labels, self.edited_scores, = [], [], []
        self.random_topk_preds, self.random_topk_labels, self.random_topk_scores = [], [], []
        self.heuristic_preds, self.heuristic_labels, self.heuristic_scores = [], [], []
        self.random_random_preds, self.random_random_labels, self.random_random_scores = [], [], []
        self.random_permutation_preds, self.random_permutation_labels, self.random_permutation_scores = [], [], []
        self.extended_preds, self.extended_labels, self.extended_scores = [], [], []
        self.qa_preds, self.qa_labels, self.qa_scores = [], [], []
        self.scores = []
        self.rewards = []
        self.rewards_extended = []
        self.random_topk_rewards = []
        self.random_random_rewards = []
    
        self.previous_permutation = []
        

        # subset_idxs is the indicies of the training sentences for this subset
        if self.idxs is not None and self.evaluate:
            self.subset_size = self.idxs.shape[0]
            subset_idxs = self.idxs
        else:
            self.subset_size = self.num_processes
            subset_idxs = np.random.choice(np.arange(len(self.train_sentences)), self.subset_size, replace=True)
        self.subset_idxs = subset_idxs
        
        
        # initialize the instruction for the subset
        self.initial_instruction = self.params['prompt_prefix']

        
        # self.terminate is the done flag for each process
        self.terminate = [False for _ in range(self.subset_size)]
        self.subset_sentences = [self.train_concat_list[int(i)] for i in subset_idxs]
        self.subset_labels = [self.train_labels[int(i)] for i in subset_idxs]
        self.episodes += 1
        

        if self.args.type_embd == 'senemb':
            self.train_sentences_embds_memory = [self.senemb_model.encode(i, convert_to_tensor=True) for i in self.subset_sentences]
            self.train_sentences_embds_memory = torch.stack(self.train_sentences_embds_memory).cpu().numpy()
        elif self.args.type_embd == 'llama':
            self.train_sentences_embds_memory = [torch.tensor(get_llama_embeddings(i, self.model, self.tokenizer)) for i in self.subset_sentences]
            self.train_sentences_embds_memory = torch.stack(self.train_sentences_embds_memory).cpu().numpy()

        # get the score for instruction + test-only. This is to compare with the score after editing
        self.current_scores, self.init_preds, self.init_labels = [], [], []
        self.current_scores = [0 for _ in range(len(self.train_sentences))]
        self.init_preds = [0 for _ in range(len(self.train_sentences))]
        self.init_labels = [0 for _ in range(len(self.train_sentences))]

        self.initial_acc = 0

        return self.train_sentences_embds_memory
    

    def step(self, action):
        # NOTE: this action is acutally for selecting the instrucion
        action = action.squeeze(-1)

        # With the actions given by RL agent, formulate the instructions
        # First we get the in-context examples permutations from memory
        # To get the in-context examples, we get the states and look up in our memory
        self.state_embds = self.train_sentences_embds_memory

        for idx, (subset_idx, state_embd, train_sentence_embd, act) in enumerate(zip(self.subset_idxs, self.state_embds, self.train_sentences_embds_memory, action)):
            if self.args.baseline == 'all':
                self.all_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
            elif self.args.baseline == 'memory':
                # print("Current steps: ", self.steps)
                self.permutation = self.memory_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
            elif self.args.baseline == 'memory_seperate':
                self.memory_seperate_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
            elif self.args.baseline == 'random_random':
                self.random_random_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
                if not self.evaluate:
                    print("Random_random baseline was called --> evaluate one and terminate!")
                    quit()
            elif self.args.baseline == 'random_topk':
                self.random_topk_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
                if not self.evaluate:
                    print("Random_topk was called --> evaluate one and terminate!")
                    quit()
            elif self.args.baseline == 'heuristic':
                self.heuristic_baseline(idx, subset_idx, state_embd, train_sentence_embd, act)
                if not self.evaluate:
                    print("Heuristic was called --> evaluate one and terminate!")
                    quit()
            elif self.args.baseline == 'memory_random':
                self.memory_random_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
            elif self.args.baseline == 'memory_qa':
                self.qa_memory_baseline_function(idx, subset_idx, state_embd, train_sentence_embd, act)
            else:
                raise f"{self.args.baseline} is not implemented yet!"
        # self.edited_prompt_embds = np.concatenate(self.edited_prompt_embds, axis=0)
                
        # End a step for a batch of episodes
        self.steps += 1

        # print("Current reward: ", self.rewards)
        # print("=====")
        
        if self.steps >= self.args.max_num_edits - 1:
            done = np.ones(self.subset_size)
            if self.evaluate:
                if self.args.baseline == 'memory':
                    info = {'episode_r': self.rewards, 'initial_acc': self.initial_acc, 'current_acc': self.subset_acc}
                elif self.args.baseline == 'memory_seperate':
                    info = {'episode_r': self.rewards, 'initial_acc': self.initial_acc, 'memory_seperate_acc': self.subset_acc}
                elif self.args.baseline == 'all':
                    info = {'episode_r': self.rewards_extended, 'initial_acc': self.initial_acc, 'extended_acc': self.extended_acc}
                elif self.args.baseline == 'memory_random':
                    info = {'episode_r': self.rewards, 'initial_acc': self.initial_acc, 'random_permutation_acc': self.random_permutation_acc}
                elif self.args.baseline == 'random_topk':
                    info = {'initial_acc': self.initial_acc, 'random_topk_acc': self.random_topk_acc}
                elif self.args.baseline == 'random_random':
                    info = {'initial_acc': self.initial_acc, 'random_random_acc': self.random_random_acc}
                elif self.args.baseline == 'heuristic':
                    info = {'initial_acc': self.initial_acc, 'heuristic_acc': self.heuristic_acc}
                elif self.args.baseline == 'memory_qa':
                    info = {'initial_acc': self.initial_acc, 'episode_r': self.rewards, 'memory_qa_acc': self.qa_memory_acc}
            else:
                if self.args.baseline == 'memory':
                    info = {'episode_r': self.rewards}
                elif self.args.baseline == 'all':
                    info = {'episode_r': self.rewards_extended}
                elif self.args.baseline == 'memory_seperate':
                    info = {'episode_r': self.rewards}
                elif self.args.baseline == 'memory_random':
                    info = {'episode_r': self.rewards}
                elif self.args.baseline == 'memory_qa':
                    info = {'episode_r': self.rewards}
                else:
                    info = {}
            if self.args.wandb:
                if not self.evaluate:
                    wandb.log({'Initial accuracy': self.initial_acc})
                    if self.args.baseline == 'memory':
                        wandb.log({'Memory accuracy': self.subset_acc})    
                        wandb.log({'reward': self.rewards})
                    elif self.args.baseline == 'memory_seperate':
                        wandb.log({'Memory seperate accuracy': self.subset_acc})
                    elif self.args.baseline == 'memory_random':
                        wandb.log({'Random permutation accuracy': self.random_permutation_acc})
                        wandb.log({'reward': self.rewards})
                    elif self.args.baseline == 'random_topk':   
                        wandb.log({'Random topk accuracy': self.random_topk_acc})
                    elif self.args.baseline == 'random_random':
                        wandb.log({'Random random accuracy': self.random_random_acc})
                    elif self.args.baseline == 'all':
                        wandb.log({'Extended accuracy': self.extended_acc})
                        wandb.log({'reward': self.rewards_extended})
                    elif self.args.baseline == 'heuristic':
                        wandb.log({'Heuristic accuracy': self.heuristic_acc})
                    elif self.args.baseline == 'memory_qa':
                        wandb.log({'QA memory accuracy': self.qa_memory_acc})
                    wandb.log({'Epsilon': self.epsilon})
        else:
            done = np.zeros(self.subset_size)
            if self.args.baseline == 'memory':
                info = {'episode_r': self.rewards} 
            elif self.args.baseline == 'all':
                info = {'episode_r': self.rewards_extended}
            elif self.args.baseline == 'memory_random':
                info = {'episode_r': self.rewards}
            elif self.args.baseline == 'memory_qa':
                info = {'episode_r': self.rewards}
            else:
                info = {}
        if self.args.baseline != 'all':
            return self.state_embds, self.rewards, done, info
        else:
            return self.state_embds, self.rewards_extended, done, info

    
    def memory_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: this is for the prompts with memory
        ic_permutation_index, self.epsilon = memory_action_epsilon_greedy(self.evaluate, self.memory_instance, train_sentence_embd, subset_idx, \
                                                            self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                            init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k, n_actions=self.args.n_actions)

        ic_permutation = self.all_permutations[ic_permutation_index]
        # get the in-context examples
        if self.evaluate:
            self.ic_examples_indicies = self.ic_examples_indicies_testing[subset_idx]
        else:
            self.ic_examples_indicies = self.ic_examples_indicies_training[subset_idx]

        self.ic_examples = [self.ic_examples_[int(i)] for i in self.ic_examples_indicies]
        self.ic_labels = [self.ic_labels_[int(i)] for i in self.ic_examples_indicies]

        ic_permutation_examples = [self.ic_examples[int(i)] for i in ic_permutation]
        ic_permutation_labels = [self.ic_labels[int(i)] for i in ic_permutation]

        # after having the ic-examples, we formulate the prompt
        if self.args.use_verbalizer:
            verbalizer_index = self.args.verbalizer_index
            self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
            prompt = construct_prompt(self.params, self.instructions[int(act)], ic_permutation_examples, ic_permutation_labels, self.train_sentences[int(subset_idx)], self.prompt_template)
        else:
            prompt = construct_prompt(self.params, self.instructions[int(act)], ic_permutation_examples, ic_permutation_labels, self.train_sentences[int(subset_idx)], None)          


        edited_score, prompt_embd, edited_pred_class, label = get_score(self.params, self.reward_llm, self.reward_tokenizer, prompt, \
                                                                        self.train_labels[subset_idx], num_predict_tokens=1,\
                                                                        gpu_id=self.gpu_id)


        # if self.args.debug_mode:
        #     print("Prompt: ", prompt)
        #     print("Params label dict: ", self.params['label_dict'])
        #     print(f"Pred: {edited_pred_class} - Label: {label}")
        #     print("Score: ", edited_score.item())
        #     print("TEMPLATE NAME: ", self.prompt_template_keys)
        #     print("Selected permutation: ", ic_permutation)
        #     print("=================================")
        #     quit()

        self.edited_scores.append(edited_score.item())
        self.preds.append(edited_pred_class)
        self.labels.append(label)

        # get accuracy for this subset
        self.subset_acc = eval_accuracy(self.preds, self.labels)

        if self.args.visualize:
            # get the count for illustration purposes
            self.permutation_counts[ic_permutation_index] += 1

        # get the reward
        if not self.evaluate:
            if self.args.reward_type == 'prob':
                reward = (edited_score.item() - self.current_scores[subset_idx]) * self.rew_scale
                self.rewards.append(reward)
                # update the memory if we are not evaluating
                self.memory_instance.update(subset_idx, train_sentence_embd, ic_permutation_index, reward)
        else:
            # if evaluate, check for visualization
            if len(self.train_sentences_embds) < 500:
                # this is the dev set
                pass
            else:
                # if we are evaluating on the test set, we can count the number of wrongly predicted examples
                if self.args.debug_mode:
                    # now for each sentence that if wrongly predicted, we permute to see if there is any permutations that help correct the prediction
                    if edited_pred_class != label:
                        self.flag = False
                        for permutation in range(len(self.all_permutations)):
                            ic_permutation = self.all_permutations[permutation]
                            # check to see if there is any permutation that can predict the test example right
                            ic_permutation_examples = [self.ic_examples[int(i)] for i in ic_permutation]
                            ic_permutation_labels = [self.ic_labels[int(i)] for i in ic_permutation]

                            # after having the ic-examples, we formulate the prompt
                            if self.args.use_verbalizer:
                                verbalizer_index = self.args.verbalizer_index
                                self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
                                prompt = construct_prompt(self.params, self.instructions[int(act)], ic_permutation_examples, ic_permutation_labels, self.train_sentences[int(subset_idx)], self.prompt_template)
                            else:
                                prompt = construct_prompt(self.params, self.instructions[int(act)], ic_permutation_examples, ic_permutation_labels, self.train_sentences[int(subset_idx)], None)          


                            edited_score_, prompt_embd, edited_pred_class_, label_ = get_score(self.params, self.reward_llm, self.reward_tokenizer, prompt, \
                                                                                            self.train_labels[subset_idx], num_predict_tokens=1,\
                                                                                            gpu_id=self.gpu_id)

                            if edited_pred_class_ == label_:
                                self.flag = True
                                break
                        if not self.flag:
                            self.n_pred_debug += 1
                    print("NUM OF WRONG PREDICTIONS: ", self.n_pred_debug)


                if self.args.visualize:
                    self.df['permutation'].append(str(ic_permutation))
                    self.df['permutation_index'].append(ic_permutation_index)
                    self.df['label'].append(label)
                    if len(self.df['permutation']) == len(self.df['text']):
                        print("VISUALIZING")
                        self.final_df = pd.DataFrame(self.df)
                        X = np.array(self.final_df['embds'].to_list(), dtype=np.float32)
                        # Change the n_components to 2 for 2D t-SNE
                        tsne = TSNE(n_components=2, random_state=0, n_iter=1000)
                        tsne_results = tsne.fit_transform(X)
                        df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
                        df_tsne['Class Name'] = self.final_df['label']
                        df_tsne['Permutation Index'] = self.final_df['permutation_index']

                        # Create a 2D scatter plot
                        fig = plt.figure(figsize=(10,8))
                        ax = fig.add_subplot(111)
                        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})

                        # Define markers based on 'Permutation Index'
                        # markers = {index: ('x' if index % 2 == 0 else 'o') for index in df_tsne['Permutation Index'].unique()}

                        # Generate a list of unique class names and assign a color to each
                        unique_classes = df_tsne['Class Name'].unique()
                        colors = sns.color_palette('hls', len(unique_classes))
                        class_color_dict = dict(zip(unique_classes, colors))

                        top_two_indices = self.final_df['permutation_index'].value_counts().nlargest(2).index
                        markers = {index: ('^' if index == top_two_indices[0] else ('s' if index == top_two_indices[1] else 'o')) for index in self.final_df['permutation_index'].unique()}
                        colors = {index: ('grey' if marker == 'o' else class_color_dict[group['Class Name'].iloc[0]]) for index, marker in markers.items()}

                        # Plot each class with the specified markers and colors
                        for (i, group) in df_tsne.groupby(['Permutation Index']):
                            group_marker = markers[i]  # Get the marker for the current group
                            group_color = colors[i]  # Get the color for the current group
                            ax.scatter(group['TSNE1'], group['TSNE2'], color=group_color, marker=group_marker, label=f'Index {i}')

                        # Plot each class with a different color/hue and marker
                        for (i, group) in df_tsne.groupby(['Permutation Index']):
                            group_colors = group['Class Name'].map(class_color_dict)  # Map the class names to colors
                            scatter = ax.scatter(group['TSNE1'], group['TSNE2'], color=group_colors, marker=markers[i[0]], label=f'Index {i[0]}')

                        # Add labels and title
                        ax.set_title('2D Scatter plot of glue/sst2 using t-SNE')
                        ax.set_xlabel('TSNE1')
                        ax.set_ylabel('TSNE2')

                        # Legend
                        legend = ax.legend(title="Class Name")
                        ax.add_artist(legend)

                        # Save the plot
                        plt.savefig(f'/home/s223540177/dai/RLforLLM/plot_figures/tsne_plot_sst2_2d_iter_{self.iteration}.pdf', format='pdf', dpi=600)                        

                        # Save the plot
                        plt.savefig(f'/home/s223540177/dai/RLforLLM/plot_figures/tsne_plot_sst2_3d_iter_{self.iteration}.pdf', format='pdf', dpi=600)

                        # Increment the iteration counter
                        self.iteration += 10

                        # Reset the data for the next visualization
                        self.df['permutation'] = []
                        self.df['permutation_index'] = []
                        self.df['label'] = []

                        print("DONE VISUALIZING")
                


    def all_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        ic_extended_permutation_index, self.epsilon = memory_action_epsilon_greedy(self.evaluate, self.memory_instance_extended, train_sentence_embd, subset_idx, \
                                                            self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                            init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k, n_actions=self.args.n_actions)

        # convert permutation index to permutation
        ic_extended_permutation = self.all_permutations_extended[ic_extended_permutation_index]


        # get the in-context examples
        self.ic_examples = [self.ic_examples_[i] for i in ic_extended_permutation]
        self.ic_labels = [self.ic_labels_[i] for i in ic_extended_permutation]

        # after having the ic-examples, we formulate the prompt
        if self.args.use_verbalizer:
            verbalizer_index = self.args.verbalizer_index
            self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
            prompt = construct_prompt(self.params, self.instructions[act], self.ic_examples, self.ic_labels, self.train_sentences[int(subset_idx)], self.prompt_template)
        else:
            prompt = construct_prompt(self.params, self.instructions[act], self.ic_examples, self.ic_labels, self.train_sentences[int(subset_idx)], None)


        edited_score, prompt_embd, edited_pred_class, label = get_score(self.params, self.reward_llm, self.reward_tokenizer, prompt, \
                                                                        self.train_labels[subset_idx], num_predict_tokens=1,\
                                                                        gpu_id=self.gpu_id)
        
        self.extended_scores.append(edited_score.item())
        self.extended_preds.append(edited_pred_class)
        self.extended_labels.append(label)

        # get accuracy for this subset
        self.extended_acc = eval_accuracy(self.extended_preds, self.extended_labels)

        # get the reward
        if not self.evaluate:
            if self.args.reward_type == 'prob':
                reward = (edited_score.item() - self.current_scores[subset_idx]) * self.rew_scale
                self.rewards_extended.append(reward)
                # update the memory if we are not evaluating
                self.memory_instance_extended.update(subset_idx, train_sentence_embd, ic_extended_permutation_index, reward)



    def memory_seperate_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: this is for the prompts with memory
        ic_permutation_index, self.epsilon = memory_action_epsilon_greedy(self.evaluate, self.memory_instance, train_sentence_embd, subset_idx, \
                                                            self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                            init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k)

        # convert permutation index to permutation
        ic_permutation = self.all_permutations[ic_permutation_index]
        # get the in-context examples
        if self.evaluate:
            self.ic_examples_indicies = self.ic_examples_indicies_testing[subset_idx]
        else:
            self.ic_examples_indicies = self.ic_examples_indicies_training[subset_idx]
        self.ic_examples = [self.ic_examples_[i] for i in self.ic_examples_indicies]
        self.ic_labels = [self.ic_labels_[i] for i in self.ic_examples_indicies]

        ic_permutation_examples = [self.ic_examples[i] for i in ic_permutation]
        ic_permutation_labels = [self.ic_labels[i] for i in ic_permutation]

        # concat the ic examples, with the test instance to get embeddings to verbalizer memory
        verbalizer_input_text = '. '.join(self.ic_examples + [self.train_sentences[subset_idx]])
        # get the embeddings
        verbalizer_embd = get_embedding(self.args, self.model, self.tokenizer, verbalizer_input_text)

        # get the verbalizer
        verbalizer_index, epsilon = memory_verbalizer_epsilon_greedy(self.evaluate, self.memory_verbalizer_instance, verbalizer_embd, subset_idx, \
                                                                     self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                                    init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k, n_verbalizers=len(self.all_prompt_templates), is_seperate=True)

        self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]

        # after having the ic-examples, we formulate the prompt
        if self.args.use_verbalizer:
            prompt = construct_prompt(self.params, self.instructions[act], ic_permutation_examples, ic_permutation_labels, self.train_sentences[subset_idx], self.prompt_template)
        else:
            prompt = construct_prompt(self.params, self.instructions[act], ic_permutation_examples, ic_permutation_labels, self.train_sentences[subset_idx], None)          

        edited_score, prompt_embd, edited_pred_class, label = get_score(self.params, self.reward_llm, self.reward_tokenizer, prompt, \
                                                                        self.train_labels[subset_idx], num_predict_tokens=1,\
                                                                        gpu_id=self.gpu_id)

        self.edited_scores.append(edited_score.item())
        self.preds.append(edited_pred_class)
        self.labels.append(label)

        # get accuracy for this subset
        self.subset_acc = eval_accuracy(self.preds, self.labels)

        # get the reward
        if not self.evaluate:
            if self.args.reward_type == 'prob':
                reward = (edited_score.item() - self.current_scores[subset_idx]) * self.rew_scale
                self.rewards.append(reward)
                # update the memory if we are not evaluating
                self.memory_instance.update(subset_idx, train_sentence_embd, ic_permutation_index, reward)
                self.memory_verbalizer_instance.update(verbalizer_embd, verbalizer_index, reward)




    def memory_random_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: this baseline gets the order memory, but randomly selects examples
        # first we select the random examples
        sel_exs, sel_labels = get_random_examples(self.ic_examples_, self.ic_labels_)

        # next we get the embeddings for them to later compare with test instance
        if self.args.type_embd == 'senemb':
            embds = self.senemb_model.encode(sel_exs)
        elif self.args.type_embd == 'llama':
            embds = []
            for ex in sel_exs:
                embd = get_llama_embeddings(ex, self.model, self.tokenizer)
                embds.append(list(embd))

        # calculate cosine similarity
        cos_sim = util.cos_sim(train_sentence_embd, embds)

        # sort the examples by the cosine similarity
        sorted_exs = [sentence for _, sentence in sorted(zip(list(cos_sim[0]), sel_exs), key=lambda x: x[0], reverse=True)]
        sorted_labels = [label for _, label in sorted(zip(list(cos_sim[0]), sel_labels), key=lambda x: x[0], reverse=True)]

        # get the permutation from order memory
        ic_permutation_index_random, self.epsilon = memory_action_epsilon_greedy(self.evaluate, self.memory_instance_random, train_sentence_embd, subset_idx, \
                                                                            self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                                                init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k)
        ic_permutation_random = self.all_permutations[ic_permutation_index_random]

        # order the examples based on the permutation
        ic_random_permutation_examples = [sorted_exs[i] for i in ic_permutation_random]
        ic_random_permutation_labels = [sorted_labels[i] for i in ic_permutation_random]

        # NOTE: for any random baseline, we use random verbalizer
        verbalizer_random_index = random.randint(0, len(self.all_prompt_templates) - 1)
        self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_random_index]]

        # after having the ic-examples, we formulate the prompt
        random_permutation_prompt = construct_prompt(self.params, self.instructions[act], ic_random_permutation_examples, ic_random_permutation_labels, self.train_sentences[subset_idx], self.prompt_template)
        random_permutation_score, random_permutation_prompt_embd, random_permutation_pred_class, random_permutation_label = \
                                                get_score(self.params, self.reward_llm, self.reward_tokenizer, random_permutation_prompt, \
                                                            self.train_labels[subset_idx], num_predict_tokens=1,\
                                                            gpu_id=self.gpu_id)
        self.random_permutation_scores.append(random_permutation_score.item())
        self.random_permutation_preds.append(random_permutation_pred_class)
        self.random_permutation_labels.append(random_permutation_label)

        # get the reward
        if self.args.reward_type == 'prob':
            reward_memory_random = (random_permutation_score.item() - self.current_scores[subset_idx]) * self.rew_scale
            self.rewards.append(reward_memory_random)
            # update the order memory if we are not evaluating
            if not self.evaluate:
                self.memory_instance_random.update(subset_idx, train_sentence_embd, ic_permutation_index_random, reward_memory_random)
            # Get accuracy for this subset
            self.random_permutation_acc = eval_accuracy(self.random_permutation_preds, self.random_permutation_labels)





    def random_random_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: random memory - random examples as well
        # NOTE: for this one, we can simply random all the examples without worrying about the order - done as in baselines
        # NOTE: if we want to change the instruction (for optimizing), we can change that later

        # randomly select 4 examples and get their labels
        random_indices = random.sample(range(len(self.ic_examples_)), 4)
        random_random_examples = [self.ic_examples_[i] for i in random_indices]
        random_random_labels_orig = [self.ic_labels_[i] for i in random_indices]
        random_random_labels = [self.params['label_dict'][i][0] for i in random_random_labels_orig]


        if self.params['dataset'] == 'glue/qnli':
            verbalizer_index = 0
            prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
            p = construct_prompt(self.params, self.instructions[act], random_random_examples, random_random_labels_orig, self.train_sentences[subset_idx], prompt_template)
        elif self.params['dataset'] == 'subj':
            p = construct_prompt(self.params, self.instructions[act], random_random_examples, random_random_labels_orig, self.train_sentences[subset_idx], None)
        elif self.params['dataset'] == 'super_glue/boolq':
            verbalizer_index = np.random.randint(0, len(self.prompt_template_keys))
            prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
            p = construct_prompt(self.params, self.instructions[act], random_random_examples, random_random_labels_orig, self.train_sentences[subset_idx], prompt_template)
        else:    
            p = fill_template(self.train_sentences[subset_idx], random_random_examples[0], random_random_labels[0], random_random_examples[1], random_random_labels[1], \
                            random_random_examples[2], random_random_labels[2], random_random_examples[3], random_random_labels[3], dataset_name=self.args.dataset)

        

        random_random_score, random_random_prompt_embd, random_random_pred_class, random_random_label = \
                                                get_score(self.params, self.reward_llm, self.reward_tokenizer, p, \
                                                            self.train_labels[subset_idx], num_predict_tokens=1,\
                                                            gpu_id=self.gpu_id)

        self.random_random_scores.append(random_random_score.item())
        self.random_random_preds.append(random_random_pred_class)
        self.random_random_labels.append(random_random_label)

        self.random_random_acc = eval_accuracy(self.random_random_preds, self.random_random_labels)




    def random_topk_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: random memory - top k examples
        random_permutation_topk_index = random.randint(0, self.n_actions_memory - 1)
        # convert random topk permutation index to permutation
        random_permutation_topk = self.all_permutations[random_permutation_topk_index]


        random_permutation_topk_examples = [self.ic_examples_[i] for i in random_permutation_topk]
        random_permutation_topk_sentences_labels = [self.ic_labels_[i] for i in random_permutation_topk]

        # Randomly select the template
        n_verbalizers = len(self.prompt_template_keys)
        random_verbalizer_idx = random.randint(0, n_verbalizers-1)
        self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[random_verbalizer_idx]]

        random_permutation_topk_prompt = construct_prompt(self.params, self.instructions[act], random_permutation_topk_examples, \
                                                        random_permutation_topk_sentences_labels, self.train_sentences[subset_idx], self.prompt_template)
        random_permutation_topk_score, random_permutation_topk_prompt_embd, random_permutation_topk_pred_class, random_permutation_topk_label = \
                                                get_score(self.params, self.reward_llm, self.reward_tokenizer, random_permutation_topk_prompt, \
                                                            self.train_labels[subset_idx], num_predict_tokens=1,\
                                                            gpu_id=self.gpu_id)      
        self.random_topk_scores.append(random_permutation_topk_score.item())
        self.random_topk_preds.append(random_permutation_topk_pred_class)
        self.random_topk_labels.append(random_permutation_topk_label)         

        self.random_topk_acc = eval_accuracy(self.random_topk_preds, self.random_topk_labels)


    def heuristic_baseline(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: this will select top_k closest sentences from the in-context example pool, then order them ascendingly/ descendingly

        if self.args.dataset in ['glue/sst2', 'imdb', 'cr', 'movie_review', 'ag_news', 'glue/cola', 'super_glue/rte', 'glue/qnli', 'super_glue/boolq']:
            assert self.evaluate, "Heuristic baseline only accept evaluation"
            
            if self.evaluate:
                self.ic_examples_indicies = self.ic_examples_indicies_testing[subset_idx]
            
            # Get the top-k in-context examples
            self.ic_examples = [self.ic_examples_[int(i)] for i in self.ic_examples_indicies]
            self.ic_labels = [self.ic_labels_[int(i)] for i in self.ic_examples_indicies]

            if self.args.heuristic_order == 'descending':
                permutation = [0,1,2,3]
            elif self.args.heuristic_order == 'ascending':
                permutation = [3,2,1,0]

            examples = [self.ic_examples[i] for i in permutation]
            labels = [self.ic_labels[i] for i in permutation]

            if self.args.use_verbalizer:
                verbalizer_index = self.args.verbalizer_index
                self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]
                prompt = construct_prompt(self.params, self.instructions[int(act)], examples, labels, self.train_sentences[int(subset_idx)], self.prompt_template)
            else:
                prompt = construct_prompt(self.params, self.instructions[int(act)], examples, labels, self.train_sentences[int(subset_idx)], None)
                
            score, prompt_embd, pred_class, label = get_score(self.params, self.reward_llm, self.reward_tokenizer, prompt, \
                                                                        self.train_labels[subset_idx], num_predict_tokens=1,\
                                                                        gpu_id=self.gpu_id)



            self.heuristic_scores.append(score.item())
            self.heuristic_preds.append(pred_class)
            self.heuristic_labels.append(label)

            self.heuristic_acc = eval_accuracy(self.heuristic_preds, self.heuristic_labels)
        else:
            raise f"{self.args.dataset} dataset is not implemented yet!"
        
        


    def qa_memory_baseline_function(self, idx, subset_idx, state_embd, train_sentence_embd, act):
        # NOTE: This marks the use of new downstream LM for QA task

        ic_permutation_index, self.epsilon = memory_action_epsilon_greedy(self.evaluate, self.memory_instance, train_sentence_embd, subset_idx, \
                                                            self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                            init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k)

        # convert permutation index to permutation
        ic_permutation = self.all_permutations[ic_permutation_index]
        # get the in-context examples
        if self.evaluate:
            self.ic_examples_indicies = self.ic_examples_indicies_testing[subset_idx]
        else:
            self.ic_examples_indicies = self.ic_examples_indicies_training[subset_idx]
        
        self.ic_examples = [self.ic_examples_all[int(i)] for i in self.ic_examples_indicies]
        self.ic_labels = [self.ic_labels_[i] for i in self.ic_examples_indicies]


        ic_permutation_examples = [self.ic_examples[i] for i in ic_permutation]
        ic_permutation_labels = [self.ic_labels[i] for i in ic_permutation]

        # get the verbalizer
        verbalizer_index, epsilon = memory_verbalizer_epsilon_greedy(self.evaluate, self.memory_verbalizer_instance, train_sentence_embd, subset_idx, \
                                                                     self.episodes, self.args.num_env_steps // self.args.num_processes,\
                                                                    init_epsilon=self.init_eps, final_epsilon=self.final_eps, top_k=self.args.memory_k, n_verbalizers=len(self.all_prompt_templates), is_seperate=False)

        self.prompt_template = self.all_prompt_templates[self.prompt_template_keys[verbalizer_index]]


        # if self.evaluate:
            # if isinstance(self.train_sentences_all, Dataset):
                # pass
            # else:
                # self.train_sentences_all = Dataset.from_dict(self.train_sentences_all)

        prompt = construct_prompt(self.params, self.instructions[act], ic_permutation_examples, ic_permutation_labels, self.train_sentences_all[int(subset_idx)], self.prompt_template)
        edited_score, answer, label = get_score_qa(self.params, self.reward_llm, self.reward_tokenizer, prompt, self.train_labels[subset_idx], num_predict_tokens=self.num_predict_tokens)

        self.qa_scores.append(edited_score)
        self.qa_preds.append(answer)
        self.qa_labels.append(label)

        self.qa_memory_acc = eval_accuracy(self.qa_preds, self.qa_labels)

        if not self.evaluate:
            if self.args.reward_type == 'prob':
                reward = (edited_score - self.current_scores[subset_idx]) * self.rew_scale
                self.rewards.append(reward)
                # update the memory if we are not evaluating
                self.memory_instance.update(subset_idx, train_sentence_embd, ic_permutation_index, reward)
                self.memory_verbalizer_instance.update(subset_idx, train_sentence_embd, verbalizer_index, reward)
                # self.memory_verbalizer_instance.update(verbalizer_embd, verbalizer_index, reward)








    def write_instructions(self, save_dir):
        """
        NOTE: execute this function once to fill in csv file
        """
        import csv
        llm_response = generate_instructions(params=self.params, model=self.action_llm, tokenizer=self.action_tokenizer, max_new_tokens=4096)    
        parse_instructions(llm_response, save_dir=save_dir)


    def load_instructions(self, data_dir):
        """
        This function loads instructions from csv file
        return: list of instructions
        """
        import csv
        instructions = []
        with open(data_dir, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx != 0:
                    instruction = row[0]
                    if '"' in instruction:
                        instruction = instruction.replace('"', '')
                    if "</s>" in instruction:
                        instruction = instruction.replace("</s>", "")
                    instructions.append(instruction)
        return instructions




