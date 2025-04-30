import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from dataclasses import dataclass
from typing import Optional, Tuple

# from .fastchat_helper import get_model_answers

import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import wandb

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, util


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


def eval_accuracy(all_preds, all_labels):
    """
    Evaluate accuracy
    """
    assert len(all_preds) == len(all_labels), "Length of all_preds and all_labels must be the same"
    correct = 0

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            correct += 1
    
    return round(correct / len(all_preds), 2)
    

def plot_mean_and_variance_from_pkls(filepaths, save_path, method_name, care=False):
    import pickle
    assert isinstance(filepaths, list), "filepaths must be a list"
    save_path += f"{method_name}.png"

    data_list = []

    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)


    # Calculate mean and variance
    means = np.mean(data_list, axis=0)
    variances = np.var(data_list, axis=0)

    y_min = 0
    y_max = 1 

    plt.plot(means, label='Mean')
    plt.fill_between(range(len(means)), np.array(means) - np.sqrt(variances), np.array(means) + np.sqrt(variances), alpha=0.5, color='gray', label='Variance')
    plt.xlabel("Steps")
    plt.ylabel("Mean Accuracy over {} runs".format(len(data_list)))
    plt.ylim(y_min, y_max)
    plt.title("Accuracy of {}".format(method_name))
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')  # 'bbox_inches' ensures the entire plot is saved




def parse_instruction_from_examples_hand_examples(params, sentence):
    """
    This sentences receive the prompt and parse into instruction and example parts
    NOTE: This function is only used for hand-written examples
    """
    if not isinstance(sentence, list):
        sentence = [sentence]
    ins, ex = sentence[0].split(params["q_prefix"], 1)
    ins = ins.rstrip()
    ex = ex.lstrip()

    # Concat back the q_prefix to the beginning of examples
    ex = params["q_prefix"] + ex

    # return texts
    return ins, ex
    

def process_data_multiple_fields(params, data):
    """
    This function receives the data and processes it into a list of sentences and a list of labels
    """
    sentences = []
    labels = []
    
    if params['dataset'] == 'super_glue/rte':
        for d in data:
            dict_data = {'premise': d['premise'], 'hypothesis': d['hypothesis'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
    else:
        raise ValueError("Dataset not supported")
    
    return sentences, labels


def convert_to_dicts(args, data):
    """
    This function receives the data and converts it into a list of dictionaries
    """
    sentences = []
    labels = []
    for d in data:
        if args.dataset in ['super_glue/rte', 'glue/mnli', 'snli']:
            dict_data = {'premise': d['premise'], 'hypothesis': d['hypothesis'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
        elif args.dataset == 'glue/mrpc':
            dict_data = {'text1': d['text1'], 'text2': d['text2'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
        elif args.dataset == 'super_glue/boolq':
            dict_data = {'passage': d['passage'], 'question': d['question'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
        elif args.dataset == 'glue/qnli':
            dict_data = {'question': d['question'], 'sentence': d['sentence'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
        elif args.dataset == 'glue/qqp':
            dict_data = {'question1': d['question1'], 'question2': d['question2'], 'label': d['label']}
            sentences.append(dict_data)
            labels.append(d['label'])
        else:
            raise ValueError("Dataset not supported")
    
    return sentences, labels


def convert_to_dicts_qa(data, dataset_name):
    """
    This function receives the data and converts it into a list of dictionaries
    """
    sentences = []
    labels = []
    if dataset_name == 'coqa':
        stories = data['story']
        questions = data['questions']
        answers = data['answers']
        for i in range(len(stories)):
            dict_data = {'story': stories[i], 'questions': questions[i], 'label': answers[i]}
            sentences.append(dict_data)
            labels.append(data['answers'][i])
    else:
        raise ValueError("Dataset not supported")
    
    return sentences, labels


def select_k_samples_per_class(args, new_samples, new_labels, k, seed):
    """
    Randomly select k samples from each class in new_labels.
    :param new_samples: List of samples.
    :param new_labels: List of corresponding labels (integer values).
    :param k: Number of samples to select for each class.
    :return: Two lists: selected samples and their corresponding labels.
    """
    np.random.seed(seed)
    random.seed(seed)
    unique_labels = np.unique(new_labels)
    selected_samples = []
    selected_labels = []


    if args.dataset == 'math_qa':
        # randomly select n_samples samples 
        labels = set(new_labels)
        n_samples = k * len(labels)
        selected_indices = random.sample(range(len(new_samples)), n_samples)
        selected_samples = [new_samples[i] for i in selected_indices]
        selected_labels = [new_labels[i] for i in selected_indices]

    else:
        for label in unique_labels:
            class_indices = np.where(new_labels == label)[0]
            if k >= len(class_indices):
                selected_indices = class_indices
            else:
                selected_indices = random.sample(list(class_indices), k)

            selected_samples.extend([new_samples[int(i)] for i in selected_indices])
            selected_labels.extend([label] * k)

    return selected_samples, selected_labels
    
def select_n_random_samples(samples, labels, n):
    """
    Randomly select n samples from the given samples and labels.
    :param samples: List of samples.
    :param labels: List of corresponding labels (integer values).
    :param n: Number of samples to randomly select.
    :return: Two lists: selected samples and their corresponding labels.
    """
    if n >= len(samples):
        selected_indices = list(range(len(samples)))
    else:
        selected_indices = random.sample(range(len(samples)), n)

    selected_samples = [samples[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    return selected_samples, selected_labels

def normalize_rewards(rewards):
    """
    z-score normalization for reward
    """
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    if std_reward > 1e-6:
        normalized_rewards = (rewards - mean_reward) / std_reward
    else:
        normalized_rewards = rewards - mean_reward
    
    return normalized_rewards.tolist()


def get_instructions(params, sentence):
    import re
    """
    This function receives the response of Action LLM and parses it into n_candidates examples.
    """
    n_candidates = params["n_candidates"]

    # Split the sentence into instructions using the pattern "Instruction" followed by a digit
    # instructions = re.split(r'Instruction\s*\d+:', sentence)[1:n_candidates+1]
    _, r = sentence.split("[/INST]")
    if "</s>" in r:
        response, _ = r.split("</s>")
    else: 
        response = r
    response = response.strip()

    # Use regular expression to extract content after "Instruction [number]:"
    instructions = re.findall(r'Instruction(?: \d+)?: (.+)', response)

    # Extract content after ":" for each instruction
    res = [instruction.strip() for instruction in instructions]

    # Check if the number of instructions matches n_candidates
    if len(res) != n_candidates:
        print("*** SOS ***")
        print("RESPONSE: ", r)
        if len(res) > n_candidates:
            print("EXTRA")
            print(f"Number of instruction candidates ({len(res)}) is bigger than {n_candidates}")
            print("ALL RES: ", res)
            # quit()
            res = res[0:n_candidates]
        else:
            while len(res) < n_candidates:
                print("Candidate not enough - adding initial instructions")
                res.append(params["prompt_prefix"])

        print("Parsed instructions:", res)
    return res


def convert_permutations(num_examples, num_slots):
    """
    This function returns a dictionary the permutations of the slots, given num_slots and num_examples
    """
    permutations = []
    num_possible_permutations = num_examples ** num_slots
    for i in range(num_possible_permutations):
        for j in range(num_possible_permutations):
            for k in range(num_possible_permutations):
                for l in range(num_possible_permutations):
                    permutations.append([i, j, k, l])

    actions = []
    for permutation in permutations:
        action = 0
        for i in range(num_slots):
            action += permutation[i] * (num_examples ** i)
        actions.append(action)

    permutation_to_action_dict = dict(zip(actions, permutations))

    return permutation_to_action_dict
    

def get_senemb_embbeddings(sentences, model):
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


def get_llama_embeddings(inputs, model, tokenizer):
    from angle_emb import Prompts, AnglE
    # def decorate_text(text: str):
    #     return Prompts.A.format(text=text)
    
    # tok = tokenizer([decorate_text(inputs)], return_tensors='pt')
    # for k, v in tok.items():
    #     tok[k] = v.cuda()
    # vec = model(output_hidden_states=True, **tok).hidden_states[-1][:, -1].float().detach().cpu()[0]
        
    # return vec
    assert type(inputs) == str, ValueError("Input must be a string")

    vec = model.encode({'text': inputs}, to_numpy=True)

    return vec[0]

# def get_llama_multiple_embeddings(inputs, model, tokenizer):
#     for i in inputs:

def get_mistral_embeddings(inputs, model):
    embeddings = model.encode(inputs)
    return embeddings


def generate_instructions(params, model, tokenizer, max_new_tokens= 1000):
    from vllm import LLM, SamplingParams
    prompt = [f"You are given this instruction: {params['prompt_prefix']}. Your job is to paraphrase that and generate 50 different and diverse instructions.\
              Each of the instruction should start with 'Instruction: '."]
    
    prompt_template = "mistral"

    response = get_model_answers(
        model=model,
        tokenizer=tokenizer,
        qs=str(prompt),
        temperature=0.0,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        prompt_template=prompt_template
    )
    
    return response

def generate_vllm(llm):
    from vllm import SamplingParams, LLM
    # llm = LLM(model='mistralai/Mixtral-8x7B-Instruct-v0.1', trust_remote_code=True, seed=1, dtype='float16')
    prompt = ["Introduce yourself"]
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

    outputs = llm.generate(prompt, sampling_params=sampling_params)
    print("OUTPUTS: ", outputs)

    for output in outputs: 
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    

def parse_instructions(response, save_dir):
    import re
    import csv
    pattern = re.compile(r'\d+\.\sInstruction:\s(.*)')

    # Find all matches in the text
    matches = pattern.findall(response)

    with open(save_dir, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Instruction'])
        writer.writerows(zip(matches))


def memory_action_epsilon_greedy(evaluate, memory_instance, current_state_embd, subset_idx, current_update_episode, total_update_episode, init_epsilon, final_epsilon, top_k, n_actions):
    """
    This function returns the index of max value in value_list if exploit, otherwise return a random index
    current_update_episode: the current episode of the update
    total_update_episode: the total number of episodes for the update
    """
    
    assert init_epsilon >= final_epsilon, "init_epsilon must be larger than final_epsilon"
    if not evaluate:
        assert current_update_episode <= total_update_episode, "current_eps must be smaller than total_eps"
    assert current_update_episode >= 0, "current_eps must be larger than 0"
    assert total_update_episode >= 0, "total_eps must be larger than 0"

    # Calculate the epsilon for this step
    epsilon = init_epsilon - (init_epsilon - final_epsilon) * current_update_episode / total_update_episode

    # NOTE: changed this to always act greedily when evaluating
    if evaluate:
        # if the state is not in memory, take top_k values
        # make sure the training memory is filled
        if not any(np.array_equal(current_state_embd, val) for val in memory_instance.states.values()):
                # # if any element of memory_instance.states is an empty list, we randomly select an action
                # if any(s == [] for s in memory_instance.states.values()):
                #     return random.randint(0, n_actions - 1), epsilon
                # else:
            top_k_values = memory_instance._get(current_state_embd, top_k=top_k)
            # print("TOP K VALUES: ", top_k_values)
            return np.argmax(top_k_values), epsilon
        else:
            value_list = memory_instance.values[subset_idx]
            return np.argmax(value_list), epsilon
    else:
        # if on training mode, we use epsilon greedy
        value_list = memory_instance.values[subset_idx]
        if random.random() < epsilon:
            return random.randint(0, len(value_list) - 1), epsilon
        else:
            return np.argmax(value_list), epsilon
            
    
def memory_verbalizer_epsilon_greedy(evaluate, memory_verbalizer_instance, current_state_embd, subset_idx, current_update_episode, total_update_episode, init_epsilon, final_epsilon, top_k, n_verbalizers, is_seperate):
    """
    This function returns the index of max value in value_list if exploit, otherwise return a random index
    current_update_episode: the current episode of the update
    total_update_episode: the total number of episodes for the update
    """

    assert init_epsilon >= final_epsilon, "init_epsilon must be larger than final_epsilon"
    if not evaluate:
        assert current_update_episode <= total_update_episode, "current_eps must be smaller than total_eps"
    assert current_update_episode >= 0, "current_eps must be larger than 0"
    assert total_update_episode >= 0, "total_eps must be larger than 0"

    # Calculate the epsilon for this step
    epsilon = init_epsilon - (init_epsilon - final_epsilon) * current_update_episode / total_update_episode

    # NOTE: changed this to always act greedily when evaluating
    if not is_seperate:
        if evaluate:
            # if the state is not in memory, take top_k values
            if not any(np.array_equal(current_state_embd, val) for val in memory_verbalizer_instance.states):
                top_k_values = memory_verbalizer_instance._get(current_state_embd, top_k=top_k)
                return np.argmax(top_k_values), epsilon
            else:
                value_list = memory_verbalizer_instance.values[subset_idx]
                return np.argmax(value_list), epsilon
        else:
            value_list = memory_verbalizer_instance.values[subset_idx]
            if random.random() < epsilon:
                return random.randint(0, len(value_list) - 1), epsilon
            else:
                return np.argmax(value_list), epsilon
    else:
        if evaluate:
            # if state is not in memory, take top_k values
            index = next((i for i, sublist in enumerate(memory_verbalizer_instance.states) if np.array_equal(sublist, current_state_embd)), None)
            if any(s == [] for s in memory_verbalizer_instance.states) or index is None:
                top_k_values = memory_verbalizer_instance._get(current_state_embd, top_k=top_k)
                return np.argmax(top_k_values), epsilon
            else:
                # if s is in memory, take the best action
                value_list = memory_verbalizer_instance.values[index]
                return np.argmax(value_list), epsilon
        else:
            # Randomly select an index
            if random.random() < epsilon:
                return random.randint(0, n_verbalizers - 1), epsilon
            else:
                if any(np.array_equal(current_state_embd, state) for state in memory_verbalizer_instance.states):
                    index = next((i for i, sublist in enumerate(memory_verbalizer_instance.states) if np.array_equal(sublist, current_state_embd)), None)
                    value_list = memory_verbalizer_instance.values[index]
                else:
                    # if current state is not in memory, we get the top_k values and then greedily select the action                        
                    value_list = memory_verbalizer_instance._get(current_state_embd, top_k=top_k)
                return np.argmax(value_list), epsilon


def memory_all_verbalizer_epsilon_greedy(evaluate, memory_verbalizer_instance, current_state_embd, subset_idx, permutation_idx, current_update_episode, total_update_episode, init_epsilon, final_epsilon, top_k, n_verbalizers, is_seperate):
    """
    This function returns the index of max value in value_list if exploit, otherwise return a random index
    current_update_episode: the current episode of the update
    total_update_episode: the total number of episodes for the update
    """

    assert init_epsilon >= final_epsilon, "init_epsilon must be larger than final_epsilon"
    if not evaluate:
        assert current_update_episode <= total_update_episode, "current_eps must be smaller than total_eps"
    assert current_update_episode >= 0, "current_eps must be larger than 0"
    assert total_update_episode >= 0, "total_eps must be larger than 0"

    # Calculate the epsilon for this step
    epsilon = init_epsilon - (init_epsilon - final_epsilon) * current_update_episode / total_update_episode

    # NOTE: changed this to always act greedily when evaluating
    if evaluate:
        # if the state is not in memory, take top_k values
        if not any(np.array_equal(current_state_embd, val) for val in memory_verbalizer_instance.states):
            top_k_values = memory_verbalizer_instance._get(permutation_idx, current_state_embd, top_k=top_k)
            return np.argmax(top_k_values), epsilon
        else:
            value_list = memory_verbalizer_instance.values[subset_idx][permutation_idx]
            return np.argmax(value_list), epsilon
    else:
        value_list = memory_verbalizer_instance.values[subset_idx][permutation_idx]
        if random.random() < epsilon:
            return random.randint(0, len(value_list) - 1), epsilon
        else:
            return np.argmax(value_list), epsilon


def get_state_embds(instruction_embds, test_embds):
    """
    This function returns the state embeddings, given instruction embeddings and test embeddings
    """
    state_embds = []
    for instruction_embd, test_embd in zip(instruction_embds, test_embds):
        state_embds.append(np.concatenate((instruction_embd.cpu().numpy(), test_embd), axis=0))
    state_embds = np.stack(state_embds)
    return state_embds


def get_permutations_dict(input_list, permutation_length):
    from itertools import permutations
    # Get all permutations of the specified length
    all_permutations = list(permutations(input_list, permutation_length))
    
    # Create a dictionary with indices as keys and permutations as values
    permutations_dict = {i: list(perm) for i, perm in enumerate(all_permutations)}
    
    return permutations_dict



def construct_prompt(params, instruction, train_sentences, train_labels, test_sentence, prompt_template):
    from promptsource.templates import DatasetTemplates
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # Somehow we have to convert test_sentence to its original form as huggingface dataset
    if train_sentences is not None and train_labels is not None:
        if params['dataset'] in ['glue/sst2', 'cr', 'movie_review', 'glue/cola']:
            train_sentences_ = []
            for ts, tl in zip(train_sentences, train_labels):
                train_sentences_.append({'sentence': ts, 'label': tl})
        elif params['dataset'] in ['ag_news', 'yelp_polarity', 'imdb', 'subj', 'mteb/twitter', 'poem_sentiment', 'SetFit/sst5', 'trec', 'hate_speech18']:
            train_sentences_ = []
            for ts, tl in zip(train_sentences, train_labels):
                train_sentences_.append({'text': ts, 'label': tl})
        elif params['dataset'] in ['super_glue/rte', 'snli', 'glue/mrpc', 'glue/qnli', 'glue/mnli', 'super_glue/boolq', 'glue/qqp']:
            # These datasets already in the form
            train_sentences_ = train_sentences
        else:
            raise ValueError("Dataset not supported")


    if params['dataset'] in ['glue/sst2', 'cr', 'movie_review', 'glue/cola']:
        test_sentence_ = {'sentence': test_sentence, 'label': None}
    elif params['dataset'] in ['ag_news', 'imdb', 'yelp_polarity', 'subj', 'mteb/twitter', 'poem_sentiment', 'SetFit/sst5', 'trec', 'hate_speech18']:
        test_sentence_ = {'text': test_sentence, 'label': None}
    elif params['dataset'] in ['super_glue/rte', 'glue/mrpc', 'glue/qnli', 'snli', 'glue/mnli', 'super_glue/boolq', 'glue/qqp']:
        test_sentence_ = test_sentence
    else:
        raise ValueError("Dataset not supported")

    # take the prompt template and fill in the training and test example
    prompt = instruction
    q_prefix = params["q_prefix"] 
    a_prefix = params["a_prefix"]
    if train_sentences is not None and train_labels is not None:
        for s, l in zip(train_sentences_, train_labels):
            if params['prompt_format'] == 'newline':
                prompt += '\n\n'
            prompt += q_prefix 
            if prompt_template is not None:
                prompt += prompt_template.apply(s)[0]
            else:
                if params['dataset'] in ['ag_news', 'glue/mrpc', 'super_glue/rte', 'imdb', 'yelp_polarity', 'subj', 'poem_sentiment', 'mteb/twitter', 'SetFit/sst5', 'trec', 'hate_speech18']:
                    prompt += s['text']
                else:
                    prompt += s['sentence']
            prompt = prompt.strip()
            if params['prompt_format'] != 'newline':
                prompt += ". "
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
                # assert params['task_format'] == 'classification'
                l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
            else:
                assert isinstance(l, str) # string labels
                assert params['task_format'] == 'qa'
                l_str = l

            if params['prompt_format'] == 'newline':
                prompt += '\n'
            prompt += a_prefix
            prompt += l_str
            prompt = prompt.strip()
            if params['prompt_format'] != 'newline':
                prompt += ". "

    else:
        pass

    # now we add the test sentence
    if params['prompt_format'] == 'newline':
        prompt += '\n\n'
    prompt += q_prefix
    if prompt_template is not None:
        prompt += prompt_template.apply(test_sentence_)[0]
    else:
        if params['dataset'] in ['ag_news', 'yelp_polarity', 'imdb', 'subj', 'mteb/twitter', 'poem_sentiment', 'SetFit/sst5', 'trec', 'hate_speech18']:
            prompt += test_sentence_['text']
        elif params['dataset'] == 'math_qa':
            prompt += test_sentence_['Problem']
        else:
            if params['dataset'] in ['super_glue/rte']:
                # NOTE: just concat here
                for k, v in test_sentence_.items():
                    # do not concat label value
                    if k != 'label' or type(v) != int:
                        prompt += v
                        prompt += ". "
            else:
                prompt += test_sentence_['sentence']

    prompt = prompt.strip()
    if params['prompt_format'] != 'newline':
        prompt += ". "
    assert a_prefix[-1] == ' '
    if params['prompt_format'] == 'newline':
        prompt += '\n'
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    if params['prompt_format'] != 'newline':
        prompt = prompt.replace("\n\n", " ")
        prompt = prompt.replace("\n", "")

    return prompt

def get_embedding(args, model, tokenizer, sentence):
    """
    naively return the sentence embedding of the sentence
    """
    if args.type_embd == 'senemb':
        sentence_embedding = get_senemb_embbeddings(sentence, model=model)
    elif args.type_embd == 'llama':
        sentence_embedding = get_llama_embeddings(sentence, model, tokenizer)
    
    return sentence_embedding


def get_ic_sentences_faiss(args, model, tokenizer, train_sentences, pool_sentences_embeddings, rest_train_sentences, rest_train_labels):
    if args.type_embd == 'senemb':
        print("Current senemb model has dim = 768")
        dim = None # TODO: fix this
    else:
        print("Current LLAMA model has dim = 4096")
        dim = 4096
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(pool_sentences_embeddings.cpu().numpy().astype('float32'))
    index.add(pool_sentences_embeddings.cpu().numpy().astype('float32'))

    k=10000

    ic_examples = {}
    ic_indicies = {}
    ic_labels = {}

    for idx, train_sentence in tqdm(enumerate(train_sentences), desc='Preparing IC examples for each training sentence'):
        ic_examples[idx] = []
        ic_indicies[idx] = []
        ic_labels[idx] = []
        list_labels = []

        if args.type_embd == 'senemb':
            train_sentence_embedding = get_senemb_embbeddings(train_sentence, model=model)
            train_sentence_vector = np.array([train_sentence_embedding.cpu().numpy()])
        elif args.type_embd == 'llama':
            train_sentence_embedding = get_llama_embeddings(train_sentence, model, tokenizer)
            train_sentence_vector = np.array([train_sentence_embedding])
        faiss.normalize_L2(train_sentence_vector)

        similarities, indicies = index.search(train_sentence_vector, k)
        
        # NOTE: list_lables now contains the labels correspond to the indicies, not following the original order
        list_labels.extend(rest_train_labels[i] for i in indicies[0])
        # unique_labels = list(set(list_labels))

        # # get the indicies for each label
        # for ul in unique_labels:
        #     i = np.where(np.array(list_labels) == ul)[0][:2]
        #     ind.extend(i)
        #     d.extend(distances[0][i])
        
        # print("List labels: ", list_labels)
        # print()

        # sorted_indicies = [x for _, x in sorted(zip(d, ind), reverse=True)]
        # ordered_indicies = [indicies[0][i] for i in sorted_indicies]
        # print("Ordered indicies: ", ordered_indicies)


        label_counts = {label:0 for label in set(list_labels)}
        # TODO: This might cause error if given 10000 examples does not appear one label
        selected_indicies, selected_labels = [], []
        
        # Iterate through sorted distances and select examples ensuring four examples are selected
        # NOTE: This code assumes that we have 4 in-context examples in each prompt
        
        for i, label, similarity in zip(indicies[0], list_labels, similarities[0]):
            if len(set(list_labels)) == 2:
                if label_counts[label] < 2:
                    selected_indicies.append(i)
                    selected_labels.append(label)
                    label_counts[label] += 1
                    print(f"Selected {label_counts[label]} examples for label {label} with distance {similarity}")
                    print(f"Sentence: {train_sentence}")
                    print(f"Selected sentence: {rest_train_sentences[i]}")
                    print("==========")
                elif label_counts[label] == 2:
                    if sum(label_counts.values()) == args.n_slots:
                        break
                    else:
                        continue
            elif len(set(list_labels)) == 4:
                if label_counts[label] < 1:
                    selected_indicies.append(i)
                    selected_labels.append(label)
                    label_counts[label] += 1
                elif label_counts[label] == 1:
                    if sum(label_counts.values()) == args.n_slots:
                        break
                    else:
                        continue

        # Check if exactly 4 examples are selected
        if len(selected_indicies) != 4:
            raise ValueError("Exactly 4 examples should be selected.")


        ic_examples[idx].extend(rest_train_sentences[i] for i in selected_indicies)
        ic_indicies[idx].extend(i for i in selected_indicies)
        ic_labels[idx].extend(rest_train_labels[i] for i in selected_indicies) 
        
        print("IC examples: ", ic_examples[idx])
        print("IC indicies: ", ic_indicies[idx])
        print("IC labels: ", ic_labels[idx])



    return ic_examples, ic_indicies, ic_labels


def get_data_from_tsv(file_path):
    import csv
    sentences, labels = [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            sentences.append(row[0])
            labels.append(int(row[1]))
        



def fill_template(sentence, ex1, label1, ex2, label2, ex3, label3, ex4, label4, dataset_name):
    if dataset_name=='glue/sst2' or dataset_name=='movie_review':
        # template = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\nReview: [EX1]\nSentiment: [LAB1]\n\nReview: [EX2]\nSentiment: [LAB2]\n\nReview: [EX3]\nSentiment: [LAB3]\n\nReview: [EX4]\nSentiment: [LAB4]\n\nReview: [SENT]\nSentiment: <mask>"
        template = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. Review: [EX1]. Sentiment: [LAB1]. Review: [EX2]. Sentiment: [LAB2]. Review: [EX3]. Sentiment: [LAB3]. Review: [EX4]. Sentiment: [LAB4]. Review: [SENT]. Sentiment: "
    elif dataset_name == 'ag_news':
        template = "Classify the news articles into the categories of World, Sports, Business, and Technology. Article: [EX1]. Answer: [LAB1]. Article: [EX2]. Answer: [LAB2]. Article: [EX3]. Answer: [LAB3]. Article: [EX4]. Answer: [LAB4]. Article: [SENT]. Answer: "
    
    elif dataset_name == 'glue/cola':
        template = "You will be given a sentence. Check whether the sentence is grammatically correct and is meaningful. If the sentence is grammatically correct, then answer with \"correct\", otherwise answer with \"incorrect\". Sentence: [EX1]. Answer: [LAB1]. Sentence: [EX2]. Answer: [LAB2]. Sentence: [EX3]. Answer: [LAB3]. Sentence: [EX4]. Answer: [LAB4]. Sentence: [SENT]. Answer: "

    elif dataset_name == 'cr':
        template = "In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. Review: [EX1]. Sentiment: [LAB1]. Review: [EX2]. Sentiment: [LAB2]. Review: [EX3]. Sentiment: [LAB3]. Review: [EX4]. Sentiment: [LAB4]. Review: [SENT]. Sentiment: "

    elif dataset_name == 'mteb/twitter':
        template = "In this task, you are given tweets. The task is to classify a tweet as \"positive\" if the sentiment of the tweet is positive or as \"negative\" if the sentiment of the tweet is negative. Tweet: [EX1]. Sentiment: [LAB1]. Tweet: [EX2]. Sentiment: [LAB2]. Tweet: [EX3]. Sentiment: [LAB3]. Tweet: [EX4]. Sentiment: [LAB4]. Tweet: [SENT]. Sentiment: "

    elif dataset_name == 'poem_sentiment':
        template = "In this task, you need to identify the sentiment of the given sentence as one of 'negative', 'positive', 'neutral' or 'mixed'. Sentence: [EX1]. Sentiment: [LAB1]. Sentence: [EX2]. Sentiment: [LAB2]. Sentence: [EX3]. Sentiment: [LAB3]. Sentence: [EX4]. Sentiment: [LAB4]. Sentence: [SENT]. Sentiment: "

    elif dataset_name == 'super_glue/boolq':
        template = "In this task you will be given a passage and a yes/no question based on the passage. You should answer the question using the information from the passage. Question: [EX1]. Answer: [LAB1]. Question: [EX2]. Answer: [LAB2]. Question: [EX3]. Answer: [LAB3]. Question: [EX4]. Answer: [LAB4]. Question: [SENT]. Answer: "

    elif dataset_name == 'glue/qqp':
        template = "In this task, you will be given two questions. You need to determine if the two questions are asking about the same information. Question 1: [EX1]. Question 2: [EX2]. Answer: [LAB1]. Question 1: [EX3]. Question 2: [EX4]. Answer: [LAB2]. Question 1: [SENT]. Question 2: <mask>. Answer: "

    elif dataset_name == 'imdb':
        template = "In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) great, and 2) terrible based on its content. Review: [EX1]. Sentiment: [LAB1]. Review: [EX2]. Sentiment: [LAB2]. Review: [EX3]. Sentiment: [LAB3]. Review: [EX4]. Sentiment: [LAB4]. Review: [SENT]. Sentiment: "

    return template.replace("[SENT]", sentence).replace("[EX1]", ex1).replace("[EX2]", ex2).replace("[LAB1]", label1).replace("[LAB2]", label2).replace("[EX3]", ex3).replace("[LAB3]", label3).replace("[EX4]", ex4).replace("[LAB4]", label4)

def get_ic_examples_qa(metric, test_embd, ic_embds):
    from collections import defaultdict
    # Fit Nearest Neighbors model
    neigh = NearestNeighbors(n_neighbors=len(ic_embds), metric=metric)
    neigh.fit(ic_embds)

    # Find nearest neighbors. at 0 sentence is closest
    distances, indices = neigh.kneighbors([test_embd])
    
    # Collect examples while ensuring diversity
    res = []
    for idx in indices[0]:
        res.append(idx)
    return res


def get_ic_examples_imbalance(args, test_embd, ic_embds, ic_labels):
    neigh = NearestNeighbors(n_neighbors=len(ic_embds), metric=args.distance_metric)
    neigh.fit(ic_embds)

    # Find nearest neighbors
    distances, indices = neigh.kneighbors([test_embd])

    res = []
    for idx in indices[0]:
        if len(res) == 4:
            break
        else:
            res.append(idx)
    
    return res


def get_ic_examples_furthest(args, test_embd, ic_embds, ic_labels, n_examples=4):
    from collections import defaultdict
    # Fit Nearest Neighbors model
    neigh = NearestNeighbors(n_neighbors=len(ic_embds), metric=args.distance_metric)
    neigh.fit(ic_embds)

    # Find nearest neighbors
    distances, indices = neigh.kneighbors([test_embd])
    
    # Collect examples while ensuring diversity
    res = []
    label_count = {i:0 for i in set(ic_labels)}
    max_per_label = n_examples // len(set(ic_labels))  # Calculate maximum examples per label

    if max_per_label == 0 or n_examples == len(set(ic_labels)):
        # if there are more labels than examples, for each sentence, we take 4 closest examples
        max_per_label = 1
        for idx in indices[0][::-1]:
            label = ic_labels[idx]
            if len(res) == n_examples:
                break
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
    elif n_examples == 6 and len(set(ic_labels)) == 4:
        # this is for agnews
        max_per_label = 1
        for idx in indices[0][::-1]:
            label = ic_labels[idx]
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
            if len(res) == 4:
                # we need to select 2 more examples
                for i in indices[0][::-1]:
                    if i not in res:
                        res.append(i)
                        if len(res) == 6:
                            break
                break

        assert len(res) == 6, "We should have 6 examples"
    else:
        if len(set(ic_labels)) == 3:
            for idx in indices[0][::-1]:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == 3:
                    # we need to find one more example
                    # select the last example by taking the closest example
                    for i in indices[0][::-1]:
                        if i not in res:
                            res.append(i)
                            break
                    break
                elif len(res) == n_examples:
                    break
        else:
            for idx in indices[0][::-1]:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == n_examples:
                    break
    return res

def get_ic_examples(args, test_embd, ic_embds, ic_labels, n_examples=4):
    from collections import defaultdict
    # Fit Nearest Neighbors model
    neigh = NearestNeighbors(n_neighbors=len(ic_embds), metric=args.distance_metric)
    neigh.fit(ic_embds)

    # Find nearest neighbors
    distances, indices = neigh.kneighbors([test_embd])

    # Collect examples while ensuring diversity
    res = []
    label_count = {i:0 for i in set(ic_labels)}


    max_per_label = n_examples // len(set(ic_labels))  # Calculate maximum examples per label


    if max_per_label == 0 or n_examples == len(set(ic_labels)):
        # if there are more labels than examples, for each sentence, we take 4 closest examples
        max_per_label = 1
        for idx in indices[0]:
            label = ic_labels[idx]
            if len(res) == n_examples:
                break
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
    elif n_examples == 6 and len(set(ic_labels)) == 4:
        # this is for agnews
        max_per_label = 1
        for idx in indices[0]:
            label = ic_labels[idx]
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
            if len(res) == 4:
                # we need to select 2 more examples
                for i in indices[0]:
                    if i not in res:
                        res.append(i)
                        if len(res) == 6:
                            break
                break

        assert len(res) == 6, "We should have 6 examples"
    else:
        if len(set(ic_labels)) == 3:
            for idx in indices[0]:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == 3:
                    # we need to find one more example
                    # select the last example by taking the closest example
                    for i in indices[0]:
                        if i not in res:
                            res.append(i)
                            break
                    break
                elif len(res) == n_examples:
                    break
        else:
            for idx in indices[0]:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == n_examples:
                    break
    return res

def get_ic_examples_bm25(args, bm25_corpus, test_sentence, ic_labels, n_examples=4):
    """
    Given a bm25 object, which stores all the ic_sentences in a list, when we prompt the object with 'get_scores()', we should receive a list of scores
    corresponding to the order that we gave to bm25 object
    """
    tokenized_test_sentence = test_sentence.split()
    ic_scores = bm25_corpus.get_scores(tokenized_test_sentence)
    ex = n_examples

    # now ic_scores is an array that saves the scores of each ic_sentence
    # we now wnat to order the list
    indices = np.argsort(ic_scores)[::-1]

    # Collect examples while ensuring diversity
    res = []
    label_count = {i:0 for i in set(ic_labels)}


    max_per_label = n_examples // len(set(ic_labels))  # Calculate maximum examples per label


    if max_per_label == 0 or n_examples == len(set(ic_labels)):
        # if there are more labels than examples, for each sentence, we take 4 closest examples
        max_per_label = 1
        for idx in indices:
            label = ic_labels[idx]
            if len(res) == n_examples:
                break
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
    elif n_examples == 6 and len(set(ic_labels)) == 4:
        # this is for agnews
        max_per_label = 1
        for idx in indices:
            label = ic_labels[idx]
            if label_count[label] < max_per_label:
                res.append(idx)
                label_count[label] += 1
            if len(res) == 4:
                # we need to select 2 more examples
                for i in indices:
                    if i not in res:
                        res.append(i)
                        if len(res) == 6:
                            break
                break

        assert len(res) == 6, "We should have 6 examples"
    else:
        if len(set(ic_labels)) == 3:
            for idx in indices:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == 3:
                    # we need to find one more example
                    # select the last example by taking the closest example
                    for i in indices:
                        if i not in res:
                            res.append(i)
                            break
                    break
                elif len(res) == n_examples:
                    break
        else:
            for idx in indices:
                label = ic_labels[idx]
                if label_count[label] < max_per_label:
                    res.append(idx)
                    label_count[label] += 1
                if len(res) == n_examples:
                    break
    return res


def get_random_examples(list_examples, list_labels, n_examples=4):
    # set fixed random seed
    random.seed(0)
    n_labels = len(set(list_labels))
    per_label = int(n_examples // n_labels)
    res = []
    lab = []
    # randomly select examples while assert that each label has per_label examples
    for label in set(list_labels):
        indices = [i for i, x in enumerate(list_labels) if x == label]
        res.extend(random.sample(indices, per_label))
        lab.extend([label] * per_label)
    return res, lab


def get_balanced_examples(list_examples, list_labels, n_examples=4):
    # set fixed random seed
    random.seed(0)
    n_labels = len(set(list_labels))
    per_label = int(n_examples // n_labels)
    res = []
    lab = []
    exs = []
    # randomly select examples while assert that each label has per_label examples
    for label in set(list_labels):
        indices = [i for i, x in enumerate(list_labels) if x == label]
        exs.extend(list_examples[i] for i in indices)
        res.extend(random.sample(exs, per_label))
        lab.extend([label] * per_label)

    if len(set(list_labels)) == 4 and len(res) != n_examples:
        # this is for agnews and n_examples = 6
        # we get 2 more examples, 1 per label
        selected_labels = random.sample(list(set(list_labels)), 2)
        for label in selected_labels:
            indices = [i for i, x in enumerate(list_labels) if x == label]
            exs.extend(list_examples[i] for i in indices)
            res.extend(random.sample(exs, 1))
            lab.extend([label])


    return res, lab

