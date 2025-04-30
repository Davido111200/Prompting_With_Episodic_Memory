import argparse
import random
import numpy as np
from tqdm import tqdm
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .reward_llm import setup_reward_llm, get_score, get_embeddings
from .data_utils import custom_load_dataset
from .arguments import get_training_args
import tiktoken
import gym
import json


from promptsource.templates import DatasetTemplates
from jinja2 import Environment
import random

def generate_template_prompts(params, train_sentences, train_labels, test_sentences, key, include_instruction, include_in_context, include_test_sentence):
    env = Environment()
    prompt_templates = DatasetTemplates(params['dataset'])
    prompt_template_keys = prompt_templates.all_template_names
    assert key < len(prompt_template_keys), "key must be less than the number of templates"
    key = prompt_template_keys[key]
    
    verbalized_answers = params['inv_label_dict'].keys()
    answer_list = prompt_templates[key].answer_choices.split("|||")

    if params['verbalize']==True:    
        for prompt_answer, correct_answer in zip(answer_list, verbalized_answers):
            prompt_templates[key].jinja = prompt_templates[key].jinja.replace(prompt_answer.strip(), correct_answer)
        

    if params['dataset'] == 'glue/sst2':
        test_template = env.from_string(prompt_templates[prompt_template_keys[0]].jinja.split("|||")[0])
    elif params['dataset'] == 'ag_news':
        test_template = env.from_string(prompt_templates[prompt_template_keys[0]].jinja.split("|||")[0])

    prompts = []

    # Precompile the primary template
    template = env.from_string(prompt_templates[key].jinja.split("|||")[0])

    # Create a dictionary for fast label lookups
    train_labels_dict = {sentence: label for sentence, label in zip(train_sentences, train_labels)}

    for i, test_sentence in enumerate(test_sentences):
        in_context_templates = train_sentences if include_in_context else []

        # Construct in-context examples
        in_context_examples = "".join([f"{params['q_prefix']}{template.render(sentence=ic)}\n"
                                      f"{params['a_prefix']}{params['label_dict'][train_labels_dict[ic]][0]}\n\n"
                                      for ic in in_context_templates]) if include_in_context else ""
        
        # Fill in the test sentence into the template
        test_sentence_to_use = test_sentence if include_test_sentence else ""
        if params['dataset'] == 'glue/sst2':
            test_prompt = test_template.render(sentence=test_sentence_to_use) if include_test_sentence else ""
        elif params['dataset'] == 'ag_news':
            test_prompt = test_template.render(text=test_sentence_to_use) if include_test_sentence else ""

        # Combine in-context examples and the test prompt
        prompt = ""


        if include_instruction==True:
            prompt += f"{params['prompt_prefix']}\n\n"
        if include_in_context==True:
            prompt += f"{in_context_examples}"
        if include_test_sentence==True:
            prompt += f"{params['q_prefix']}{test_prompt}\n{params['a_prefix']}"
        prompts.append(prompt)
        
    return prompts


def tokenize_sentence(sentence, model='gpt2'):
    """
    Tokenize a sentence into tokens, corresponding to the model
    """
    encoder = tiktoken.encoding_for_model(model)
    return encoder.decode(encoder.encode(sentence))

def construct_initial_prompt(params, train_sentences, train_labels, test_sentence, test_label, selection='random', hand_craft_examples=True, hand_craft_test_prompt=True):
    """
    Construct a prompt with given parameters
    :param params: a dictionary of parameters
    :param train_sentences: a list of training sentences
    :param train_labels: a list of training labels
    :param test_sentence: final test sentence
    :param method: method to construct the prompt
    :param selection: selection method for selecting training sentences. Default is to random sampling from the training set
    :param hand_craft_test_prompt: whether to use hand-crafted test prompt or not
    """
    # check if params has key 'prompt_prefix'
    assert 'prompt_prefix' in params.keys(), "Prompt prefix not found in params"
    # check if length of train_sentences and train_labels are the same
    assert len(train_sentences) == len(train_labels), "Length of train_sentences and train_labels must be the same"
    # check if params has key 'n_examples', which is the number of examples in the initial prompt
    assert 'n_examples' in params.keys(), "Number of examples not found in params"

    # check if method is instruction_first
    # TODO: Automate this
    assert 'action_instruction_prefix' in params.keys(), 'action_instruction_prefix not found in params'

    prompt = params['action_instruction_prefix']
    prompt += "\n\n" + params['prompt_prefix']

    # convert train_sentences and test_sentence to list of strings, if current is type dict
    if isinstance(train_sentences[0], dict):
        train_sentences = [sentence['sentence'] for sentence in train_sentences]
    if isinstance(test_sentence, dict):
        test_sentence = test_sentence['sentence']

    # check if there is params['label_dict']
    if 'label_dict' not in params.keys():
        text_train_labels = train_labels
    else:
        text_train_labels = [value for _, value in params['label_dict'].items()]

    # NOTE: if we do not want the examples and test prompt to be affected by LM, we just append if after the initial prompt is edited
    if not hand_craft_examples:
        # add initial in-context examples
        prompt = append_examples(params, prompt, train_sentences, train_labels, text_train_labels)

    if not hand_craft_test_prompt:
        prompt += "\n" + params["q_prefix"] + test_sentence
        prompt += "\n" + params["a_prefix"]

    return prompt, test_label

def append_examples(params, prompt, train_sentences, train_labels, text_train_labels):
    random_indices = random.sample(range(len(train_sentences)), params['n_examples'])
    random_sentences = [train_sentences[idx] for idx in random_indices]
    random_labels = [train_labels[idx] for idx in random_indices]
    # convert random_labels to text
    random_labels_text = [text_train_labels[idx][0] for idx in random_labels]
    for train_sentence, train_label in zip(random_sentences, random_labels_text):
        # train_label available in one-element list
        prompt += params["q_prefix"] + train_sentence + "\n" + params["a_prefix"] + train_label + "\n\n"

    return prompt

def append_test_example(params, prompt, final_example):
    prompt += "\n" + params['q_prefix'] + params['test_prefix'] + "\n" + final_example + "\n" + params['a_prefix']
    return prompt

def construct_prompt(params, train_sentences, train_labels, test_sentence, method='random', hand_craft_examples=False, hand_craft_test_prompt=False):
    """
    Construct a prompt with given parameters
    :param params: a dictionary of parameters
    :param train_sentences: a list of training sentences
    :param train_labels: a list of training labels
    :param test_sentence: final test sentence
    :param method: method to construct the prompt
    """
    # check if params has key 'prompt_prefix'
    assert 'prompt_prefix' in params.keys(), "Prompt prefix not found in params"
    # check if length of train_sentences and train_labels are the same
    assert len(train_sentences) == len(train_labels), "Length of train_sentences and train_labels must be the same"

    # check if method is instruction_first
    q_prefix = params['q_prefix']
    a_prefix = params['a_prefix']

    # TODO: Optimize this
    prompt = params['prompt_prefix']
    prompt += "\n"

    if not hand_craft_examples:
        for s, l in zip(train_sentences, train_labels):
            prompt += q_prefix
            prompt += s + "\n"
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): #int label
                l_str = params['label_dict'][l][0] if isinstance(params['label_dict'][l], list) else params['label_dict'][l]
            else:
                assert isinstance(l, str)
                l_str = l
    if not hand_craft_test_prompt:
        prompt += a_prefix
        prompt += l_str + "\n\n"

        prompt += q_prefix
        prompt += test_sentence + "\n"
        assert a_prefix[-1] == " "
        prompt += a_prefix[:-1]

    return prompt

def convert_to_template_prompts(params, train_sentences, train_labels, test_sentence, test_label, verbalizer=False):
    """
    Convert the train_sentences and test_sentence to template prompts
    """
    from promptsource.templates import DatasetTemplates
    from jinja2 import BaseLoader, Environment
    env = Environment()

    prompt_templates = DatasetTemplates(params['dataset'])
    prompt_template_keys = prompt_templates.all_template_names

    prompts = []
    labels = []

    for key in prompt_template_keys:
        if verbalizer:
            answer_list = prompt_templates[key].answer_choices.split("|||")
            for prompt_answer, correct_answer in zip(answer_list, params['inv_label_dict'].keys()):
                prompt_templates[key].jinja = prompt_templates[key].jinja.replace(prompt_answer.strip(), correct_answer)
        
        # Now we construct prompts with the template
        prompt = params['prompt_prefix']
        template = env.from_string(prompt_templates[key].jinja.split("|||")[0])

        if test_sentence is None and test_label is None:
            i_range = [i for i in range(len(train_sentences))]
            pass

        for (idx, s), l in tqdm(zip(enumerate(train_sentences), train_labels), desc="Constructing prompts"):
            indices = random.sample(i_range[:idx] + i_range[idx+1:], params['n_examples'])
            random_train_examples = [train_sentences[i] for i in indices]
            random_train_labels = [train_labels[i] for i in indices]
            random_text_train_labels = [params['label_dict'][l] for l in random_train_labels]

            cur_prompt = append_examples(params, '', random_train_examples, random_train_labels, random_text_train_labels)
            
            test_example = s + "\n" + params['a_prefix'] 

            test_template_example = template.render(sentence=test_example)

            prompt_sentence = prompt + cur_prompt + params['q_prefix'] + test_template_example        
    
            prompts.append(prompt_sentence)
            labels.append(l)
        
    return prompts, labels

def manual_prompt(params, test_sentences, test_labels, verbalizer=False):
    """
    Construct a manual prompt with given parameters and save the data to a JSON file with different keys for labels.
    :param params: a dictionary of parameters
    :param test_sentences: a list of test sentences
    :param test_labels: a list of test labels
    :param verbalizer: a flag for verbalization (not used in this example)
    """
    from promptsource.templates import DatasetTemplates
    from jinja2 import BaseLoader, Environment
    env = Environment()

    prompt_templates = DatasetTemplates(params['dataset'])
    prompt_template_keys = prompt_templates.all_template_names

    # Create a dictionary to store prompts for each template key
    prompt_dict = {}

    for key in prompt_template_keys:
        temp = prompt_templates[key]
        prompts, labels = [], []

        for s, l in zip(test_sentences, test_labels):
            template = env.from_string(temp.jinja.split("|||")[0])

            if key == prompt_template_keys[0]:
                prompt = params['q_prefix'] + template.render(sentence=s) + "\n" + params['a_prefix'] + '<mask> '
            elif key == prompt_template_keys[1]:
                q_and_test_sentence = params['q_prefix'] + s + "\n" + params['a_prefix'] + '<mask> '
                prompt = template.render(sentence=q_and_test_sentence)
            elif key == prompt_template_keys[2]:
                q_and_test_sentence = params['q_prefix'] + s + "\n"
                prompt = template.render(sentence=q_and_test_sentence) + '<mask>'
            elif key == prompt_template_keys[3]:
                q_and_test_sentence = s
                prompt = params['prompt_prefix'] + template.render(sentence=q_and_test_sentence) + params['a_prefix'] + '<mask>'
            elif key == prompt_template_keys[4]:
                q_and_test_sentence = s + "\n"
                prompt = params['prompt_prefix'] + template.render(sentence=q_and_test_sentence) + params['a_prefix'] + '<mask>'

            prompts.append(prompt)
            labels.append(l)

        if key not in prompt_dict:
            prompt_dict[key] = {}

        for label, prompt in zip(labels, prompts):
            if label not in prompt_dict[key]:
                prompt_dict[key][label] = []
            prompt_dict[key][label].append(prompt)

    # Save the prompt_dict to a JSON file
    with open('/home/s223540177/dai/RLforLLM/data/sst2/train_prompts.json', 'w') as json_file:
        json.dump(prompt_dict, json_file, indent=4)

    return prompt_dict



def eval_no_prompt():
    args = get_training_args()

    params = {}
    params = {'dataset': args.dataset}
    params['n_examples'] = 3
    params['lambda1'] = 2.0
    params['lambda2'] = 1.8

    train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels = custom_load_dataset(params, change_params=True)
    if isinstance(train_sentences[0], dict):
        train_sentences = [sentence['sentence'] for sentence in train_sentences]
    if isinstance(valid_sentences[0], dict):
        valid_sentences = [sentence['sentence'] for sentence in valid_sentences]
    if isinstance(test_sentences[0], dict):
        test_sentences = [sentence['sentence'] for sentence in test_sentences]

    # prompts, labels = convert_to_template_prompts(params, train_sentences, train_labels, None, None, verbalizer=False)

    noedit_eval_episode_preds = []
    ac_eval_episode_preds = []

    import torch   
    from vec_env import PromptEnv
    from evaluation import eval_accuracy
    reward_llm, tokenizer_llm = setup_reward_llm(args.model_name, gpu_id=0)
    # results for manual prompt
    valid_prompts, valid_labels = manual_prompt(params, valid_sentences, valid_labels)
    preds = []

    for p, l in tqdm(zip(valid_prompts, valid_labels), desc="Manual"):
        _, _, noedit_pred, _ = get_score(params, reward_llm, tokenizer_llm, p, [l], num_predict_tokens=1, gpu_id=0)
        preds.append(noedit_pred)

    assert len(preds) == len(valid_labels)
    print("Accuracy on the test set: ", eval_accuracy(preds, valid_labels))


