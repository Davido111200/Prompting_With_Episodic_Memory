import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, set_start_method

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, AutoTokenizer, AutoModelForCausalLM
from .utils import parse_instruction_from_examples_hand_examples

import numpy as np
from typing import List, Optional
import queue
import time
import threading
from llama import Llama, Dialog
import os

import re

def extract_mathqa_answer(input_string):
    # Use regular expression to find all occurrences of letters either in brackets or followed by a closing parenthesis or square bracket
    matches = re.findall(r'\((?=[a-eA-E]\))|\[(?=[a-eA-E]\])|([a-eA-E])(?=[\)\]])', input_string)
    if matches:
        # Return the last occurrence found
        return matches[-1]
    else:
        return None

def setup_reward_llm(model_name, gpu_id):
    # Check the validity of the GPU ID
    if model_name == 'roberta-large':

        config = RobertaConfig.from_pretrained(model_name)
        roberta_tokenzier = RobertaTokenizer.from_pretrained(model_name)
        roberta_model = RobertaForMaskedLM.from_pretrained(model_name, config=config)

        if gpu_id is not None:
            roberta_model.eval().to('cuda:'+str(gpu_id))
            print("Finish Reward LLM Setup on GPU {}".format(gpu_id))
        else:
            roberta_model.eval()
            print("Finish Reward LLM Setup on CPU")
        return roberta_model, roberta_tokenzier
    
def setup_qa_llm(model_name, load4bit=True):
    dir = "/weka/Projects/local_llms/model_weights/"
    model_path = os.path.join(dir, model_name)
    assert os.path.exists(model_path), "Model path does not exist"    


    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )

    if not load4bit:
        # config = AutoConfig.from_pretrained(model_path)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)
        
        # device_map = "auto"
        # model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map,
        #                                     no_split_module_classes=model._no_split_modules)
        # model = model.eval()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto', 
            trust_remote_code=True, load_in_4bit=False
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto',
            load_in_4bit=True
        ).eval()
    return model, tokenizer

def inference_worker(model, tokenizer, input_text, gpu_id):
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensor='pt').to('cuda:'+str(gpu_id))
        outputs = model.generate(input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
def get_score(params, reward_model, reward_tokenizer, edited_prompts, labels, num_predict_tokens, gpu_id):
    """
    This function will return score the reward model, given the editted prompts
    param params: parameters of the DATASET
    param reward_model: the reward model
    param reward_tokenizer: the reward tokenizer
    param editted_prompts: list of editted prompts
    param num_predict_tokens: number of tokens to predict
    """

    # TODO: modify to work with multiple inputs
    # add extra <mask> to the end of the prompt
    if not isinstance(edited_prompts, list):
        edited_prompts = [edited_prompts]

    if not isinstance(labels, list):
        labels = [labels]

    if '<mask>' not in edited_prompts[0]:
        for idx, edit_prompt in enumerate(edited_prompts):
            edited_prompts[idx] += ' <mask>'
    labels = labels[0]    

    # TODO: may add additional verbalizer
    label_idx = [reward_tokenizer.convert_tokens_to_ids('\u0120'+label) for label in params['inv_label_dict'].keys()]

    
    input_ids = reward_tokenizer.batch_encode_plus(edited_prompts, return_tensors='pt').to(reward_model.device)
    if input_ids['input_ids'].shape[1] > 512:
        input_ids['input_ids'] = input_ids['input_ids'][:, -512:]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, -512:]


    mask_ids = (input_ids['input_ids'] == reward_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        
    attention_mask = (input_ids['input_ids'] != reward_tokenizer.pad_token_id).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    for key in input_ids:
        input_ids[key] = input_ids[key].to('cuda:'+str(gpu_id))
    mask_ids = mask_ids.to('cuda:'+str(gpu_id))
    attention_mask = attention_mask.to('cuda:'+str(gpu_id))
    position_ids = position_ids.to('cuda:'+str(gpu_id))
    reward_model = reward_model.to('cuda:'+str(gpu_id))

    with torch.no_grad():
        outputs = reward_model(**input_ids, output_hidden_states=True)
        

    logits = outputs.logits.detach()

    # logits = logits.to('cuda:'+str(gpu_id))
    # mask_ids = mask_ids.to('cuda:'+str(gpu_id))

    # what we want is the gap between the label probability and the highest other probability
    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(logits[:, mask_ids].float(), dim=2)

    
    # NOTE: the reward is the difference between the probability of the label and the highest other probability
    # we just cant add another reward, as only finished prompts will receive reward
    top_tokens = torch.cat([torch.tensor(np.array(label_idx)).to(probs.device).unsqueeze(0).unsqueeze(0) for _ in range(probs.shape[0])], dim=0)
    top_probs = probs[:, :, torch.tensor(np.array(label_idx)).to(probs.device)] 
    log_probs = torch.log(top_probs)    

    # get the embeddings for the prompt
    try:
        embeddings = outputs.hidden_states[-1][:, mask_ids[0]].detach()
    except IndexError:
        print("PROMPT: ", edited_prompts)
        quit()

    # top_probs contains the probability of the label
    # here we can get the score of a prompt s
    # flatten the log_probs
    log_probs = log_probs.flatten()
    # this collects the top_k log_probs, with LABELS ONLY

    values, indices = torch.topk(log_probs, len(log_probs))

    lambda1 = params['lambda1']
    lambda2 = params['lambda2']

    # check if we are doing test set prediction
    if labels != -1:
        if labels == indices[0]:
            # This is when the predicted label is also the highest probability
            score = lambda1 * values[0] - lambda2 * values[1]
            pred_class = labels
            correct = 1
        else:
            # This is when the predicted label is not the highest probability
            score = lambda1 * values[indices.tolist().index(labels)].item() - lambda2 * values[0]
            pred_class = indices[0].item()
            correct = 0
    else:
        pred_class = indices[0]
        score = -9999 # default for test prediction
    
    return score, embeddings, pred_class, labels

def format_prompt_llama(prompt):
    template = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 
    Always give the answer first.
    Always wrap your answer in brackets.
    Do not repeat the question.
    Only generate the answer.
    Do not response with a greeting or anything other than the answer.
    <</SYS>>

    [PROMPT] [/INST]"""
    return template.replace("[PROMPT]", prompt)




def get_score_qa(params, qa_model, qa_tokenizer, edited_prompts, labels, num_predict_tokens):
    """
    This function will return score the reward model, given the editted prompts
    param params: parameters of the DATASET
    param reward_model: the reward model
    param reward_tokenizer: the reward tokenizer
    param editted_prompts: list of editted prompts
    param num_predict_tokens: number of tokens to predict
    """
    if params['reward_model_name'] in ['llama-v2-7B-chat']:
        # We need to format the prompt
        formatted_prompt = format_prompt_llama(edited_prompts)

    inputs = qa_tokenizer(formatted_prompt, return_tensors="pt")

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        outputs = qa_model.generate(**inputs, max_new_tokens=num_predict_tokens, return_dict_in_generate=True, output_scores=True)

    logits = outputs.scores[0][0].detach().cpu()
    
    # this contains logits for the predicted token. size = vocab_size

    next_token_probs = torch.softmax(logits, dim=-1)

    next_token_labels_logits = logits.detach().cpu()
    next_token_labels_probs = next_token_probs.detach().cpu()

    # get the top 2 highest probability tokens indices as a list
    top_2_indices = torch.topk(next_token_labels_probs, 2).indices.detach().cpu().numpy()

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    generated_answer = qa_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    extracted_answer = extract_mathqa_answer(generated_answer)

    # get extracted_answer_token
    extracted_answer_token = qa_tokenizer.convert_tokens_to_ids(extracted_answer)
    
    if not extracted_answer_token:
        # QUICK SOLUTION
        score = -9999
    else:
        if extracted_answer_token == top_2_indices[0]:
            # this is when the predicted label got the highest probability
            score = params['lambda1'] * next_token_labels_logits[extracted_answer_token] - params['lambda2'] * next_token_labels_logits[top_2_indices[1]]
        else:
            # this is when the predicted label is not the highest probability
            score = params['lambda1'] * next_token_labels_logits[extracted_answer_token] - params['lambda2'] * next_token_labels_logits[top_2_indices[0]]
        
        score = score.item()


    # convert string to label
    if params['dataset'] == 'math_qa':
        if extracted_answer not in params['inv_label_dict']:
            pass
        else:
            extracted_answer = params['inv_label_dict'][extracted_answer]

    return score, generated_answer, labels



def get_embeddings(params, reward_model, reward_tokenizer, input_sentence_, model_name='roberta-large', gpu_id=0):
    """
    This function receive a sentence and return the embedded vector with roberta-large model
    """
    if not isinstance(input_sentence_, list):
        input_sentence = [input_sentence_]
    else:
        input_sentence = input_sentence_

    input_sentence_copy = input_sentence.copy()


    if '<mask>' not in input_sentence_copy[0].split(' '):
        for idx, edit_prompt in enumerate(input_sentence_copy):
            input_sentence_copy[idx] += ' <mask>'

    label_idx = [reward_tokenizer.convert_tokens_to_ids('\u0120'+label) for label in params['inv_label_dict'].keys()]
    
    input_ids = reward_tokenizer.batch_encode_plus(input_sentence_copy, padding=True, truncation=True, return_tensors='pt').to(reward_model.device)

    if input_ids['input_ids'].shape[1] > 512:
        input_ids['input_ids'] = input_ids['input_ids'][:, -512:]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, -512:]


    if reward_tokenizer.mask_token_id not in input_ids['input_ids']:
        # first of all raise an error
        print("Your input sentence input_ids exceeded max_num_tokens, thus <mask> id truncated. To fix, instruction is truncated.")
        # split the prompt into instruction and examples
        instruction, examples = parse_instruction_from_examples_hand_examples(params, input_sentence_copy)
        
        # get the tokens from instruction and examples
        examples_tokens = reward_tokenizer.tokenize(examples)

        max_token_limit = reward_tokenizer.model_max_length

        input_tokens = examples_tokens
        truncated_prompt = reward_tokenizer.convert_tokens_to_string(input_tokens)

        # now get the input_ids of the instruction truncated prompt
        input_ids = reward_tokenizer.batch_encode_plus([truncated_prompt], padding=True, truncation=True, return_tensors='pt')


    mask_ids = (input_ids['input_ids'] == reward_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    attention_mask = (input_ids['input_ids'] != reward_tokenizer.pad_token_id).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # push to device
    # input_ids = input_ids.to('cuda:'+str(gpu_id))

    with torch.no_grad():
        outputs = reward_model(**input_ids, output_hidden_states=True)
    
    embeddings = outputs.hidden_states[-1][:, mask_ids[0]].detach().squeeze()

    # We need to detach the inputs and outputs from the graph
    return embeddings
