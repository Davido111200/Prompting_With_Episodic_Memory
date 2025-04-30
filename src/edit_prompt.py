import torch

import argparse
import langchain as lc
import tiktoken
import string
from typing import List, Optional
from src.models.llama import Llama, Dialog

def contruct_initial_edit_prompt(params, initial_prompts, edited_prompts, test_prompt, method='instruction_first'):
    """
    Construct an edit prompt to feed into Action LLM
    :param params: a dictionary of parameters
    :param initial_prompts: a list of initial prompts
    :param edited_prompts: a list of edited prompts
    :param test_prompt: final test prompt
    """

    assert 'instruction_prefix' in params.keys(), "Instruction prefix not found in params"
    assert 'init_prompt_prefix' in params.keys(), "Initial prompt prefix not found in params"
    assert 'edit_prompt_prefix' in params.keys(), "Edit prompt prefix not found in params"

    # TODO: automate this
    params['action_instruction_prefix'] = "In this task, you are given a prompt with examples. Your task is to improve the prompt."

    if method=="instruction_first":
        prompt = params['action_instruction_prefix']
        prompt += params['instruction_prefix']

        for init_prompt, edited_prompt in zip(initial_prompts, edited_prompts):
            prompt += '\n' + params['init_prompt_prefix'] + init_prompt
            prompt += '\n' + params['edit_prompt_prefix'] + edited_prompt

        # add the final test prompt
        prompt += '\n' + params['init_prompt_prefix'] + test_prompt
        prompt += '\n' + params['edit_prompt_prefix']
        
        return prompt
