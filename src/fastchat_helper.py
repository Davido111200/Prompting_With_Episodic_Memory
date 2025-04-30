import os
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, '/home/s223540177/dai/RLforLLM/FastChat')
from fastchat.conversation import get_conv_template

import torch

def load_model(model_path, load8bit, load4bit):
    model_path = os.path.expanduser(model_path)

    if model_path == "mistralai/Mixtral-8x7B-v0.1":
        print("USING MIXTRAL 8x7B")
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", pad_token_id=tokenizer.eos_token_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, pad_token_id=tokenizer.eos_token_id)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if load4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto',
                load_in_4bit=True, pad_token_id=tokenizer.eos_token_id
            ).eval()
            # config = AutoConfig.from_pretrained(model_path)
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_config(config)
            
            # device_map = "auto"
            # model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map,
            #                                     no_split_module_classes=model._no_split_modules)
            # model = model.eval()
        elif load8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto',
                load_in_8bit=True, pad_token_id=tokenizer.eos_token_id
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, pad_token_id=tokenizer.eos_token_id
            ).eval()
    return model, tokenizer


def get_model_answers(model, tokenizer, qs, temperature, top_p, max_new_tokens, prompt_template="mistral"):
    # conv = get_conversation_template(model_path, prompt_template=prompt_template)
    if prompt_template is not None:
        conv = get_conv_template(prompt_template)
        if prompt_template == 'llama-v2':
            print("Using llama template")
            conv.set_system_message("You are a helpful, honest assistant. Listen to the user's question and answer it.")
        elif prompt_template == 'mistral':
            conv.set_system_message("You are a helpful, honest assistant. Listen to the user's question and answer it.")
            # conv.set_system_message("You are a helpful assistant. You are given a prompt with instruction and in-context examples below. Your job is to improve and response 3 prompts, each starts with 'Instruction'. Do not change the classes of the instructions, which are 'great' and 'terrible'. Do not generate anythingelse beside the 3 improved in")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        prompt = qs

    input_ids = tokenizer([prompt], return_tensors="pt").to(torch.device('cuda:0'))

    output_ids = model.generate(
        **input_ids,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        # no_repeat_ngram_size=2,
        # num_return_sequences=1,
        # early_stopping=True,
        # top_k=50,
        top_p=top_p
    )
    # output_ids = output_ids[len(input_ids[0]):]

    # output_ids = output_ids[0]
    # outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    outputs = tokenizer.batch_decode(output_ids)[0]
    # outputs = conv.process_output(outputs)
    return outputs

