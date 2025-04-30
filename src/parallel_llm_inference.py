import logging
import os
import time

import fire
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_configs():
    generation_cfg = {
        'max_new_tokens': 1024,
        'temperature': 0.2,
        'top_p': 0.95,
        'repetition_penalty': 1.15,
        'do_sample': True
    }
    inference_cfg = {
        'max_length': 1024,
        'tok_batch_size': 16,
        'inference_batch_size': 8,
    }
    model_cfg = {
        'pad_to_multiple_of': 8,
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    prompt_config = {
        'system_format': "### System:\n{system}\n\n",
        'system_no_input_prompt': "Below is an instruction that describes a task. Write a response "
                                  "that appropriately completes the request.\n\n",
        'turn_no_input_format': "### Instruction:\n{instruction}\n\n### Response:\n"
    }

    return generation_cfg, inference_cfg, model_cfg, bnb_config, prompt_config


def get_input_prompt(instruction, prompt_config):
    res = prompt_config['system_format'].format(system=prompt_config['system_no_input_prompt']) \
        + prompt_config['turn_no_input_format'].format(instruction=instruction)
    return res


def run_generation(generation_cfg, dataloader, tokenizer, model, accelerator):
    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.wait_for_everyone()

    output_sequences = []
    start_time = time.time()

    for batch in tqdm(dataloader):
        unwrapped_model = accelerator.unwrap_model(model)

        with torch.inference_mode():
            generated_tokens = unwrapped_model.generate(**batch, **generation_cfg)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()

        outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_tokens]
        output_sequences.extend(outputs)
        print("Output sequences: ", output_sequences)

    generation_end_time = time.time()
    logger.info(f"Generation time: {generation_end_time - start_time} sec")
    return output_sequences


def get_model(model_name, adapter_checkpoint_dir, model_cfg, bnb_config, accelerator):
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config
        )
        # model = PeftModel.from_pretrained(model, adapter_checkpoint_dir)
        model.resize_token_embeddings(model.config.vocab_size + 1,
                                      pad_to_multiple_of=model_cfg['pad_to_multiple_of'])
        # model = model.to_bettertransformer()
        return model


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            #   config=AutoConfig.from_pretrained(cfg_path),
                                              padding_side='left',
                                              truncation=True)

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    return tokenizer


# def get_dataloader(data_path, tokenizer, inference_cfg, prompt_config, accelerator):
#     with accelerator.main_process_first():
#         dataset = load_dataset("json", data_files=data_path)
#         dataset = dataset.map(lambda e: {'prompt': get_input_prompt(e['instruction'], prompt_config)})
#         columns = dataset['train'].column_names
#         tokenized = dataset['train'].map(
#             lambda e: tokenizer(e['prompt'], truncation=True, return_tensors='pt',
#                                 padding='max_length', max_length=inference_cfg['max_length']),
#             batched=True,
#             batch_size=inference_cfg['tok_batch_size'])
#         tokenized = tokenized.remove_columns(columns)
#         data_collator = DataCollatorWithPadding(tokenizer)
#         dataloader = DataLoader(tokenized, batch_size=inference_cfg['inference_batch_size'],
#                                 collate_fn=data_collator)
#         return dataloader

class PromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def fill(self, input_text):
        return self.template.replace("[INS]", input_text)


def get_prompt(template, input_text):
    return template.fill(input_text)

def get_dataloader(tokenizer, template, accelerator):
    with accelerator.main_process_first():
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(lambda e: {'prompt': get_prompt(template, e['sentence'])})
        print("Example prompt: ", dataset['train']['prompt'][0])
        columns = dataset['train'].column_names
        tokenized = dataset['train'].map(lambda e: tokenizer(e['prompt'], truncation=True, return_tensors='pt', padding='max_length', max_length=1024), batched=True, batch_size=16)
        tokenized = tokenized.remove_columns(columns)
        data_collator = DataCollatorWithPadding(tokenizer)
        dataloader = DataLoader(tokenized, batch_size=8, collate_fn=data_collator)
        return dataloader

def main(model_name,
         adapter_checkpoint_dir,
         data_path,
         output_dir):
    # cfg_path = os.path.join(adapter_checkpoint_dir, 'tokenizer_config.json')
    
    n_candidates = 3

    # downstream_instruction = f"<s>[INST]You are given a prompt with instruction and in-context examples and a test example, each is separated by 2 newlines. Your job is to generate {n_candidates} better instructions. Each of the the generated instructions should be a completed instruction. Start each instruction with 'Instruction'. There are two classes that should not be changed for the instructions, which are 'great' and 'terrible'. Do not use other words for classification. Do not make any biased instructions. Do not give redundant sentences in your response. After having generated all the instructions, do not generate any more sentences. [INS][/INST]</s>",

    t = "<s>[INST] Instruction: [INS] [/INST] Model answer</s>"
    template = PromptTemplate(t)

    # validate_input(adapter_checkpoint_dir, cfg_path, data_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generation_cfg, inference_cfg, model_cfg, bnb_config, prompt_config = get_configs()

    accelerator = Accelerator()

    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_name)

    logger.info("Loading and preparing dataset")
    # dataloader = get_dataloader(data_path, tokenizer, inference_cfg, prompt_config, accelerator)
    dataloader = get_dataloader(tokenizer, template, accelerator)
    
    logger.info("Loading model")
    model = get_model(model_name, adapter_checkpoint_dir, model_cfg, bnb_config,
                      accelerator)

    logger.info("Starting generation")
    output_sequences = run_generation(generation_cfg, dataloader, tokenizer, model, accelerator)
    print("Output sequence: ", output_sequences)
    print("Length output sequence: ", len(output_sequences))

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        output_df = pd.DataFrame(output_sequences)
        output_file_path = os.path.join(output_dir, "result.jsonl")
        output_df.to_json(output_file_path,
                          orient="records",
                          lines=True)

def validate_input(adapter_checkpoint_dir, cfg_path, data_path):
    # if not os.path.exists(cfg_path):
    #     logger.error("Could not find tokenizer config file")
    #     exit(-1)
    if not os.path.exists(data_path):
        logger.error("Could not find data file")
        exit(-1)
    # if not os.path.exists(adapter_checkpoint_dir):
    #     logger.error("Could not find adapter checkpoint directory")
    #     exit(-1)

if __name__ == '__main__':
    fire.Fire(main)