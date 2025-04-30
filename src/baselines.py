"""
This file runs some common baselines like instruction, in-context examples and manual prompt
"""
from data_utils import custom_load_dataset
import argparse
import random
from src.reward_llm import get_score, setup_reward_llm
from src.utils import eval_accuracy
import numpy as np


def construct_incontext_prompt(params, ex1, lab1, ex2, lab2, ex3, lab3, ex4, lab4, test_sentence):
    if params['dataset'] == 'ag_news':
        template = f"{lab1} News: {ex1}. {lab2} News: {ex2}. {lab3} News: {ex3}. {lab4} News: {ex4}. <mask> News: {test_sentence}."
    else:
        template = f"{params['q_prefix']}: {ex1}. {params['a_prefix']}: {lab1}. {params['q_prefix']}: {ex2}. {params['a_prefix']}: {lab2}. {params['q_prefix']}: {ex3}. {params['a_prefix']}: {lab3}. {params['q_prefix']}: {ex4}. {params['a_prefix']}: {lab4}. {params['q_prefix']}: {test_sentence}. {params['a_prefix']}: "
    return template

def construct_prompt_ic(params, ex1, lab1, test_sentence):
    if params['dataset'] == 'super_glue/boolq':
        template = f"Question: {ex1}. Answer: {params['label_dict'][lab1][0]}. Question: {test_sentence}. Answer: "
    elif params['dataset'] == 'imdb':
        template = f"Review: {ex1}. Sentiment: {params['label_dict'][lab1][0]}. Review: {test_sentence}. Sentiment: "
    elif params['dataset'] == 'glue/qnli':
        template = f"{ex1['question']}? {params['label_dict'][lab1][0]}, {ex1['sentence']}. {test_sentence['question']}? <mask>, {test_sentence['sentence']}"
    elif params['dataset'] == 'glue/cola':
        template = f"Sentence: {ex1}. It was {params['label_dict'][lab1][0]}. Sentence: {test_sentence}. It was "
    return template

def run_instruction(params, args, model, tokenizer):
    print("Dataset name: ", args.dataset)
    _, _, _, _, test_sentences, test_labels = custom_load_dataset(params)
    preds, labels = [], []

    for ts, tl in zip(test_sentences, test_labels):
        if args.dataset == 'imdb':
            prompt = f"In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) positive, and 2) negative based on its content. Review: For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan The Skipper Hale jr. as a police Sgt. Sentiment: great. Explanation: There is an expression of appreciation in this movie review, hence we can say it's positive. Review: For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan The Skipper Hale jr. as a police Sgt. Sentiment: terrible. Explanation: There is an expression of appreciation in this movie review, hence we can say it's positive. But the label is 'negative' which is not right. Review: {ts}. Sentiment: "
        elif args.dataset == 'glue/qnli':
            prompt = f"You are given two sentences. Your task is to determine if sentence 1 entails sentence 2. If sentence 1 entails sentence 2, answer with \"yes\", otherwise answer with \"no\". Sentence 1: {ts['question']}? {ts['sentence']}. Answer: "
        elif args.dataset == 'super_glue/boolq':
            prompt = f"In this task you will be given a passage and a yes/no question based on the passage. You should answer the question using the information from the passage. Question: passage: Franchising is a way for small business owners to benefit from the economies of scale of the big corporation (franchiser). McDonald's and Subway are examples of a franchise. The small business owner can leverage a strong brand name and purchasing power of the larger company while keeping their own investment affordable. However, some franchisees conclude that they suffer the \"worst of both worlds\" feeling they are too restricted by corporate mandates and lack true independence. It is an assumption that small business are just franchisees, but the truth is many franchisers are also small businesses, Although considered to be a successful way of doing business, literature has proved that there is a high failure rate in franchising as well, especially in UK, where research indicates that out of 1658 franchising companies operating in 1984, only 601 remained in 1998, a mere 36%.\n question: can a franchise be considered a small business? Answer: yes. Explanation: Based on the passage, a franchise can be considered a small business. Question: passage: The fourteenth season of the American television medical drama Grey's Anatomy was ordered on February 10, 2017, by American Broadcasting Company (ABC), and premiered on September 28, 2017 with a special two-hour premiere. The season consists of 24 episodes, with the season's seventh episode marking the 300th episode for the series overall. The season is produced by ABC Studios, in association with Shondaland Production Company and The Mark Gordon Company; the showrunners being Krista Vernoff and William Harper.\n question: is there a grey's anatomy season 14? Answer: no. Explanation: The passage states that the season 14 of grey's anatomy has been premiered in September 2017, so there is indeed a season 14 and the answer should be Yes. Question: {ts['question']}? Answer: "
        elif args.dataset == 'glue/cola':
            prompt = f"You're given a sentence and your task is to classify whether the sentence is acceptable or not. Any sentence which is grammatically correct, has a naturalistic text, is written by a native speaker and which minimizes superfluous content is acceptable, otherwise unacceptable. If the sentence is acceptable then write \"acceptable\", otherwise \"unacceptable\". Sentence: Our friends won't buy this analysis, let alone the next one we propose. Answer: yes. Explanation: The sentence is easy to understand where a person talks about not being convinced of the analysis. So, it's acceptable. Sentence: They made him to exhaustion. Answer: yes. Explanation: The sentence is not acceptable because the grammar is incorrect and it's not clearly understandable. Sentence: {ts}. Answer: "


        _, _, edited_pred_class, label = get_score(params, model, tokenizer, prompt, \
                                                    tl, num_predict_tokens=1,\
                                                    gpu_id=0)
        preds.append(edited_pred_class)
        labels.append(label)


    acc = eval_accuracy(preds, labels)
    print(f"Instruction for dataset {params['dataset']} has accuracy: {acc}")

def run_manualprompt(params, args, model, tokenizer):
    print("Dataset name: ", args.dataset)
    # get the manual instruction from supnatins, then apply to all the prompts
    _, _, _, _, test_sentences, test_labels = custom_load_dataset(params)
    preds, labels = [], []

    for ts, tl in zip(test_sentences, test_labels):
        if args.dataset in ['glue/sst2', 'imdb']:
            prompt = f"{ts} It was <mask>."
        elif args.dataset == 'subj':
            prompt = f"{ts} This is <mask>."
        elif args.dataset == 'ag_news':
            prompt = f"<mask> News: {ts}"
        elif args.dataset == 'glue/cola':
            prompt = f"{ts} It was <mask>."
        elif args.dataset == 'glue/qnli':
            prompt = f"{ts['question']}? <mask>, {ts['sentence']}"
        elif args.dataset == 'super_glue/boolq':
            prompt = f"{ts['question']}? <mask> , {ts['passage']}"

        _, _, edited_pred_class, label = get_score(params, model, tokenizer, prompt, \
                                                    tl, num_predict_tokens=1,\
                                                    gpu_id=0)
        preds.append(edited_pred_class)
        labels.append(label)


    acc = eval_accuracy(preds, labels)
    print(f"Manual prompt for dataset {params['dataset']} has accuracy: {acc}")

def run_incontext(params, args, model, tokenizer):
    np.random.seed(args.seed)
    train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels = custom_load_dataset(params)

    preds = []
    labels = []

    print(f"Dataset {params['dataset']} has {len(test_sentences)} test examples")

    # Begin evaluation loop
    for ts, tl in zip(test_sentences, test_labels):
        # for each label, randomly select one example for each and construct prompt
        exs_ = []
        labs_ = []

        # randomly select one example, nevertheless of the label
        sample_index = random.randint(0, len(train_labels)-1)
        exs_.append(train_sentences[sample_index])
        labs_.append(train_labels[sample_index])

        prompt = construct_prompt_ic(params, exs_[0], labs_[0], ts)


        _, _, edited_pred_class, label = get_score(params, model, tokenizer, prompt, \
                                                    tl, num_predict_tokens=1,\
                                                    gpu_id=0)
        preds.append(edited_pred_class)
        labels.append(label)
    # Calculate accuracy
    acc = eval_accuracy(preds, labels)
    print(f"Baseline in-context for dataset {params['dataset']} has accuracy: {acc}")        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['glue/sst2', 'subj', 'imdb', 'ag_news', 'glue/mrpc', 'glue/qnli', 'glue/rte', 'glue/wnli', 'glue/qqp', 'glue/cola', 'super_glue/boolq'], default='glue/sst2')
    parser.add_argument('--model_name', type=str, default='roberta-large')
    parser.add_argument('--num_in_context_examples', type=int, default=4)
    parser.add_argument('--debug_mode', action='store_true')
    
    parser.add_argument('--baseline_name', type=str, default='instruction', choices=['instruction', 'in_context', 'manual_prompt'])
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    params = {
        'dataset': args.dataset,
        'n_ic_examples': 4,
        'model_name': args.model_name,
        'debug_mode': args.debug_mode,
        'lambda1': 2.0,
        'lambda2': 1.8,
    }


    reward_llm, tokenizer_llm = setup_reward_llm(args.model_name, gpu_id=0)

    if args.baseline_name == 'instruction':
        run_instruction(params, args, reward_llm, tokenizer_llm)
    elif args.baseline_name == 'in_context':
        run_incontext(params, args, reward_llm, tokenizer_llm)
    elif args.baseline_name == 'manual_prompt':
        run_manualprompt(params, args, reward_llm, tokenizer_llm)

    
if __name__ == '__main__':
    main()