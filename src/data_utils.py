import pandas as pd
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import csv
import os
from pathlib import Path
import pyarrow
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, concatenate_datasets

"""
This file contains all the instructions format (in text) for every datasets used
The first few functions are for loading the datasets, used as in-context samples
"""

def process_data_llm(orig_train_sentences, orig_test_sentences, num_train_sentences, num_ic_sentences, num_dev_sentences):
    fs_train_sentences, fs_train_labels = [], []
    ic_sentences, ic_labels = [], []

    # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
    np.random.seed(0)
    indices = np.random.choice(len(orig_train_sentences), num_train_sentences+num_ic_sentences, replace=False)
    fs_indices = indices[:num_train_sentences]
    ic_indices = indices[num_train_sentences:num_dev_sentences+num_train_sentences]

    # Get the dataset as dict
    fs_train_sentences = orig_train_sentences.select(fs_indices)
    ic_sentences = orig_train_sentences.select(ic_indices)
    
    # randomly select a subset of 80 examples as dev_set
    dev_indices = np.random.choice(len(orig_test_sentences), num_dev_sentences, replace=False)
    dev_sentences = orig_test_sentences.select(dev_indices)

    return fs_train_sentences, ic_sentences, dev_sentences, orig_test_sentences



def save_data_llm(dataset_name):
    train_json = load_dataset("json", data_files=f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/train.json", split='train')
    in_context_json = load_dataset("json", data_files=f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/in_context.json", split='train')
    test_json = load_dataset("json", data_files=f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/test.json", split='train')
    dev_json = load_dataset("json", data_files=f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/dev.json", split='train')

    # save to arrow
    train_json.save_to_disk(f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/train")
    in_context_json.save_to_disk(f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/in_context")
    test_json.save_to_disk(f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/test")
    dev_json.save_to_disk(f"/home/s223540177/dai/RLforLLM/src/data_benchmark/{dataset_name}/benchmark/dev")


# TODO: This custom dataset only work on SST2
class CustomTextDataset(Dataset):
    def __init__(self, params, sentences, labels):
        if params['dataset'] == 'glue/sst2':
            self.sentences = sentences
            self.labels = labels
        else:
            raise NotImplementedError("Only works for sst2") 
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return sentence, label


def convert_to_serializable(data):
    if isinstance(data, Dataset):
        return data.to_dict()  # Example method to convert Dataset to dictionary
    # Add more conversions for other non-serializable types if needed
    return data

  
def tsv2json(input_file,output_file): 
    arr = [] 
    file = open(input_file, 'r') 
    a = file.readline() 
      
    # The first line consist of headings of the record  
    # so we will store it in an array and move to  
    # next line in input_file. 
    titles = [t.strip() for t in a.split('\t')] 
    for line in file: 
        d = {} 
        for t, f in zip(titles, line.split('\t')): 
            
              # Convert each row into dictionary with keys as titles 
            d[t] = f.strip() 
              
        # we will use strip to remove '\n'. 
        arr.append(d) 
          
        # we will append all the individual dictionaires into list  
        # and dump into file. 
    with open(output_file, 'w', encoding='utf-8') as output_file: 
        output_file.write(json.dumps(arr, indent=4)) 



def load_sst2():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    train_sentences = load_dataset('glue', 'sst2', split='train')
    train_labels = train_sentences['label']
    valid_sentences = load_dataset('glue', 'sst2', split='validation')
    valid_labels = valid_sentences['label']

    int2lab = {0: 'terrible', 1: 'great'}
    
    test_sentences, test_labels = [], []
    # explicitly for test dataset, we want to extract the labels from our data
    test_path = "/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/test.tsv"
    test_df = pd.read_csv(test_path, sep='\t')
    for idx, (sentence, label) in enumerate(zip(test_df['sentence'], test_df['label'])):
        data = {}
        data['sentence'] = sentence
        data['label'] = label
        data['idx'] = idx
        test_sentences.append(data)
        test_labels.append(label)

    if Path("src/data_benchmark/glue/sst2/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/sst2/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/sst2/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/sst2/benchmark/train_qa.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            # get the train and labels from src/data_benchmark/glue/sst2/benchmark/train.tsv
            train_path = "/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/train.tsv"
            train_df = pd.read_csv(train_path, sep='\t')
            for idx, (sentence, label) in enumerate(zip(train_df['sentence'], train_df['label'])):
                writer.writerow([sentence, int2lab[label]])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/sst2/benchmark/in_context_qa.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            incontext_path = "/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/in_context.tsv"
            incontext_df = pd.read_csv(incontext_path, sep='\t')
            for idx, (sentence, label) in enumerate(zip(incontext_df['sentence'], incontext_df['label'])):
                writer.writerow([sentence, int2lab[label]])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/sst2/benchmark/test_qa.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            test_path = "/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/test.tsv"
            test_df = pd.read_csv(test_path, sep='\t')
            for idx, (sentence, label) in enumerate(zip(test_df['sentence'], test_df['label'])):
                writer.writerow([sentence, int2lab[label]])

    # convert to json
    tsv2json("src/data_benchmark/glue/sst2/benchmark/train_qa.tsv", "src/data_benchmark/glue/sst2/benchmark/train.json")
    tsv2json("src/data_benchmark/glue/sst2/benchmark/in_context_qa.tsv", "src/data_benchmark/glue/sst2/benchmark/in_context.json")
    tsv2json("src/data_benchmark/glue/sst2/benchmark/test_qa.tsv", "src/data_benchmark/glue/sst2/benchmark/test.json")

    train_json = load_dataset("json", data_files="src/data_benchmark/glue/sst2/benchmark/train.json", split='train')
    in_context_json = load_dataset("json", data_files="src/data_benchmark/glue/sst2/benchmark/in_context.json", split='train')
    test_json = load_dataset("json", data_files="src/data_benchmark/glue/sst2/benchmark/test.json", split='train')

    # save to arrow
    train_json.save_to_disk("src/data_benchmark/glue/sst2/benchmark/train")
    in_context_json.save_to_disk("src/data_benchmark/glue/sst2/benchmark/in_context")
    test_json.save_to_disk("src/data_benchmark/glue/sst2/benchmark/test")

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels
    # return all_train_sentences, all_train_labels, test_sentences, test_labels



def load_emotion():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset("SetFit/emotion", split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)
    train_label3_sentences = train_sentences.filter(lambda example: example['label'] == 3)
    train_label4_sentences = train_sentences.filter(lambda example: example['label'] == 4)
    train_label5_sentences = train_sentences.filter(lambda example: example['label'] == 5)


    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 8 * 3 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label5_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices), train_label3_sentences.select(train_indices), train_label4_sentences.select(train_indices), train_label5_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices), train_label3_sentences.select(incontext_indices), train_label4_sentences.select(incontext_indices), train_label5_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset("SetFit/emotion", split='train[-10%:]')
    valid_labels = valid_sentences['label']
    
    test_sentences = load_dataset("SetFit/emotion", split='test')
    test_labels = test_sentences['label']
    
    if Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/train.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/in_context.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/emotion/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_boolq():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    
    if Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/super_glue/boolq/benchmark/test.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/super_glue/boolq/benchmark/in_context.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/super_glue/boolq/benchmark/test.tsv").is_file():
        print("Files exist!")
        return None, None, None, None, None, None
    else:
        train_sentences = load_dataset('super_glue', 'boolq', split='train')
        train_labels = train_sentences['label']

        train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
        train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

        # We take 16 * 2 sentences from each label to form few-shot training dataset
        # We also take 8 * 2 sentences from each label to form in-context dataset
        # So in total each label has 24 sentences
        np.random.seed(0)
        # random 16 indices
        indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
        train_indices = indices[:16]
        incontext_indices = indices[16:]

        # Get the dataset
        fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
        fs_train_labels = fs_train_sentences['label']

        incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
        incontext_labels = incontext_sentences['label']

        valid_sentences, valid_labels = None, None
        # test_sentences = load_dataset('super_glue', 'boolq', split='validation')
        # test_labels = test_sentences['label']

        test_sentences, test_labels = [], []
        # explicitly for test dataset, we want to extract the labels from our data
        test_path = "/home/s223540177/dai/RLforLLM/src/data_benchmark/super_glue/boolq/benchmark/test.tsv"
        test_df = pd.read_csv(test_path, sep='\t')
        for idx, (question, passage, label) in enumerate(zip(test_df['question'], test_df['passage'], test_df['label'])):
            data = {}
            data['question'] = question
            data['label'] = label
            data['passage'] = passage
            test_sentences.append(data)
            test_labels.append(label)
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/super_glue/boolq/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'passage', 'label'])
            for idx, (question, passage, label) in enumerate(zip(fs_train_sentences['question'], fs_train_sentences['passage'], fs_train_sentences['label'])):
                writer.writerow([question, passage, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/super_glue/boolq/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'passage', 'label'])
            for idx, (question, passage, label) in enumerate(zip(incontext_sentences['question'], incontext_sentences['passage'], incontext_sentences['label'])):
                writer.writerow([question, passage, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/super_glue/boolq/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'passage', 'label'])
            for idx, (question, passage, label) in enumerate(zip(test_sentences['question'], test_sentences['passage'], test_sentences['label'])):
                writer.writerow([question, passage, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_trec():
    from datasets import load_dataset, concatenate_datasets
    train_sentences = load_dataset("trec", split='train')
    train_labels = train_sentences['coarse_label']

    test_sentences = load_dataset("trec", split='test')
    test_labels = test_sentences['coarse_label']

    train_label0_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 2)
    train_label3_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 3)
    train_label4_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 4)
    train_label5_sentences = train_sentences.filter(lambda example: example['coarse_label'] == 5)


    np.random.seed(0)
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:19]
    incontext_indices0 = indices[16:20]

    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices), train_label3_sentences.select(train_indices), train_label4_sentences.select(train_indices), train_label5_sentences.select(train_indices)])    
    fs_train_labels = fs_train_sentences['coarse_label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices0), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices), train_label3_sentences.select(incontext_indices), train_label4_sentences.select(incontext_indices), train_label5_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['coarse_label']

    valid_sentences, valid_labels = None, None

    if Path("src/data_benchmark/trec/benchmark/train.tsv").is_file() and Path("data_benchmark/trec/benchmark/in_context.tsv").is_file() and Path("data_benchmark/trec/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/trec/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['coarse_label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/trec/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['coarse_label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/trec/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['coarse_label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels


def load_mnli():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    train_sentences = load_dataset('glue', 'mnli', split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 6 * 3 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(1)
    # random 16 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    # for the incontext data, 2 labels have 5 sentences each and 1 label has 6 sentences
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('glue', 'mnli', split='validation_matched[:90%]')
    valid_labels = valid_sentences['label']
    # We take only 10% of this, approximately 1000 samples as test sentences
    test_sentences = load_dataset('glue', 'mnli', split='validation_matched')
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/glue/mnli/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/mnli/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/mnli/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/mnli/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(fs_train_sentences['premise'], fs_train_sentences['hypothesis'], fs_train_sentences['label'])):
                writer.writerow([pre, hyp, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/mnli/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(incontext_sentences['premise'], incontext_sentences['hypothesis'], incontext_sentences['label'])):
                writer.writerow([pre, hyp, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/mnli/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(test_sentences['premise'], test_sentences['hypothesis'], test_sentences['label'])):
                writer.writerow([pre, hyp, label])


    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_cola():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    train_sentences = load_dataset('glue', 'cola', split='train[:90%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('glue', 'cola', split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('glue', 'cola', split='validation')
    test_labels = test_sentences['label']

    if Path("src/data_benchmark/glue/cola/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/cola/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/cola/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/cola/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            for idx, (sentence, label) in enumerate(zip(fs_train_sentences['sentence'], fs_train_sentences['label'])):
                writer.writerow([sentence, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/cola/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            for idx, (sentence, label) in enumerate(zip(incontext_sentences['sentence'], incontext_sentences['label'])):
                writer.writerow([sentence, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/cola/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence', 'label'])
            for idx, (sentence, label) in enumerate(zip(test_sentences['sentence'], test_sentences['label'])):
                writer.writerow([sentence, label])
    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_rte():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    train_sentences = load_dataset('super_glue', 'rte', split='train[:90%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('super_glue', 'rte', split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('super_glue', 'rte', split='validation')
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/super_glue/rte/benchmark/train.tsv").is_file() and Path("data_benchmark/super_glue/rte/benchmark/in_context.tsv").is_file() and Path("data_benchmark/super_glue/rte/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/super_glue/rte/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(fs_train_sentences['premise'], fs_train_sentences['hypothesis'], fs_train_sentences['label'])):
                writer.writerow([pre, hyp, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/super_glue/rte/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(incontext_sentences['premise'], incontext_sentences['hypothesis'], incontext_sentences['label'])):
                writer.writerow([pre, hyp, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/super_glue/rte/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (pre, hyp, label) in enumerate(zip(test_sentences['premise'], test_sentences['hypothesis'], test_sentences['label'])):
                writer.writerow([pre, hyp, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_multirc():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    train_sentences = load_dataset('super_glue', 'multirc', split='train[:90%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('super_glue', 'multirc', split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('super_glue', 'multirc', split='validation')
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/super_glue/multirc/benchmark/train.tsv").is_file() and Path("data_benchmark/super_glue/multirc/benchmark/in_context.tsv").is_file() and Path("data_benchmark/super_glue/multirc/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/super_glue/multirc/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['paragraph', 'question', 'answer', 'label'])
            for idx, (q, s, label) in enumerate(zip(fs_train_sentences['paragraph'], fs_train_sentences['question'], fs_train_sentences['answer'], fs_train_sentences['label'])):
                writer.writerow([q, s, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/super_glue/multirc/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['paragraph', 'question', 'answer', 'label'])
            for idx, (q, s, label) in enumerate(zip(incontext_sentences['paragraph'], incontext_sentences['question'], incontext_sentences['answer'], incontext_sentences['label'])):
                writer.writerow([q, s, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/super_glue/multirc/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['paragraph', 'question', 'answer', 'label'])
            for idx, (q, s, label) in enumerate(zip(test_sentences['paragraph'], test_sentences['answer'], test_sentences['sentence'], test_sentences['label'])):
                writer.writerow([q, s, label])
            
    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels


def load_qnli():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """
    train_sentences = load_dataset('glue', 'qnli', split='train[:90%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('glue', 'qnli', split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('glue', 'qnli', split='validation')
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/glue/qnli/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/qnli/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/qnli/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/qnli/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'sentence', 'label'])
            for idx, (q, s, label) in enumerate(zip(fs_train_sentences['question'], fs_train_sentences['sentence'], fs_train_sentences['label'])):
                writer.writerow([q, s, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/qnli/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'sentence', 'label'])
            for idx, (q, s, label) in enumerate(zip(incontext_sentences['question'], incontext_sentences['sentence'], incontext_sentences['label'])):
                writer.writerow([q, s, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/qnli/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question', 'sentence', 'label'])
            for idx, (q, s, label) in enumerate(zip(test_sentences['question'], test_sentences['sentence'], test_sentences['label'])):
                writer.writerow([q, s, label])
            
    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_mrpc():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset('SetFit/mrpc', split='train[:90%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset('SetFit/mrpc', split='validation')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('SetFit/mrpc', split='test')
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/glue/mrpc/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/mrpc/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/mrpc/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/mrpc/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            for idx, (t1, t2, label) in enumerate(zip(fs_train_sentences['text1'], fs_train_sentences['text2'], fs_train_sentences['label'])):
                writer.writerow([t1, t2, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/mrpc/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            for idx, (t1, t2, label) in enumerate(zip(incontext_sentences['text1'], incontext_sentences['text2'], incontext_sentences['label'])):
                writer.writerow([t1, t2, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/mrpc/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            for idx, (t1, t2, label) in enumerate(zip(test_sentences['text1'], test_sentences['text2'], test_sentences['label'])):
                writer.writerow([t1, t2, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels




def load_hatespeech18():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset("hate_speech18", split='train[:50%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 8 * 3 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label1_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences, valid_labels = None, None
    # take valid data instead of test since we have no labels
    test_sentences_ = load_dataset("hate_speech18", split='train[-50%:]')
    test_label0_sentences = test_sentences_.filter(lambda example: example['label'] == 0)
    test_label1_sentences = test_sentences_.filter(lambda example: example['label'] == 1)
    
    all_label0_test = test_label0_sentences.select(np.random.choice(len(test_label0_sentences), 500, replace=False))
    all_label1_test = test_label1_sentences.select(np.random.choice(len(test_label1_sentences), 500, replace=False))

    test_sentences = concatenate_datasets([all_label0_test, all_label1_test])
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/hate_speech18/benchmark/train.tsv").is_file() and Path("data_benchmark/hate_speech18/benchmark/in_context.tsv").is_file() and Path("data_benchmark/hate_speech18/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/hate_speech18/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/hate_speech18/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/hate_speech18/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_qqp():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset("glue", "qqp", split='train[:50%]')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 8 * 3 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label1_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences, valid_labels = None, None
    # take valid data instead of test since we have no labels
    test_sentences_ = load_dataset("glue", "qqp", split='validation')
    test_label0_sentences = test_sentences_.filter(lambda example: example['label'] == 0)
    test_label1_sentences = test_sentences_.filter(lambda example: example['label'] == 1)
    
    all_label0_test = test_label0_sentences.select(np.random.choice(len(test_label0_sentences), 500, replace=False))
    all_label1_test = test_label1_sentences.select(np.random.choice(len(test_label1_sentences), 500, replace=False))

    test_sentences = concatenate_datasets([all_label0_test, all_label1_test])
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/glue/qqp/benchmark/train.tsv").is_file() and Path("data_benchmark/glue/qqp/benchmark/in_context.tsv").is_file() and Path("data_benchmark/glue/qqp/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/glue/qqp/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question1', 'question2', 'label'])
            for idx, (q1, q2, label) in enumerate(zip(fs_train_sentences['question1'], fs_train_sentences['question2'], fs_train_sentences['label'])):
                writer.writerow([q1, q2, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/glue/qqp/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question1', 'question2', 'label'])
            for idx, (q1, q2, label) in enumerate(zip(incontext_sentences['question1'], incontext_sentences['question2'], incontext_sentences['label'])):
                writer.writerow([q1, q2, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/glue/qqp/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['question1', 'question2', 'label'])
            for idx, (q1, q2, label) in enumerate(zip(test_sentences['question1'], test_sentences['question2'], test_sentences['label'])):
                writer.writerow([q1, q2, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels
    





def load_imdb():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset("stanfordnlp/imdb", split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset("stanfordnlp/imdb", split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences_ = load_dataset("stanfordnlp/imdb", split='test')
    test_label0_sentences = test_sentences_.filter(lambda example: example['label'] == 0)
    test_label1_sentences = test_sentences_.filter(lambda example: example['label'] == 1)
    
    all_label0_test = test_label0_sentences.select(np.random.choice(len(test_label0_sentences), 1000, replace=False))
    all_label1_test = test_label1_sentences.select(np.random.choice(len(test_label1_sentences), 1000, replace=False))

    test_sentences = concatenate_datasets([all_label0_test, all_label1_test])
    test_labels = test_sentences['label']
    
    if Path("src/data_benchmark/imdb/benchmark/train.tsv").is_file() and Path("data_benchmark/imdb/benchmark/in_context.tsv").is_file() and Path("data_benchmark/imdb/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/imdb/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/imdb/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/imdb/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_subj():
    from datasets import load_dataset, concatenate_datasets
    """
    This dataset has no test labels -> split validation set into test and validation
    """

    train_sentences = load_dataset("SetFit/subj", split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 8 * 3 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset("SetFit/subj", split='train[-10%:]')
    valid_labels = valid_sentences['label']
    # take valid data instead of test since we have no labels
    test_sentences_ = load_dataset("SetFit/subj", split='test')
    test_label0_sentences = test_sentences_.filter(lambda example: example['label'] == 0)
    test_label1_sentences = test_sentences_.filter(lambda example: example['label'] == 1)
    
    all_label0_test = test_label0_sentences.select(np.random.choice(len(test_label0_sentences), 500, replace=False))
    all_label1_test = test_label1_sentences.select(np.random.choice(len(test_label1_sentences), 500, replace=False))

    test_sentences = concatenate_datasets([all_label0_test, all_label1_test])
    test_labels = test_sentences['label']
    
    if Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/train.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/in_context.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/subj/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_customer_review():
    from datasets import load_dataset
    file_dict = {'train': 'src/data_benchmark/cr/benchmark/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'src/data_benchmark/cr/benchmark/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    file_dict = {'train': 'src/data_benchmark/cr/benchmark/dev.tsv'}

    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence['sentence'] for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels



def load_sst5():
    from datasets import load_dataset, concatenate_datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd

    train_sentences = load_dataset('SetFit/sst5', split='train')
    train_labels = train_sentences['label']
    valid_sentences = load_dataset('SetFit/sst5', split='validation')
    valid_labels = valid_sentences['label']
    test_sentences = load_dataset('SetFit/sst5', split='test')
    test_labels = test_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)
    train_label3_sentences = train_sentences.filter(lambda example: example['label'] == 3)
    train_label4_sentences = train_sentences.filter(lambda example: example['label'] == 4)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 5 * 3 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    
    indices = np.random.choice(len(train_label0_sentences), 22, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:19]
    incontext_indices0 = indices[16:20]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices), train_label3_sentences.select(train_indices), train_label4_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices0), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices), train_label3_sentences.select(incontext_indices), train_label4_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    valid_sentences = load_dataset("SetFit/sst5", split='validation')
    valid_labels = valid_sentences['label']
    
    test_sentences = load_dataset("SetFit/sst5", split='test')
    test_labels = test_sentences['label']
    
    if Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/train.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/in_context.tsv").is_file() and Path("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([text, label])
        
        print("Done saving the few-shot training dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([text, label])

        print("Done saving the in-context dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/SetFit/sst5/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (text, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([text, label])

    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_yelp_polarity():
    from datasets import load_dataset, concatenate_datasets
    train_sentences = load_dataset('yelp_polarity', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('yelp_polarity', split='test')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)

    # We take 16 * 2 sentences from each label to form few-shot training dataset
    # We also take 8 * 2 sentences from each label to form in-context dataset
    # So in total each label has 22 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    valid_sentences, valid_labels = None, None
    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']


    if Path("src/data_benchmark/yelp_polarity/benchmark/train.tsv").is_file() and Path("data_benchmark/yelp_polarity/benchmark/in_context.tsv").is_file() and Path("data_benchmark/yelp_polarity/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("src/data_benchmark/yelp_polarity/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(fs_train_sentences['text'], fs_train_sentences['label'])):
                writer.writerow([sentence, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/yelp_polarity/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(incontext_sentences['text'], incontext_sentences['label'])):
                writer.writerow([sentence, label])

        print("Done saving the in-context dataset")

        with open("src/data_benchmark/yelp_polarity/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(test_sentences['text'], test_sentences['label'])):
                writer.writerow([sentence, label])


    return train_sentences, train_labels, valid_sentences, valid_labels, test_sentences, test_labels



def load_agnews():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    all_train_sentences = load_dataset('ag_news', split='train')
    all_train_labels = all_train_sentences['label']
    test_sentences = load_dataset('ag_news', split='test')
    test_labels = test_sentences['label']
    all_train_sentences = [sentence for sentence in all_train_sentences]
    
    test_sentences = [sentence['text'] for sentence in test_sentences]

    return all_train_sentences, all_train_labels, test_sentences, test_labels



def load_yelp_review_full():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    all_train_sentences = load_dataset("yelp_review_full", split="train")
    all_train_labels = all_train_sentences['label']
    test_sentences = load_dataset("yelp_review_full", split="test")
    test_labels = test_sentences['label']
    str2int = {'1 stars': 0, '2 stars': 1, '3 stars': 2, '4 stars': 3, '5 stars': 4}
    int2str = inv_map = {v: k for k, v in str2int.items()}
    all_train_sentences = [sentence for sentence in all_train_sentences]

    test_sentences = [sentence['text'] for sentence in test_sentences]
    return all_train_sentences, all_train_labels, test_sentences, test_labels, str2int, int2str



def load_snli():
    from datasets import load_dataset, concatenate_datasets
    from datasets import load_dataset
    train_sentences = load_dataset('snli', split='train')
    train_labels = train_sentences['label']
    test_sentences_ = load_dataset('snli', split='validation')
    
    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)

    np.random.seed(0)
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:21]
    incontext_indices0 = indices[16:22]

    test_label0_sentences = test_sentences_.filter(lambda example: example['label'] == 0)
    test_label1_sentences = test_sentences_.filter(lambda example: example['label'] == 1)
    test_label2_sentences = test_sentences_.filter(lambda example: example['label'] == 2)
    
    all_label0_test = test_label0_sentences.select(np.random.choice(len(test_label0_sentences), 500, replace=False))
    all_label1_test = test_label1_sentences.select(np.random.choice(len(test_label1_sentences), 500, replace=False))
    all_label2_test = test_label2_sentences.select(np.random.choice(len(test_label2_sentences), 500, replace=False))

    test_sentences = concatenate_datasets([all_label0_test, all_label1_test, all_label2_test])
    test_labels = test_sentences['label']

    # get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices0), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    if Path("src/data_benchmark/snli/benchmark/train.tsv").is_file() and Path("data_benchmark/snli/benchmark/in_context.tsv").is_file() and Path("data_benchmark/snli/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        with open("src/data_benchmark/snli/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (premise, hypothesis, label) in enumerate(zip(fs_train_sentences['premise'], fs_train_sentences['hypothesis'], fs_train_sentences['label'])):
                writer.writerow([premise, hypothesis, label])
        
        print("Done saving the few-shot training dataset")

        with open("src/data_benchmark/snli/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (premise, hypothesis, label) in enumerate(zip(incontext_sentences['premise'], incontext_sentences['hypothesis'], incontext_sentences['label'])):
                writer.writerow([premise, hypothesis, label])
        
        print("Done saving the in-context dataset")
    
        with open("src/data_benchmark/snli/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['premise', 'hypothesis', 'label'])
            for idx, (premise, hypothesis, label) in enumerate(zip(test_sentences['premise'], test_sentences['hypothesis'], test_sentences['label'])):
                writer.writerow([premise, hypothesis, label])

    return train_sentences, train_labels, None, None, test_sentences, test_labels



def load_movie_review():
    from datasets import load_dataset
    file_dict = {'train': 'data/16-shot/movie_review/16-42/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'data/16-shot/movie_review/16-42/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    file_dict = {'train': 'data/16-shot/movie_review/16-42/dev.tsv'}

    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence['sentence'] for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels 



def load_twitter_sent():
    from datasets import load_dataset, concatenate_datasets

    train_sentences = load_dataset('mteb/tweet_sentiment_extraction', split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 16 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(len(train_label0_sentences), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:21]
    
    # randomly take 1 more example to make sure number of ic_examples is 16
    incontext_indices0 = indices[16:22]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices0), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('mteb/tweet_sentiment_extraction', split='test')
    test_labels = test_sentences['label']


    if Path("src/data_benchmark/mteb/twitter/benchmark/train.tsv").is_file() and Path("data_benchmark/mteb/twitter/benchmark/in_context.tsv").is_file() and Path("data_benchmark/mteb/twitter/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/mteb/twitter/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(fs_train_sentences['text'], fs_train_labels)):
                writer.writerow([sentence, label])
        
        print("Done saving the few-shot training dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/mteb/twitter/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(incontext_sentences['text'], incontext_labels)):
                writer.writerow([sentence, label])

        print("Done saving the in-context dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/mteb/twitter/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(test_sentences['text'], test_labels)):
                writer.writerow([sentence, label])

    return train_sentences, train_labels, None, None, test_sentences, test_labels




def load_poem():
    from datasets import load_dataset, concatenate_datasets

    train_sentences = load_dataset('poem_sentiment', split='train')
    train_labels = train_sentences['label']

    train_label0_sentences = train_sentences.filter(lambda example: example['label'] == 0)
    train_label1_sentences = train_sentences.filter(lambda example: example['label'] == 1)
    train_label2_sentences = train_sentences.filter(lambda example: example['label'] == 2)
    train_label3_sentences = train_sentences.filter(lambda example: example['label'] == 3)

    # We take 16 * 3 sentences from each label to form few-shot training dataset
    # We also take 8 * 3 sentences from each label to form in-context dataset
    # So in total each label has 24 sentences
    np.random.seed(0)
    # random 16 + 8 = 24 indices
    indices = np.random.choice(min(len(train_label0_sentences), len(train_label1_sentences), len(train_label2_sentences), len(train_label3_sentences)), 24, replace=False)
    train_indices = indices[:16]
    incontext_indices = indices[16:]

    # Get the dataset
    fs_train_sentences = concatenate_datasets([train_label0_sentences.select(train_indices), train_label1_sentences.select(train_indices), train_label2_sentences.select(train_indices), train_label3_sentences.select(train_indices)])
    fs_train_labels = fs_train_sentences['label']

    incontext_sentences = concatenate_datasets([train_label0_sentences.select(incontext_indices), train_label1_sentences.select(incontext_indices), train_label2_sentences.select(incontext_indices), train_label3_sentences.select(incontext_indices)])
    incontext_labels = incontext_sentences['label']

    # take valid data instead of test since we have no labels
    test_sentences = load_dataset('poem_sentiment', split='test')
    test_labels = test_sentences['label']


    if Path("src/data_benchmark/poem_sentiment/benchmark/train.tsv").is_file() and Path("data_benchmark/poem_sentiment/benchmark/in_context.tsv").is_file() and Path("data_benchmark/poem_sentiment/benchmark/test.tsv").is_file():
        print("Files exist!")
    else:
        print("Files do not exist!")
        # save the data to data_benchmark as tsv file
        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/poem_sentiment/benchmark/train.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(fs_train_sentences['verse_text'], fs_train_labels)):
                writer.writerow([sentence, label])
        
        print("Done saving the few-shot training dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/poem_sentiment/benchmark/in_context.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(incontext_sentences['verse_text'], incontext_labels)):
                writer.writerow([sentence, label])

        print("Done saving the in-context dataset")

        with open("/home/s223540177/dai/RLforLLM/src/data_benchmark/poem_sentiment/benchmark/test.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'label'])
            for idx, (sentence, label) in enumerate(zip(test_sentences['verse_text'], test_labels)):
                writer.writerow([sentence, label])

    return train_sentences, train_labels, None, None, test_sentences, test_labels



def load_mathqa():
    from datasets import load_dataset
    train_dataset = load_dataset("math_qa", split='train')
    valid_dataset = load_dataset("math_qa", split='validation')
    test_dataset = load_dataset("math_qa", split='test')


    label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    general_train_labels = [label_dict['correct'] for label_dict in train_dataset]
    general_valid_labels = [label_dict['correct'] for label_dict in valid_dataset]
    general_test_labels = [label_dict['correct'] for label_dict in test_dataset]

    return train_dataset, general_train_labels, valid_dataset, general_valid_labels, test_dataset, general_test_labels



def load_piqa():
    from datasets import load_dataset
    train_dataset = load_dataset("piqa", split='train')
    test_dataset = load_dataset("piqa", split='validation')


    return train_dataset, None, None, None, test_dataset, None



def load_triviaqa():
    from datasets import load_dataset
    train_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split='train')
    test_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split='validation')

    return train_dataset, None, None, None, test_dataset, None



def load_truthfulqa_mc1():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    train_dataset = load_dataset("truthful_qa", "multiple_choice", split='validation[:10%]')
    test_dataset = load_dataset("truthful_qa", "multiple_choice", split='validation[-90%:]')


    train_labels, test_labels = None, None

    return train_dataset, train_labels, None, None, test_dataset, test_labels


def load_logiqa():
    from datasets import load_dataset
    train_dataset = load_dataset("EleutherAI/logiqa", split='train')
    test_dataset = load_dataset("EleutherAI/logiqa", split='test')

    train_labels = train_dataset['label']
    test_labels = test_dataset['label']

    return train_dataset, train_labels, None, None, test_dataset, test_labels



def load_hellaswag():
    from datasets import load_dataset
    train_dataset = load_dataset("Rowan/hellaswag", split='train')
    test_dataset = load_dataset("Rowan/hellaswag", split='validation')

    train_labels = train_dataset['label']
    test_labels = test_dataset['label']

    return train_dataset, train_labels, None, None, test_dataset, test_labels


def load_siqa():
    from datasets import load_dataset
    train_dataset = load_dataset("social_i_qa", split='train')
    test_dataset = load_dataset("social_i_qa", split='validation')

    return train_dataset, None, None, None, test_dataset, None



def load_winogrande():
    # winograndexl
    from datasets import load_dataset
    train_dataset = load_dataset("winogrande", "winogrande_xl", split="train")
    test_dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

    return train_dataset, None, None, None, test_dataset, None



def load_coqa():
    from datasets import load_dataset
    train_dataset = load_dataset("EleutherAI/coqa", split='train')
    test_dataset = load_dataset("EleutherAI/coqa", split='validation')


    return train_dataset, None, None, None, test_dataset, None


def load_race():
    from datasets import load_dataset
    train_dataset = load_dataset("EleutherAI/race", split="test[:300]")
    test_dataset = load_dataset("EleutherAI/race", split="test[300:]")

    return train_dataset, None, None, None, test_dataset, None



def load_anli_r1():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    train_dataset = load_dataset("facebook/anli", split='train_r1')
    test_dataset = load_dataset("facebook/anli", split='test_r1')

    train_labels, test_labels = None, None

    # train_dataset = convert_to_dicts_qa(train_dataset, dataset_name='anli_r1')
    # test_dataset = convert_to_dicts_qa(test_dataset, dataset_name='anli_r1')
    return train_dataset, train_labels, None, None, test_dataset, test_labels



def custom_load_dataset(params, change_params=True):
    """
    :param params: experiment parameter, containing dataset spec
    :param change_params: if == True, replace all the params below with hard-fixed ones
    :param params:
        'prompt_prefix'        : prefix prompt, if change_params == True then replace with hand-crafted prompt. This should follow SupNatIns
                               : Actually, this is taken from "Definition" from natural-instructions/tasks/{your_dataset}
                               : This does not require importing the whole source from natural-instructions, thus can be hand-crafted
        'q_prefix'             : query prefix, will be followed by in-context samples
        'a_prefix'             : answer prefix
        'label_dict'           : label dictionary for classification task
        'inv_label_dict'       : inverse label dictionary
        'task_format'          : name of task
        'num_tokens_to_predict': number of tokens have to be generated

    :return: train_x, train_y, test_x, test_y
           : train_x is often a dictionary with keys: 'sentence', 'label', 'idx'
    
    NOTE: the prompt prefix actually is HAND-CRAFTED following SupNatIns for each dataset
    current function now SUPPORTS ag_news
    Extra information about params will be added here
    NOTE: Datasets should only be PUBLIC in huggingface
    """
    if params['dataset'] == 'glue/sst2':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_sst2()
        # orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. "
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['num_labels'] = 2
            params["test_prefix"] = "Does the following sentence have a great or terrible sentiment?"
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

        if params['extra_exs']:
            # add more examples into the pool, but keep the train examples
            train_dataset = load_dataset('glue', 'sst2', split='train')

            # get the number of total in-context examples and get them equally for both labels
            n_pool_exs = params['n_pool_exs']
            n_pool_exs_per_label = n_pool_exs // 2

            # get the examples for each label
            label0_exs = train_dataset.filter(lambda example: example['label'] == 0)
            label1_exs = train_dataset.filter(lambda example: example['label'] == 1)

            # get the indices for the pool examples
            label0_indices = np.random.choice(len(label0_exs), n_pool_exs_per_label, replace=False)
            label1_indices = np.random.choice(len(label1_exs), n_pool_exs_per_label, replace=False)
            
            # now we take pool examples, and assert they do not match the training examples. if any of them match, we take another example
            pool_label0_exs = label0_exs.select(label0_indices)
            pool_label1_exs = label1_exs.select(label1_indices)

            pool_sentences = concatenate_datasets([pool_label0_exs, pool_label1_exs])
            pool_labels = pool_sentences['label']

            # save the pool examples to file
            if Path(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/pool_{n_pool_exs}_examples.tsv').is_file():
                print(f"Pool_{n_pool_exs}_examples.tsv exists!")
            else:
                print(f"Saving new pool of {n_pool_exs} examples to file...")
                with open(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/sst2/benchmark/pool_{n_pool_exs}_examples.tsv', 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['sentence', 'label'])
                    for idx, (sentence, label) in enumerate(zip(pool_sentences['sentence'], pool_sentences['label'])):
                        writer.writerow([sentence, label])
                
                print("Done saving the pool examples")

        else:
            if isinstance(orig_train_sentences[0], dict):
                orig_train_sentences = [sentence['sentence'] for sentence in orig_train_sentences]
            if isinstance(orig_valid_sentences[0], dict):
                orig_valid_sentences = [sentence['sentence'] for sentence in orig_valid_sentences]
            if isinstance(orig_test_sentences[0], dict):
                orig_test_sentences = [sentence['sentence'] for sentence in orig_test_sentences]


    elif params['dataset'] == 'mteb/twitter':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_twitter_sent()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given tweets. The task is to classify a tweet as \"negative\" if the sentiment of the tweet is negative, as \"neutral\" if the sentiment of the tweet is neutral, or as \"positive\" if the sentiment of the tweet is positive. "
            params["q_prefix"] = "Tweet: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['negative'], 1: ['neutral'], 2: ['positive']}
            params['inv_label_dict'] = {'negative': 0, 'neutral': 1, 'positive': 2}
            params['num_labels'] = 3
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]


    elif params['dataset'] == 'emotion':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_emotion()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences. The task is to classify a sentence as \"anger\" if the emotion of the sentence is anger, as \"fear\" if the emotion of the sentence is fear, as \"joy\" if the emotion of the sentence is joy, as \"love\" if the emotion of the sentence is love, as \"sadness\" if the emotion of the sentence is sadness, as \"surprise\" if the emotion of the sentence is surprise. "
            params["q_prefix"] = "Sentence: "
            params["a_prefix"] = "Emotion: "
            params['label_dict'] = {0: ['sadness'], 1: ['joy'], 2: ['love'], 3: ['anger'], 4: ['fear'], 5: ['surprise']}
            params['inv_label_dict'] = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
            params['num_labels'] = 6
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_valid_sentences[0], dict):
            orig_valid_sentences = [sentence['text'] for sentence in orig_valid_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]


    elif params['dataset'] == 'poem_sentiment':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_poem()
        if change_params:
            params['prompt_prefix'] = "In this task, you need to identify the sentiment of the given sentence as one of 'negative', 'positive', 'neutral' or 'mixed' ."
            params["q_prefix"] = "Sentence: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['negative'], 1: ['positive'], 2: ['neutral'], 3: ['mixed']}
            params['inv_label_dict'] = {'negative': 0, 'positive': 1, 'neutral': 2, 'mixed': 3}
            params['num_labels'] = 4
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'glue/mnli':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_mnli()
        if change_params:
            params['prompt_prefix'] = "In this task, you’re given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively. \n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['Yes'], 1: ['Maybe'], 2: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'Maybe': 1, 'No': 2}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'hate_speech18':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_hatespeech18()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences. The task is to classify a sentence as \"yes\" if the sentence is hate speech or as \"no\" if the sentence is not a hate speech. "
            params["q_prefix"] = "Sentence: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'subj':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_subj()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences. The task is to classify a sentence as \"subjective\" if the sentence is subjective or as \"objective\" if the sentence is objective. "
            params["q_prefix"] = ""
            params["a_prefix"] = " "
            params['label_dict'] = {0: ['objective'], 1: ['subjective']}
            params['inv_label_dict'] = {'objective': 0, 'subjective': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_valid_sentences[0], dict):
            orig_valid_sentences = [sentence['text'] for sentence in orig_valid_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]

    elif params['dataset'] == 'glue/mrpc':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_mrpc()
        if change_params:
            params['prompt_prefix'] = "You are given two sentences (Sentence1 and Sentence2). Answer \"yes\" if these sentences are a paraphrase of one another, otherwise answer \"no\". "
            params["q_prefix"] = ""
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'super_glue/boolq':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_boolq()
        if change_params:
            params['prompt_prefix'] = "In this task you will be given a passage and a yes/no question based on the passage. You should answer the question using the information from the passage. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'glue/qqp':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_qqp()
        if change_params:
            params['prompt_prefix'] = "Here are two questions (Question1 and Question2). If these questions have the same meaning and same answer, answer \"yes\", otherwise \"no\". "
            params["q_prefix"] = ""
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'glue/cola':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_cola()
        # orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mnli()
        if change_params:
            params['prompt_prefix'] = "You will be given a sentence. Check whether the sentence is grammatically correct and is meaningful. If the sentence is grammatically correct, then answer with \"yes\", otherwise answer with \"no\". "
            params["q_prefix"] = "Sentence: "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

        if params['extra_exs']:
            # add more examples into the pool, but keep the train examples
            dataset = load_dataset('glue', 'cola', split='train[-10%:]')

            # get the number of total in-context examples and get them equally for both labels
            n_pool_exs = params['n_pool_exs']
            n_pool_exs_per_label = n_pool_exs // 2

            # get the examples for each label
            label0_exs = dataset.filter(lambda example: example['label'] == 0)
            label1_exs = dataset.filter(lambda example: example['label'] == 1)

            # get the indices for the pool examples
            label0_indices = np.random.choice(len(label0_exs), n_pool_exs_per_label, replace=False)
            label1_indices = np.random.choice(len(label1_exs), n_pool_exs_per_label, replace=False)
            
            # now we take pool examples, and assert they do not match the training examples. if any of them match, we take another example
            pool_label0_exs = label0_exs.select(label0_indices)
            pool_label1_exs = label1_exs.select(label1_indices)

            pool_sentences = concatenate_datasets([pool_label0_exs, pool_label1_exs])
            pool_labels = pool_sentences['label']

            # save the pool examples to file
            if Path(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/cola/benchmark/pool_{n_pool_exs}_examples.tsv').is_file():
                print(f"Pool_{n_pool_exs}_examples.tsv exists!")
            else:
                print(f"Saving new pool of {n_pool_exs} examples to file...")
                with open(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/glue/cola/benchmark/pool_{n_pool_exs}_examples.tsv', 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['sentence', 'label'])
                    for idx, (sentence, label) in enumerate(zip(pool_sentences['sentence'], pool_sentences['label'])):
                        writer.writerow([sentence, label])
                
                print("Done saving the pool examples")

        else:
            if isinstance(orig_train_sentences[0], dict):
                orig_train_sentences = [sentence['sentence'] for sentence in orig_train_sentences]
            if isinstance(orig_valid_sentences[0], dict):
                orig_valid_sentences = [sentence['sentence'] for sentence in orig_valid_sentences]
            if isinstance(orig_test_sentences[0], dict):
                orig_test_sentences = [sentence['sentence'] for sentence in orig_test_sentences]

    elif params['dataset'] == 'glue/qnli':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_qnli()
        if change_params:
            params['prompt_prefix'] = "You are given two sentences. Your task is to determine if sentence 1 entails sentence 2. If sentence 1 entails sentence 2, answer with \"yes\", otherwise answer with \"no\". "
            params["q_prefix"] = ""
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['yes'], 1: ['no']}
            params['inv_label_dict'] = {'yes': 0, 'no': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'super_glue/rte':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_rte()
        # orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mnli()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = ""
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['Yes'], 1: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'No': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'imdb':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_imdb()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) great, and 2) terrible based on its content. "
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_valid_sentences[0], dict):
            orig_valid_sentences = [sentence['text'] for sentence in orig_valid_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]
        
    elif params['dataset'] == 'SetFit/sst5':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_sst5()
        # orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst5()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"terrible\" if the sentiment of the sentence is terrible, \"bad\" if the sentiment of the sentence is negative, \"okay\" if the sentiment of the sentence is neutral, \"good\" if the sentiment of the sentence is positive, or \"great\" if the sentiment of the sentence is great. "
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['bad'], 2: ['okay'], 3: ['good'], 4: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'bad': 1, 'okay': 2, 'good': 3, 'great': 4}
            params['num_labels'] = 5
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_valid_sentences[0], dict):
            orig_valid_sentences = [sentence['text'] for sentence in orig_valid_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]


    elif params['dataset'] == 'snli':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_snli()
        if change_params:
            params['prompt_prefix'] = "In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively.\n\n"
            params["q_prefix"] = " "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['Yes'], 1: ['Maybe'], 2: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'Maybe': 1, 'No': 2}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'yelp_polarity':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_yelp_polarity()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from Yelp reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
        if isinstance(orig_train_sentences[0], dict):
            orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
        if isinstance(orig_test_sentences[0], dict):
            orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]


    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_trec()
        if change_params:
            params['prompt_prefix'] = "You are given a question. You need to detect which category better describes the question. A question belongs to the description category if it asks about description and abstract concepts. Entity questions are about entities such as animals, colors, sports, etc. Abbreviation questions ask about abbreviations and expressions abbreviated. Questions regarding human beings, description of a person, and a group or organization of persons are categorized as Human. Quantity questions are asking about numeric values and Location questions ask about locations, cities, and countries. Answer with \"Abbreviation\", \"Entity\", \"Description\", \"Person\", \"Location\", and \"Quantity\""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Category: "
            params['label_dict'] = {0: ['Abbreviation'], 1: ['Entity'], 2: ['Description'], 3: ['Person'], 4: ['Location'], 5: ['Quantity']}
            params['inv_label_dict'] = {'Abbreviation': 0, 'Entity': 1, 'Description': 2, 'Person': 3, 'Location': 4, 'Quantity': 5}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'cr':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_customer_review()
        # this dataset has no validation set
        orig_valid_sentences, orig_valid_labels = None, None
        # orig_train_sentences, orig_valid_sentences, orig_train_labels, orig_valid_labels = train_test_split(orig_train_sentences, orig_train_labels, test_size=0.2, random_state=1)
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. "
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

            if params['extra_exs']:
                # add more examples into the pool, but keep the train examples
                file_dict = {'train': 'src/data_benchmark/cr/benchmark/dev.tsv'}
                train_dataset = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')

                # get the number of total in-context examples and get them equally for both labels
                n_pool_exs = params['n_pool_exs']
                n_pool_exs_per_label = n_pool_exs // 2

                # get the examples for each label
                label0_exs = train_dataset.filter(lambda example: example['label'] == 0)
                label1_exs = train_dataset.filter(lambda example: example['label'] == 1)

                # get the indices for the pool examples
                label0_indices = np.random.choice(len(label0_exs), n_pool_exs_per_label, replace=False)
                label1_indices = np.random.choice(len(label1_exs), n_pool_exs_per_label, replace=False)
                
                # now we take pool examples, and assert they do not match the training examples. if any of them match, we take another example
                pool_label0_exs = label0_exs.select(label0_indices)
                pool_label1_exs = label1_exs.select(label1_indices)

                pool_sentences = concatenate_datasets([pool_label0_exs, pool_label1_exs])
                pool_labels = pool_sentences['label']

                # save the pool examples to file
                if Path(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/cr/benchmark/pool_{n_pool_exs}_examples.tsv').is_file():
                    print(f"Pool_{n_pool_exs}_examples.tsv exists!")
                else:
                    print(f"Saving new pool of {n_pool_exs} examples to file...")
                    with open(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/cr/benchmark/pool_{n_pool_exs}_examples.tsv', 'w', newline='') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow(['sentence', 'label'])
                        for idx, (sentence, label) in enumerate(zip(pool_sentences['sentence'], pool_sentences['label'])):
                            writer.writerow([sentence, label])
                    
                    print("Done saving the pool examples")


    elif params['dataset'] == 'movie_review':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_movie_review()
        # this dataset has no validation set
        orig_valid_sentences, orig_valid_labels = None, None
        # orig_train_sentences, orig_valid_sentences, orig_train_labels, orig_valid_labels = train_test_split(orig_train_sentences, orig_train_labels, test_size=0.2, random_state=1)
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. "
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['num_labels'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'ag_news':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        # ag_news has no validation set, so we split the train set into train and valid
        orig_train_sentences, orig_valid_sentences, orig_train_labels, orig_valid_labels = train_test_split(orig_train_sentences, orig_train_labels, test_size=0.2, random_state=1)
        if change_params:
            params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology. "
            params["q_prefix"] = "Article: "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology']}
            params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3} # notice index start from 1 here
            params['task_format'] = 'classification'
            params['num_labels'] = 4
            params['num_tokens_to_predict'] = 1


        if params['extra_exs']:
            # add more examples into the pool, but keep the train examples
            file_dict = {'train': f'/home/s223540177/dai/RLforLLM/src/data_benchmark/ag_news/benchmark/in_context.tsv'}
            dataset = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')

            # get the number of total in-context examples and get them equally for both labels
            n_pool_exs = params['n_pool_exs']
            n_pool_exs_per_label = n_pool_exs // 4

            # get the examples for each label
            label0_exs = dataset.filter(lambda example: example['label'] == 0)
            label1_exs = dataset.filter(lambda example: example['label'] == 1)
            label2_exs = dataset.filter(lambda example: example['label'] == 2)
            label3_exs = dataset.filter(lambda example: example['label'] == 3)

            # get the indices for the pool examples
            label0_indices = np.random.choice(len(label0_exs), n_pool_exs_per_label, replace=False)
            label1_indices = np.random.choice(len(label1_exs), n_pool_exs_per_label, replace=False)
            label2_indices = np.random.choice(len(label2_exs), n_pool_exs_per_label, replace=False)
            label3_indices = np.random.choice(len(label3_exs), n_pool_exs_per_label, replace=False)
            
            # now we take pool examples, and assert they do not match the training examples. if any of them match, we take another example
            pool_label0_exs = label0_exs.select(label0_indices)
            pool_label1_exs = label1_exs.select(label1_indices)
            pool_label2_exs = label2_exs.select(label2_indices)
            pool_label3_exs = label3_exs.select(label3_indices)

            pool_sentences = concatenate_datasets([pool_label0_exs, pool_label1_exs, pool_label2_exs, pool_label3_exs])
            pool_labels = pool_sentences['label']

            # save the pool examples to file
            if Path(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/ag_news/benchmark/pool_{n_pool_exs}_examples.tsv').is_file():
                print(f"Pool_{n_pool_exs}_examples.tsv exists!")
            else:
                print(f"Saving new pool of {n_pool_exs} examples to file...")
                with open(f'/home/s223540177/dai/RLforLLM/src/data_benchmark/ag_news/benchmark/pool_{n_pool_exs}_examples.tsv', 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['text', 'label'])
                    for idx, (sentence, label) in enumerate(zip(pool_sentences['text'], pool_sentences['label'])):
                        writer.writerow([sentence, label])
                
                print("Done saving the pool examples")

        else:
            if isinstance(orig_train_sentences[0], dict):
                orig_train_sentences = [sentence['text'] for sentence in orig_train_sentences]
            if isinstance(orig_valid_sentences[0], dict):
                orig_valid_sentences = [sentence['text'] for sentence in orig_valid_sentences]
            if isinstance(orig_test_sentences[0], dict):
                orig_test_sentences = [sentence['text'] for sentence in orig_test_sentences]
    
    elif params["dataset"] == "yelp_review_full":
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, str2int, int2str = load_yelp_review_full()
        if change_params:
            params['prompt_prefix'] = "You are given a review about a place. You need to provide a rating from \"1 star\" to \"5 stars\" for this place."
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['1 star'], 1: ['2 stars'], 2: ['3 stars'], 3: ['4 stars'], 4: ['5 stars']}
            params['inv_label_dict'] = {'1 star': 0, '2 stars': 1, '3 stars': 2, '4 stars': 3, '5 stars': 4}
            params['task_format'] = 'classification'
            params['num_labels'] = 5
            params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'anli':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_anli_r1()
        if change_params:
            params['prompt_prefix'] = "The task is about reading the given story and question, then finding an answer to the given question. Based on the passage provided and the given question, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1

        """
        If not mentioned, we take 80 sentences as train dataset and 80 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_labels), 160, replace=False)
        fs_indices = indices[:80]
        ic_indices = indices[80:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        fs_train_labels = fs_train_sentences['label']

        ic_sentences = orig_train_sentences.select(ic_indices)
        ic_labels = ic_sentences['label']
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_valid_labels), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)
        dev_labels = dev_sentences['label']

        with open("src/data_benchmark/anli_r1/benchmark/dev.tsv", "w", newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['premise', 'hypothesis', 'label'])
                for sentence, label in zip(dev_sentences, dev_labels):
                    writer.writerow([sentence['premise'], sentence['hypothesis'], sentence['label']])

        # # Check if the files are already exist. If yes then skip the part below
        # if Path("src/data_benchmark/anli_r1/benchmark/train.arrow").is_file() and Path("data_benchmark/anli_r1/benchmark/in_context.arrow").is_file() and Path("data_benchmark/anli_r1/benchmark/test.arrow").is_file():
        #     print("Files exist!")
        # else:
        #     # save the data to data_benchmark as tsv file
        #     with open("src/data_benchmark/anli_r1/benchmark/train.tsv", "w", newline='') as f:
        #         writer = csv.writer(f, delimiter='\t')
        #         writer.writerow(['premise', 'hypothesis', 'label'])
        #         for sentence, label in zip(fs_train_sentences, fs_train_labels):
        #             writer.writerow([sentence['premise'], sentence['hypothesis'], sentence['label']])
            
        #     print("Done saving the few-shot training dataset")

        #     with open("src/data_benchmark/anli_r1/benchmark/in_context.tsv", "w", newline='') as f:
        #         writer = csv.writer(f, delimiter='\t')
        #         writer.writerow(['premise', 'hypothesis', 'label'])
        #         for sentence, label in zip(ic_sentences, ic_labels):
        #             writer.writerow([sentence['premise'], sentence['hypothesis'], sentence['label']])

        #     print("Done saving the in-context dataset")

        #     with open("src/data_benchmark/anli_r1/benchmark/test.tsv", "w", newline='') as f:
        #         writer = csv.writer(f, delimiter='\t')
        #         writer.writerow(['premise', 'hypothesis', 'label'])
        #         for sentence, label in zip(orig_test_sentences, orig_test_labels):
        #             if sentence['label'] is None:
        #                 continue
        #             else:
        #                 writer.writerow([sentence['premise'], sentence['hypothesis'], sentence['label']])

        #     # convert to json
        #     tsv2json("src/data_benchmark/anli_r1/benchmark/train.tsv", "src/data_benchmark/anli_r1/benchmark/train.json")
        #     tsv2json("src/data_benchmark/anli_r1/benchmark/in_context.tsv", "src/data_benchmark/anli_r1/benchmark/in_context.json")
        #     tsv2json("src/data_benchmark/anli_r1/benchmark/test.tsv", "src/data_benchmark/anli_r1/benchmark/test.json")

        # train_json = load_dataset("json", data_files="src/data_benchmark/anli_r1/benchmark/train.json", split='train')
        # in_context_json = load_dataset("json", data_files="src/data_benchmark/anli_r1/benchmark/in_context.json", split='train')
        # test_json = load_dataset("json", data_files="src/data_benchmark/anli_r1/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/anli_r1/benchmark/dev.json", split='train')

        # save to arrow
        # train_json.save_to_disk("src/data_benchmark/anli_r1/benchmark/train")
        # in_context_json.save_to_disk("src/data_benchmark/anli_r1/benchmark/in_context")
        # test_json.save_to_disk("src/data_benchmark/anli_r1/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/anli_r1/benchmark/dev")



    elif params['dataset'] == 'truthfulqa_mc1':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_truthfulqa_mc1()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1

        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        ic_sentences = orig_train_sentences.select(ic_indices)
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)

        with open("src/data_benchmark/truthfulqa_mc1/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "mc1_targets": convert_to_serializable(sentence['mc1_targets']),
                    "mc2_targets": convert_to_serializable(sentence['mc2_targets'])
                } for sentence in dev_sentences], f)
        print("Done saving the dev dataset")

        with open("src/data_benchmark/truthfulqa_mc1/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "mc1_targets": convert_to_serializable(sentence['mc1_targets']),
                    "mc2_targets": convert_to_serializable(sentence['mc2_targets'])
                } for sentence in fs_train_sentences], f)
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/truthfulqa_mc1/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "mc1_targets": convert_to_serializable(sentence['mc1_targets']),
                    "mc2_targets": convert_to_serializable(sentence['mc2_targets'])
                } for sentence in ic_sentences], f)
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/truthfulqa_mc1/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "mc1_targets": convert_to_serializable(sentence['mc1_targets']),
                    "mc2_targets": convert_to_serializable(sentence['mc2_targets'])
                } for sentence in orig_test_sentences], f)
        print("Done saving the test dataset")



        train_json = load_dataset("json", data_files="src/data_benchmark/truthfulqa_mc1/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/truthfulqa_mc1/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/truthfulqa_mc1/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/truthfulqa_mc1/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/truthfulqa_mc1/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/truthfulqa_mc1/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/truthfulqa_mc1/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/truthfulqa_mc1/benchmark/dev")

    elif params['dataset'] == 'piqa':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_piqa()
        if change_params:
            params['prompt_prefix'] = "The task is about reading the given story and question, then finding an answer to the given question. Based on the passage provided and the given question, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """
        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)

        ic_sentences = orig_train_sentences.select(ic_indices)
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)
        dev_labels = dev_sentences['label']

        with open("src/data_benchmark/piqa/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "goal": sentence['goal'],
                    "sol1": sentence['sol1'],
                    "sol2": sentence['sol2'],
                    "label": sentence['label']
                } for sentence in dev_sentences], f)
        print("Done saving the dev dataset")

        with open("src/data_benchmark/piqa/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "goal": sentence['goal'],
                    "sol1": sentence['sol1'],
                    "sol2": sentence['sol2'],
                    "label": sentence['label']
                } for sentence in fs_train_sentences], f)
            
        print("Done saving the few-shot training dataset")
    
        # Saving the in-context dataset
        with open("src/data_benchmark/piqa/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "goal": sentence['goal'],
                    "sol1": sentence['sol1'],
                    "sol2": sentence['sol2'],
                    "label": sentence['label']
                } for sentence in ic_sentences], f)
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/piqa/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "goal": sentence['goal'],
                    "sol1": sentence['sol1'],
                    "sol2": sentence['sol2'],
                    "label": sentence['label']
                } for sentence in orig_test_sentences], f)
            
        train_json = load_dataset("json", data_files="src/data_benchmark/piqa/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/piqa/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/piqa/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/piqa/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/piqa/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/piqa/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/piqa/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/piqa/benchmark/dev")


    elif params['dataset'] == 'drop':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_drop()
        if change_params:
            params['prompt_prefix'] = "You are given a passage and a question based on the passage. You should answer the question using the information from the passage. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1    
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """
        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)

        ic_sentences = orig_train_sentences.select(ic_indices)
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)
        dev_labels = dev_sentences['label']

        with open("src/data_benchmark/drop/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "section_id": sentence['section_id'],
                    "query_id": sentence['query_id'],
                    "passage": sentence['passage'],
                    "question": sentence['question'],
                    "answers_spans": sentence['answers_spans']
                } for sentence in dev_sentences], f)
        print("Done saving the dev dataset")

        with open("src/data_benchmark/drop/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "section_id": sentence['section_id'],
                    "query_id": sentence['query_id'],
                    "passage": sentence['passage'],
                    "question": sentence['question'],
                    "answers_spans": sentence['answers_spans']
                } for sentence in fs_train_sentences], f)
            
        print("Done saving the few-shot training dataset")
    
        # Saving the in-context dataset
        with open("src/data_benchmark/drop/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "section_id": sentence['section_id'],
                    "query_id": sentence['query_id'],
                    "passage": sentence['passage'],
                    "question": sentence['question'],
                    "answers_spans": sentence['answers_spans']
                } for sentence in ic_sentences], f)
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/drop/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "section_id": sentence['section_id'],
                    "query_id": sentence['query_id'],
                    "passage": sentence['passage'],
                    "question": sentence['question'],
                    "answers_spans": sentence['answers_spans']
                } for sentence in orig_test_sentences], f)
            
        train_json = load_dataset("json", data_files="src/data_benchmark/drop/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/drop/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/drop/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/drop/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/drop/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/drop/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/drop/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/drop/benchmark/dev")
        

    elif params['dataset'] == 'triviaqa':
        from datasets import load_dataset, concatenate_datasets
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_triviaqa()
        if change_params:
            params['prompt_prefix'] = "The task is about reading the given story and question, then finding an answer to the given question. Based on the passage provided and the given question, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
    
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """


        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 160 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)

        ic_sentences = orig_train_sentences.select(ic_indices)
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)


        with open("src/data_benchmark/triviaqa/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "question_id": sentence['question_id'],
                    "question_source": sentence['question_source'],
                    "entity_pages": sentence['entity_pages'],
                    "search_results": sentence['search_results'],
                    "answer": sentence['answer']
                } for sentence in dev_sentences], f)
        print("Done saving the dev dataset")

        with open("src/data_benchmark/triviaqa/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "question_id": sentence['question_id'],
                    "question_source": sentence['question_source'],
                    "entity_pages": sentence['entity_pages'],
                    "search_results": sentence['search_results'],
                    "answer": sentence['answer']
                } for sentence in fs_train_sentences], f)
            
        print("Done saving the few-shot training dataset")
    
        # Saving the in-context dataset
        with open("src/data_benchmark/triviaqa/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "question_id": sentence['question_id'],
                    "question_source": sentence['question_source'],
                    "entity_pages": sentence['entity_pages'],
                    "search_results": sentence['search_results'],
                    "answer": sentence['answer']
                } for sentence in ic_sentences], f)
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/triviaqa/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "question": sentence['question'],
                    "question_id": sentence['question_id'],
                    "question_source": sentence['question_source'],
                    "entity_pages": sentence['entity_pages'],
                    "search_results": sentence['search_results'],
                    "answer": sentence['answer']
                } for sentence in orig_test_sentences], f)
            
        train_json = load_dataset("json", data_files="src/data_benchmark/triviaqa/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/triviaqa/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/triviaqa/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/triviaqa/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/triviaqa/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/triviaqa/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/triviaqa/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/triviaqa/benchmark/dev")



    elif params['dataset'] == 'hellaswag':
        from datasets import load_dataset
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_hellaswag()
        if change_params:
            params['prompt_prefix'] = "The task is about reading the given story and question, then finding an answer to the given question. Based on the passage provided and the given question, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        ic_sentences = orig_train_sentences.select(ic_indices)

        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)
        dev_labels = dev_sentences['label']

        with open("src/data_benchmark/hellaswag/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "ctx_a": sentence['ctx_a'],
                    "ctx_b": sentence['ctx_b'],
                    "ctx": sentence['ctx'],
                    "ind": sentence['ind'],
                    "activity_label": sentence['activity_label'],
                    "endings": sentence['endings'],
                    "source_id": sentence['source_id'],
                    "split": sentence['split'],
                    "split_type": sentence['split_type'],
                    "label": sentence['label']
                } for sentence in dev_sentences], f)

        print("Done saving the dev dataset")

        with open("src/data_benchmark/hellaswag/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "ctx_a": sentence['ctx_a'],
                    "ctx_b": sentence['ctx_b'],
                    "ctx": sentence['ctx'],
                    "ind": sentence['ind'],
                    "activity_label": sentence['activity_label'],
                    "endings": sentence['endings'],
                    "source_id": sentence['source_id'],
                    "split": sentence['split'],
                    "split_type": sentence['split_type'],
                    "label": sentence['label']
                } for sentence in fs_train_sentences], f)
        
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/hellaswag/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "ctx_a": sentence['ctx_a'],
                    "ctx_b": sentence['ctx_b'],
                    "ctx": sentence['ctx'],
                    "ind": sentence['ind'],
                    "activity_label": sentence['activity_label'],
                    "endings": sentence['endings'],
                    "source_id": sentence['source_id'],
                    "split": sentence['split'],
                    "split_type": sentence['split_type'],
                    "label": sentence['label']
                } for sentence in ic_sentences], f)
        
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/hellaswag/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "ctx_a": sentence['ctx_a'],
                    "ctx_b": sentence['ctx_b'],
                    "ctx": sentence['ctx'],
                    "ind": sentence['ind'],
                    "activity_label": sentence['activity_label'],
                    "endings": sentence['endings'],
                    "source_id": sentence['source_id'],
                    "split": sentence['split'],
                    "split_type": sentence['split_type'],
                    "label": sentence['label']
                } for sentence in orig_test_sentences], f)
        
        train_json = load_dataset("json", data_files="src/data_benchmark/hellaswag/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/hellaswag/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/hellaswag/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/hellaswag/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/hellaswag/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/hellaswag/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/hellaswag/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/hellaswag/benchmark/dev")


    elif params['dataset'] == 'logiqa':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_logiqa()
        if change_params:
            params['prompt_prefix'] = "The task is about reading the given story and question, then finding an answer to the given question. Based on the passage provided and the given question, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. "
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1

        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        # randomly select 32 indices from the dataset as fs_train_sentences and ic_sentences
        np.random.seed(0)
        indices = np.random.choice(len(orig_train_labels), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        ic_sentences = orig_train_sentences.select(ic_indices)
        
        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_labels), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)
        dev_labels = dev_sentences['label']

        with open("src/data_benchmark/logiqa/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "context": sentence['context'],
                    "question": sentence['question'],
                    "options": sentence['options'],
                    "label": sentence['label']
                } for sentence in dev_sentences], f)
        print("Done saving the dev dataset")

        with open("src/data_benchmark/logiqa/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "context": sentence['context'],
                    "question": sentence['question'],
                    "options": sentence['options'],
                    "label": sentence['label']
                } for sentence in fs_train_sentences], f)
            
        print("Done saving the few-shot training dataset")
    
        # Saving the in-context dataset
        with open("src/data_benchmark/logiqa/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "context": sentence['context'],
                    "question": sentence['question'],
                    "options": sentence['options'],
                    "label": sentence['label']
                } for sentence in ic_sentences], f)
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/logiqa/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "context": sentence['context'],
                    "question": sentence['question'],
                    "options": sentence['options'],
                    "label": sentence['label']
                } for sentence in orig_test_sentences], f)
            
        train_json = load_dataset("json", data_files="src/data_benchmark/logiqa/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/logiqa/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/logiqa/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/logiqa/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/logiqa/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/logiqa/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/logiqa/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/logiqa/benchmark/dev")


    elif params['dataset'] == 'coqa':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_coqa()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        ic_sentences = orig_train_sentences.select(ic_indices)

        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)


        with open("src/data_benchmark/coqa/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "source": sentence['source'],
                    'story': sentence['story'],
                    'questions': sentence['questions'],
                    'answers': sentence['answers'],
                    'additional_answers': sentence['additional_answers']
                } for sentence in dev_sentences], f)

        print("Done saving the dev dataset")

        with open("src/data_benchmark/coqa/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "source": sentence['source'],
                    'story': sentence['story'],
                    'questions': sentence['questions'],
                    'answers': sentence['answers'],
                    'additional_answers': sentence['additional_answers']
                } for sentence in fs_train_sentences], f)
        
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/coqa/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "source": sentence['source'],
                    'story': sentence['story'],
                    'questions': sentence['questions'],
                    'answers': sentence['answers'],
                    'additional_answers': sentence['additional_answers']
                } for sentence in ic_sentences], f)
        
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/coqa/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "source": sentence['source'],
                    'story': sentence['story'],
                    'questions': sentence['questions'],
                    'answers': sentence['answers'],
                    'additional_answers': sentence['additional_answers']
                } for sentence in orig_test_sentences], f)
        
        train_json = load_dataset("json", data_files="src/data_benchmark/coqa/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/coqa/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/coqa/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/coqa/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/coqa/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/coqa/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/coqa/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/coqa/benchmark/dev")


    elif params['dataset'] == 'race':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_race()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, fs_train_labels = [], []
        ic_sentences, ic_labels = [], []

        np.random.seed(0)
        indices = np.random.choice(len(orig_train_sentences), 32, replace=False)
        fs_indices = indices[:16]
        ic_indices = indices[16:]

        # Get the dataset as dict
        fs_train_sentences = orig_train_sentences.select(fs_indices)
        ic_sentences = orig_train_sentences.select(ic_indices)

        # randomly select a subset of 80 examples as dev_set
        dev_indices = np.random.choice(len(orig_test_sentences), 80, replace=False)
        dev_sentences = orig_test_sentences.select(dev_indices)


        with open("src/data_benchmark/race/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "article": sentence['article'],
                    "problems": sentence['problems'],
                } for sentence in dev_sentences], f)

        print("Done saving the dev dataset")

        with open("src/data_benchmark/race/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "article": sentence['article'],
                    "problems": sentence['problems'],
                } for sentence in fs_train_sentences], f)
        
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/race/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "article": sentence['article'],
                    "problems": sentence['problems'],
                } for sentence in ic_sentences], f)
        
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/race/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "article": sentence['article'],
                    "problems": sentence['problems'],
                } for sentence in orig_test_sentences], f)
        
        train_json = load_dataset("json", data_files="src/data_benchmark/race/benchmark/train.json", split='train')
        in_context_json = load_dataset("json", data_files="src/data_benchmark/race/benchmark/in_context.json", split='train')
        test_json = load_dataset("json", data_files="src/data_benchmark/race/benchmark/test.json", split='train')
        dev_json = load_dataset("json", data_files="src/data_benchmark/race/benchmark/dev.json", split='train')

        # save to arrow
        train_json.save_to_disk("src/data_benchmark/race/benchmark/train")
        in_context_json.save_to_disk("src/data_benchmark/race/benchmark/in_context")
        test_json.save_to_disk("src/data_benchmark/race/benchmark/test")
        dev_json.save_to_disk("src/data_benchmark/race/benchmark/dev")



    elif params['dataset'] == 'winogrande':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_winogrande()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, ic_sentences, dev_sentences, test_sentences = process_data_llm(orig_train_sentences, orig_test_sentences, 16, 16, 80)
        


        with open("src/data_benchmark/winogrande/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    "sentence": sentence['sentence'],
                    "option1": sentence['option1'],
                    "option2": sentence['option2'],
                    "answer": sentence['answer']
                } for sentence in dev_sentences], f)

        print("Done saving the dev dataset")

        with open("src/data_benchmark/winogrande/benchmark/train.json", "w") as f:
            json.dump([
                {
                    "sentence": sentence['sentence'],
                    "option1": sentence['option1'],
                    "option2": sentence['option2'],
                    "answer": sentence['answer']
                } for sentence in fs_train_sentences], f)
        
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/winogrande/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    "sentence": sentence['sentence'],
                    "option1": sentence['option1'],
                    "option2": sentence['option2'],
                    "answer": sentence['answer']
                } for sentence in ic_sentences], f)
        
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/winogrande/benchmark/test.json", "w") as f:
            json.dump([
                {
                    "sentence": sentence['sentence'],
                    "option1": sentence['option1'],
                    "option2": sentence['option2'],
                    "answer": sentence['answer']
                } for sentence in test_sentences], f)
        
        save_data_llm(params['dataset'])
        

    elif params['dataset'] == 'siqa':
        orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels = load_siqa()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = "Question: "
            params["a_prefix"] = "Answer: "
            params['task_format'] = 'qa'
            params['num_tokens_to_predict'] = 1
        
        """
        If not mentioned, we take 16 sentences as train dataset and 16 in-context sentences
        """

        fs_train_sentences, ic_sentences, dev_sentences, test_sentences = process_data_llm(orig_train_sentences, orig_test_sentences, 16, 16, 80)
        


        with open("src/data_benchmark/siqa/benchmark/dev.json", "w") as f:
            json.dump([
                {
                    'context': sentence['context'],
                    'question': sentence['question'],
                    'answerA': sentence['answerA'],
                    'answerB': sentence['answerB'],
                    'answerC': sentence['answerC'],
                    'label': sentence['label']
                } for sentence in dev_sentences], f)

        print("Done saving the dev dataset")

        with open("src/data_benchmark/siqa/benchmark/train.json", "w") as f:
            json.dump([
                {
                    'context': sentence['context'],
                    'question': sentence['question'],
                    'answerA': sentence['answerA'],
                    'answerB': sentence['answerB'],
                    'answerC': sentence['answerC'],
                    'label': sentence['label']
                } for sentence in fs_train_sentences], f)
        
        print("Done saving the few-shot training dataset")

        # Saving the in-context dataset
        with open("src/data_benchmark/siqa/benchmark/in_context.json", "w") as f:
            json.dump([
                {
                    'context': sentence['context'],
                    'question': sentence['question'],
                    'answerA': sentence['answerA'],
                    'answerB': sentence['answerB'],
                    'answerC': sentence['answerC'],
                    'label': sentence['label']
                } for sentence in ic_sentences], f)
        
        print("Done saving the in-context dataset")

        # Saving the test dataset
        with open("src/data_benchmark/siqa/benchmark/test.json", "w") as f:
            json.dump([
                {
                    'context': sentence['context'],
                    'question': sentence['question'],
                    'answerA': sentence['answerA'],
                    'answerB': sentence['answerB'],
                    'answerC': sentence['answerC'],
                    'label': sentence['label']
                } for sentence in test_sentences], f)

        save_data_llm(params['dataset'])

    else:
        raise NotImplementedError


    return orig_train_sentences, orig_train_labels, orig_valid_sentences, orig_valid_labels, orig_test_sentences, orig_test_labels

    # return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels
