from datasets import concatenate_datasets 
import datasets
import pandas as pd
import random
from kutils.base import *

def _standardize_answers(answers):
    answers = dict(answers)
    return {
        'text': list(answers['text']), 
        'answer_start': list(int(ans) for ans in answers['answer_start']), 
    }

def standardize_dataset(dataset): 
    'Hacky way to convert all datasets to similar format'
    FEATURES = ['id', 'context', 'question', 'answers']
    df = pd.DataFrame.from_dict(dataset.to_dict())
    df = df[FEATURES]
    df.answers = df.answers.apply(_standardize_answers)
    dataset = datasets.Dataset.from_pandas(df)
    return dataset

def random_example(dataset): 
    i = random.randint(0, len(dataset))
    ex = dataset[i]
    q, c, a = ex['question'], ex['context'], ex['answers']['text']
    print(f'question ({len(q)} chars): {q}')
    print(f'answer ({len(a)} chars): {a}')
    print(f'context ({len(c)} chars): {c}')
    print()

def get_ans_len(example): 
    answers = example['answers']
    if isinstance(answers, str): return len(answers)
    if len(answers['text']) == 0: return 0
    return len(answers['text'][0])

def avg_len(dataset, col_name): 
    return sum([len(ex) for ex in dataset[col_name]]) / len(dataset)    

def print_dataset_info(dataset, name): 
    print()
    print(f'----------------------- {red(name)} Dataset -----------------------')
    print(f'Total {name} examples: ', green(len(dataset)))
    print(f'Average answer length: ', blue(sum(get_ans_len(ex) for ex in dataset)/len(dataset)))
    print(f'Average context length: ', blue(avg_len(dataset, 'context')))
    print(f'Average question length: ', blue(avg_len(dataset, 'question')))
    print('--------------------------------------------------------------------')
    random_example(dataset)

def filter_long_answers(example): 
    ans_len = get_ans_len(example)
    if ans_len < 16: return True
    if ans_len < 32: random.uniform(0, 1) < 0.9
    if ans_len < 64: return random.uniform(0, 1) < 0.75
    if ans_len < 128: return random.uniform(0, 1) < 0.5
    if ans_len < 256: return random.uniform(0, 1) < 0.25
    return False

def load_splits(name, config=None, verbose=True): 
    if config is not None: dataset = datasets.load_dataset(name, config)
    else: dataset = datasets.load_dataset(name)
    ds_list = []
    for split in ['train', 'validation', 'test']: 
        if split in dataset.keys():
            split_dataset = dataset[split]
            # split_dataset = split_dataset.filter(filter_long_answers)
            split_dataset = standardize_dataset(split_dataset)
            ds_list.append(split_dataset)
    dataset = concatenate_datasets(ds_list).shuffle()
    if verbose: print_dataset_info(dataset, name)
    return dataset