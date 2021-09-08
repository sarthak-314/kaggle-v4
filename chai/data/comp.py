import pandas as pd
from chai.data.utils import WORD_LENS, add_word_len_tokens

def fix_start(df): 
    def func(row): 
        return row.context.find(row.answer_text) 
    df['answer_start'] = df.apply(func, axis=1)
    return df

def clean_kaggle_noise(df):
    df = df.set_index('id')
    df.loc['1a2160a69', 'answer_text'] = 'किर्काल्दी'
    df.loc['632c16ba0', 'answer_text'] = 'चार जोड़े'
    df = df[df.index!='2b41f3744']
    df = df[df.index!='bc9f0d533'] 
    df = df.reset_index()
    df = fix_start(df)
    return df

def remove_multi_answer_sentences(df): 
    cleaned_dict = {'context_pruned': [], 'id': []}
    for i, row in df.iterrows(): 
        lines = []
        seen_ans_once = False
        for line in row.context.split('\n'): 
            if row.answer_text in line:
                if seen_ans_once: continue 
                seen_ans_once = True 
            lines.append(line)
        context = '\n'.join(lines)
        cleaned_dict['context_pruned'].append(context)
        cleaned_dict['id'].append(row.id)
    x = pd.DataFrame(cleaned_dict)
    df = x.merge(df)
    df['context'] = df['context_pruned']
    df = fix_start(df)
    return df

def add_word_len_tokens_to_df(df, word_lens=WORD_LENS, hindi_split='।', tamil_split='.'): 
    hindi = add_word_len_tokens(df[df.language=='hindi'], word_lens, hindi_split) # ।
    tamil = add_word_len_tokens(df[df.language=='hindi'], word_lens, tamil_split) # .
    df = pd.concat([hindi, tamil]).sample(frac=1.)
    df = fix_start(df)
    return df

def build_goldp(df): 
    # Take the first sentence where the answer appears
    gold = {'goldp': [], 'id': []}
    for i, row in df.iterrows(): 
        for line in row.context.split('\n'): 
            if row.answer_text in line: 
                if row.id in gold['id']:
                    continue
                gold['gold_s'].append(line)
                gold['id'].append(row.id)

    gold = pd.DataFrame(gold)
    gold = df.merge(gold)
    gold['context'] = gold['goldp']
    gold = fix_start(gold)
    return gold

def build_uniform_negative(df, min_paragraph_len=128):
    negative = {'negative_p': [], 'id': []}
    for i, row in df.iterrows(): 
        for line in row.context.split('\n'): 
            if row.answer_text not in line: 
                if len(line) < min_paragraph_len: continue
                negative['negative_p'].append(line)
                negative['id'].append(row.id)

    negative = pd.DataFrame(negative)
    negative = df.merge(negative)
    negative['context'] = negative['negative_p']
    negative['answer_text'] = ''
    negative['answer_start'] = 0

    del negative['context_with_token']
    del negative['org_context']
    del negative['negative_p']
    return negative