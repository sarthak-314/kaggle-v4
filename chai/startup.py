import sys

from sklearn.manifold import trustworthiness 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kutils.startup import * 
from chai.working_hf_data import *

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')

def get_word_len_tokens(word_lens): 
    return [f'[WORD={word_len}]' for word_len in word_lens]

WORD_LENS =[0, 10, 20, 50, 100, 200, 400, 1000, 2000, 4000, 10000, 25000]
def add_word_len_tokens(df, word_lens=WORD_LENS, split_on='\n'): 
    df_dict = {'context_with_token': [], 'id': [], 'answer_start_temp': []}
    for i, row in df.iterrows(): 
        lines = []
        word_count = 0
        answer_found = False
        answer_start = row.answer_start
        for line in row.context.split(split_on):
            for lower, upper in zip(word_lens, word_lens[1:]): 
                if word_count < upper: 
                    break
            token = f'[WORD={lower}]'
            add_token = len(line) > 8
            if add_token: 
                word_count += len(line) + 1
                line = token + line
                lines.append(line)
                if not answer_found: 
                    answer_start += 1
            else: 
                word_count += len(line)
                lines.append(line)
            
            if row.answer_text in line: 
                answer_found = True
        context = split_on.join(lines)
        df_dict['context_with_token'].append(context)
        df_dict['id'].append(row.id)
        df_dict['answer_start_temp'].append(answer_start)
    
    df = df.merge(pd.DataFrame(df_dict))
    df['org_context'] = df['context']
    df['context'] = df['context_with_token']
    df['org_answer_start'] = df['answer_start']
    df['answer_start'] = df['answer_start_temp']
    
    return df