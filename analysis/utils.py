
import pandas as pd
from const import *
import ast    

def load_and_filter_data(filepath):
    df = pd.read_csv(filepath)
    return df

def add_inferred_columns(df):
    def get_target_token_length(tokenization_str, char_idx,task):
        if not (task in ['find_first_occurrence','find_last_occurrence']):
            return None
        char_idx = int(char_idx)
        tokens = ast.literal_eval(tokenization_str)
        current_pos = 0
        for token in tokens:
            if char_idx >= current_pos and char_idx < current_pos + len(token):
                return len(token)
            current_pos += len(token)
    

    def get_number_of_tokens(tokenization_str):
        return len(ast.literal_eval(tokenization_str))
    
    df = df.copy()
    df['word_length'] = df.apply(
        lambda row: len(row['word']), 
        axis=1
    )
    df['target_token_length'] = df.apply(
        lambda row: get_target_token_length(row['tokenization'], row['correct_answer'],row['task']), 
        axis=1
    )
    
    df['number_of_tokens'] = df.apply(
        lambda row: get_number_of_tokens(row['tokenization']), 
        axis=1
    )

    df['gold_truth'] = df.apply(
        lambda row: row['correct_answer'], 
        axis=1
    )

    df['gold_truth_devided_by_word_length'] = df.apply(
        lambda row: row['correct_answer']/row['word_length'], 
        axis=1
    )

    df['unique_characters'] = df.apply(
        lambda row: len(set(row['word'])), 
        axis=1
    )

    df['compression_rate'] = df.apply(
        lambda row: row['word_length']/row['number_of_tokens'], 
        axis=1
    )
    return df

def get_short_model_name(model_name):
    return MODEL_NAMES_TO_SHORT[model_name]

    