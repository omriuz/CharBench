import json
import random
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import argparse
import numpy as np
import re


MIN_WORD_LENGTH = 4
MAX_WORD_LENGTH = 10
NUM_TASKS = 4


def main(args):
    set_seed(int(args.seed))

    if not os.path.exists(f"unique_words_{args.language}.json"):
        load_corpus_to_file(args.corpus, args.num_unique_words, f"unique_words_{args.language}.json", args.language)

    create_benchmark(f"unique_words_{args.language}.json",args.benchmark_file)


def create_benchmark(file_name, output_file):

    with open(file_name, "r") as f:
        unique_words = json.load(f)

    print(f"Working with {len(unique_words)} unique words")

    samples_per_length_bucket = 25000
    samples_per_length_bucket_per_task = samples_per_length_bucket // NUM_TASKS
    
    word_length_buckets = {}
    for word in unique_words:
        if str(len(word)) not in word_length_buckets:
            word_length_buckets[str(len(word))] = []
        word_length_buckets[str(len(word))].append(word)
    
    # sort buckets by length
    word_length_buckets = dict(sorted(word_length_buckets.items(), key=lambda item: int(item[0])))
    
    for bucket in word_length_buckets:
        assert len(word_length_buckets[bucket]) >= samples_per_length_bucket
        word_length_buckets[bucket] = word_length_buckets[bucket][:samples_per_length_bucket]
        random.shuffle(word_length_buckets[bucket])
    
    benchmark = pd.DataFrame(columns=["question","answer", "word", "character","task"])

    # Get templates for the tasks
    templates = get_character_reasoning_templates()
    
    # Counting
    benchmark = count_chars_in_words_task(benchmark, word_length_buckets, templates,"count_unique_chars", True,samples_per_length_bucket_per_task)
    benchmark = count_character_frequency_task(benchmark, word_length_buckets, templates,"count_character_frequency",samples_per_length_bucket_per_task)

    # Indexing
    benchmark = find_index_of_occurrence_task(benchmark, word_length_buckets, templates["find_first_occurrence"],True,"find_first_occurrence",samples_per_length_bucket_per_task)
    benchmark = find_index_of_occurrence_task(benchmark, word_length_buckets, templates["find_last_occurrence"],False,"find_last_occurrence",samples_per_length_bucket_per_task)
    
    # print information about the benchmark
    print(f"Benchmark contains {len(benchmark)} questions")
    print(f"Benchmark contains {len(benchmark['word'].unique())} unique words")
    print(f"Benchmark contains {len(benchmark['character'].unique())} unique characters")
    print(f"Benchmark contains {len(benchmark['task'].unique())} unique tasks")
    # Save to files
    benchmark.reset_index(drop=True, inplace=True)
    benchmark.insert(0, "question_idx", range(len(benchmark)))
    benchmark.to_csv(output_file, index=False)

    os.remove(file_name)


def count_chars_in_words_task(benchmark: pd.DataFrame, word_buckets: dict, templates: dict,task: str, should_count_unique_characters: bool,samples_per_length_bucket_per_task):
    for _, words in tqdm(word_buckets.items(), desc="Creating character count questions"):
        for word in words[:samples_per_length_bucket_per_task]:
            answer = len(word) if not should_count_unique_characters else len(set(word))
            
            question = templates[task].format(word=word)
            
            # Add to benchmark DataFrame
            benchmark = pd.concat([
                benchmark, 
                pd.DataFrame([{
                    "question": question,
                    "answer": str(answer),
                    "word": word,
                    "character": None,
                    "task": task
                }])
            ], ignore_index=False)
    # remove used words from word_buckets
    for bucket in word_buckets:
        word_buckets[bucket] = word_buckets[bucket][samples_per_length_bucket_per_task:]
    return benchmark

def count_character_frequency_task(benchmark: pd.DataFrame, word_buckets: dict, templates: dict, task: str,samples_per_length_bucket_per_task):
    for _, words in tqdm(word_buckets.items(), desc="Creating character frequency questions"):
        for word in words[:samples_per_length_bucket_per_task]:
            character = random.choice(word)
            answer = word.count(character)
            question = templates[task].format(word=word, character=character)
            benchmark = pd.concat([
                benchmark, 
                pd.DataFrame([{
                    "question": question,
                    "answer": str(answer),
                    "word": word,
                    "character": character,
                    "task": task
                }]) 
            ], ignore_index=True)
    # remove used words from word_buckets
    for bucket in word_buckets:
        word_buckets[bucket] = word_buckets[bucket][samples_per_length_bucket_per_task:]
    return benchmark


def find_index_of_occurrence_task(benchmark: pd.DataFrame, word_buckets: dict, template: str, is_first_occurrence: bool, task_name: str,samples_per_length_bucket_per_task):
    for _, words in tqdm(word_buckets.items(), desc="Creating occurrence questions"):
        for word in words[:samples_per_length_bucket_per_task]:
            character = random.choice(word)
            if is_first_occurrence:
                answer = word.index(character)
            else:
                answer = word.rindex(character)
            question = template.format(word=word, character=character)

            benchmark = pd.concat([
                benchmark, 
                pd.DataFrame([{
                    "question": question,
                    "answer": str(answer),
                    "word": word,
                    "character": character,
                    "task": task_name
                }])
            ], ignore_index=True)
    # remove used words from word_buckets
    for bucket in word_buckets:
        word_buckets[bucket] = word_buckets[bucket][samples_per_length_bucket_per_task:]
    return benchmark 

def load_corpus_to_file(dataset_name: str, num_samples: int, file_name: str, language: str):
    dataset = load_dataset(dataset_name)
    corpus = " ".join(dataset['train'][:num_samples]['text'])
    regex = get_regex(language)
    unique_words = list(set(re.findall(regex, corpus)))
    # filter words that are too long/short
    unique_words = [word for word in unique_words if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH]
    with open(file_name, "w") as f:
        json.dump(unique_words, f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_regex(language: str):
    if language == "english":
        return r'\b[a-zA-Z]+\b'
    return None

def get_character_reasoning_templates() -> dict:
    """Returns templates for string manipulation questions."""
    return {
        "count_character_frequency": "How many times does the character '{character}' appear in the string '{word}'?",
        
        "count_unique_chars": "How many unique characters appear in the string '{word}'?",
                        
        "find_first_occurrence": "What is the index of the first occurrence of the character '{character}' in the string '{word}'?\nStart counting from 0.",
        
        "find_last_occurrence": "What is the index of the last occurrence of the character '{character}' in the string '{word}'?\nStart counting from 0.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Parser")
    parser.add_argument("-c", "--corpus",default="JeanKaddour/minipile")
    parser.add_argument("-l","--language",default="english")
    parser.add_argument("-b", "--benchmark_file",default=f"benchmark_english.csv")
    parser.add_argument("-s","--seed",default=42)
    parser.add_argument("-n","--num_unique_words",default=100000,help="The number of unique words to extract from the input dataset")
    args = parser.parse_args()
    main(args)

