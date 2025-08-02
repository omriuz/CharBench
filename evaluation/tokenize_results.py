import tiktoken
import os
import pandas as pd
from transformers import AutoTokenizer
import argparse
from const import MODEL_TO_TOKENIZER_LIBRARY


def get_tokenization(word, model_name, model_to_tokenizer=None):
    tokenizer, tokenizer_library = model_to_tokenizer[model_name]
    if tokenizer_library == "huggingface":
        # Get tokens without special tokens
        inputs = tokenizer(str(word), return_tensors="pt", add_special_tokens=False)
        token_ids = inputs["input_ids"][0]  # tensor of IDs
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        if model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            if tokens[0][0] == "▁" and len(tokens[0]) > 1:
                tokens[0] = tokens[0][1:]
        return tokens
    else:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(str(word))
        return [encoding.decode([token]) for token in tokens]


def main(args):
    df = pd.read_csv(args.input)

    model_to_tokenizer = {}
    for model in df['model'].unique():
        if MODEL_TO_TOKENIZER_LIBRARY[model] == "huggingface":
            #cut turbo from model name
            model_name = model.split("-Turbo")[0]
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            model_to_tokenizer[model] = (tokenizer, "huggingface")
        else:
            model_to_tokenizer[model] = (None, "openai")

    
    result_df = df.copy()

    print("Calculating tokenization metrics...")

    result_df['tokenization'] = result_df.apply(
        lambda row: get_tokenization(row['word'], row['model'], model_to_tokenizer), axis=1)
    
    result_df.to_csv(args.input, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
        default="english_evaluation_results.csv",
        help="Path to the results file"
    )
    args = parser.parse_args()
    main(args)