import argparse
from openai import OpenAI, AsyncOpenAI
import os
import pandas as pd
from typing import List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
from utils import generate_prompt, extract_response
# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_batch(batch_items, model, test_df, client):
    tasks = []
    for idx in batch_items:
        prompt = generate_prompt(test_df, idx)
        tasks.append(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        )
    return await asyncio.gather(*tasks, return_exceptions=True)

async def run_evaluation_async(df: pd.DataFrame, async_client, model: str = "gpt-4", batch_size: int = 5) -> pd.DataFrame:
    results = []
    start_idx = 0
    # Process in batches
    for batch_start in tqdm(range(start_idx, len(df), batch_size), desc=f"Evaluating with {model}"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_indices = list(range(batch_start, batch_end))
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                batch_responses = await process_batch(batch_indices, model, df, async_client)
                for idx, response in zip(batch_indices, batch_responses):
                    if isinstance(response, Exception):
                        print(f"\nError on question {idx}: {str(response)}")
                        continue
                        
                    model_answer = response.choices[0].message.content.strip()
                    correct_answer = df.iloc[idx]['answer']
                    
                    predicted_answer = extract_response(model_answer)
                    
                    result = {
                        'question_idx': df.iloc[idx]['question_idx'],
                        'question': df.iloc[idx]['question'],  
                        'model_prediction': predicted_answer,
                        'correct_answer': correct_answer,
                        'is_valid': predicted_answer is not None,
                        'is_correct': predicted_answer == correct_answer,
                        'word': df.iloc[idx]['word'],
                        'word_length': len(df.iloc[idx]['word']),
                        'task': df.iloc[idx]['task'],
                        'character': df.iloc[idx]['character'],
                    }
                    
                    results.append(result)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                print(f"\nError on batch {batch_start}-{batch_end}, attempt {retry_count}/{max_retries}: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed to process batch {batch_start}-{batch_end} after {max_retries} attempts")
    
    return pd.DataFrame(results)

async def evaluate_all_models(df: pd.DataFrame, models: List[str], batch_size: int = 5):
    all_results = []
    
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as async_client:
        for model in models:
            print(f"\nEvaluating model: {model}")
            results_df = await run_evaluation_async(df, async_client, model, batch_size)
            results_df['model'] = model
            all_results.append(results_df)
    
    # Combine all results
    return pd.concat(all_results, ignore_index=True)

def run_evaluation(df: pd.DataFrame, models: List[str], batch_size: int = 5) -> pd.DataFrame:
    return asyncio.run(evaluate_all_models(df, models, batch_size))

def main(args):
    # Load the  data and filter by task
    df = pd.read_csv(args.input_file)
    if args.task != "all":
        df = df[df['task'] == args.task]
    
    # Run all models in a single async context
    final_results = run_evaluation(df, args.models, args.batch_size)
    
    # Save results
    final_results.to_csv(args.output_file, index=False)  # Use the modified output_file name
    
    # Print summary of all models
    print("\nSummary of all models:")
    summary = final_results.groupby('model')['is_correct'].agg(['mean', 'count'])
    summary['mean'] = summary['mean'].apply(lambda x: f"{x:.2%}")
    print(summary)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--models",default=["gpt-4o","gpt-4o-mini","gpt-3.5-turbo"],type=List,
                        help="List of OpenAI models to use")
    parser.add_argument("-i","--input_file", type=str, default="benchmark.csv", help="Input CSV file")
    parser.add_argument("-o","--output_file", type=str, default="evaluation_results.csv", help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of requests to batch together")
    parser.add_argument("--task", type=str, default="all", help="Task to evaluate")
    args = parser.parse_args()
    main(args)
