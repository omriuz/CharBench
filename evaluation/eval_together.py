import argparse
import pandas as pd
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
from together import Together
from utils import generate_prompt, extract_response
import os

# Load environment variables
load_dotenv()

client = Together()


async def process_question_async(idx, model, df):
    try:
        prompt = generate_prompt(df, idx)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return idx, response
    except Exception as e:
        print(f"\nError processing question {idx}: {str(e)}")
        return idx, e

async def process_batch(batch_indices, model, df, max_retries=3):
    tasks = []
    results = {}
    
    # Create tasks for all questions in batch
    for idx in batch_indices:
        task = asyncio.create_task(process_question_async(idx, model, df))
        tasks.append(task)
    
    # Use as_completed to process results as they come in
    for completed_task in asyncio.as_completed(tasks):
        idx, result = await completed_task
        
        if isinstance(result, Exception):
            # Handle retries for failed requests
            retry_count = 0
            current_result = result
            
            while retry_count < max_retries:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"\nRetrying question {idx}, attempt {retry_count}/{max_retries} after {wait_time}s: {str(current_result)}")
                await asyncio.sleep(wait_time)
                
                try:
                    _, new_result = await process_question_async(idx, model, df)
                    if not isinstance(new_result, Exception):
                        results[idx] = new_result
                        break
                    current_result = new_result
                except Exception as e:
                    current_result = e
            
            if idx not in results:
                print(f"Failed to process question {idx} after {max_retries} retries")
        else:
            # Success
            results[idx] = result
    
    return results

async def run_evaluation_async(df: pd.DataFrame, model: str, batch_size: int = 10) -> pd.DataFrame:
    results = []
    num_questions = len(df)
    pbar = tqdm(total=num_questions, desc=f"Evaluating {model}")
    
    # Process in batches
    for i in range(0, num_questions, batch_size):
        batch_indices = list(range(i, min(i + batch_size, num_questions)))
        batch_results = await process_batch(batch_indices, model, df)
        
        # Process successful results
        for idx, response in batch_results.items():
            if isinstance(response, Exception):
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
        
        batch_size_completed = len(batch_indices)  # Use actual batch size
        pbar.update(batch_size_completed)
    
    pbar.close()
    return pd.DataFrame(results)

async def evaluate_all_models_async(df: pd.DataFrame, models: List[str], batch_size: int = 10) -> pd.DataFrame:
    all_results = []
    for model in models:
        print(f"\nEvaluating model: {model}")
        results_df = await run_evaluation_async(df, model, batch_size)
        results_df['model'] = model  # Add model name to results
        all_results.append(results_df)
    
    # Combine all results
    return pd.concat(all_results, ignore_index=True)

async def async_main(df, args):
    # Run all models with async batch processing
    batch_size = args.batch_size
    final_results = await evaluate_all_models_async(df, args.models, batch_size)
        
    # Save results
    if os.path.exists(args.output_file):
        existing = pd.read_csv(args.output_file)
        final_results = pd.concat([existing, final_results], ignore_index=True)

    final_results.to_csv(args.output_file, index=False)
    
    # Print summary of all models
    print("\nSummary of all models:")
    summary = final_results.groupby('model')['is_correct'].agg(['mean', 'count'])
    summary['mean'] = summary['mean'].apply(lambda x: f"{x:.2%}")
    print(summary)

def main(args):
    df = pd.read_csv(args.input_file)
    if args.task != "all":
        df = df[df['task'] == args.task]

    asyncio.run(async_main(df, args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--models",default=["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo","deepseek-ai/DeepSeek-V3","mistralai/Mistral-7B-Instruct-v0.2",], 
                        help="List of Together.ai models to use")
    parser.add_argument("-i","--input_file", type=str, default="benchmark_english.csv", help="Input CSV file")
    parser.add_argument("-o","--output_file", type=str, default="english_evaluation_results.csv", help="Output CSV file")
    parser.add_argument("--task", type=str, default="all", help="Task to evaluate")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of concurrent API calls")
    args = parser.parse_args()
    main(args)
