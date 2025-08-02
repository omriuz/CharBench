import pandas as pd
import argparse
import os
from utils import load_and_filter_data, add_inferred_columns, get_short_model_name
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import numpy as np
from const import *
import seaborn as sns


def analyze_word_length_vs_target_token(df,task,models,output_folder):
    output_folder = Path(output_folder) / task
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    models = args.models

    df = df[df['model'].isin(models)]

    metric_cols = ['target_token_length','word_length']

    model_errors = {m: df[(df['model'] == m) & (~df['is_correct'])] for m in models}

    for model in models:
        plt.figure()
        for i,col in enumerate(metric_cols):
            model_error = model_errors[model]
            grouped = model_error.groupby(col)['question_idx'].apply(set)
            lengths = sorted([l for l in grouped.keys()])

            mdf = df[df['model'] == model]
            dist = mdf.groupby(col)['question_idx'].apply(set).to_dict()

            avg_acc = [(len(dist[l]) - len(grouped[l])) / len(dist[l]) for l in lengths]

            y = mdf['is_correct'].astype(int).tolist()
            x = mdf[col].tolist()
            corr = pearsonr(x, y)[0]
            std = [((p * (1 - p)) / len(dist[l])) ** 0.5 for p, l in zip(avg_acc, lengths)]

            plt.errorbar(lengths, avg_acc, yerr=std, marker=MARKERS[i],linestyle=LINESTYLES[i], capsize=3,
                     label=f'{col} (r={corr:.2f})')

        plt.ylabel("Average Accuracy",fontsize='x-large')
        plt.xlabel("Length in Characters",fontsize='x-large')
        plt.legend( loc="best", fontsize='x-large')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_folder / f"{task}-{get_short_model_name(model)}.png",
                dpi=300, bbox_inches="tight")


def analyze_tokens_spread(df,task,models,output_folder):
    output_folder = Path(output_folder) / task
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    models = args.models
    df = df[df['model'].isin(models)]

    metric_cols = ['gold_truth','word_length','number_of_tokens']

    model_errors = {m: df[(df['model'] == m) & (~df['is_correct'])] for m in models}

    for model in models:
        plt.figure()
        for i,col in enumerate(metric_cols):
            model_error = model_errors[model]
            grouped = model_error.groupby(col)['question_idx'].apply(set)
            lengths = sorted([l for l in grouped.keys()])


            mdf = df[df['model'] == model]
            dist = mdf.groupby(col)['question_idx'].apply(set).to_dict()
            avg_acc = [(len(dist[l]) - len(grouped[l])) / len(dist[l]) for l in lengths]

            y = mdf['is_correct'].astype(int).tolist()
            x = mdf[col].tolist()

            corr = pearsonr(x, y)[0]
            std = [((p * (1 - p)) / len(dist[l])) ** 0.5 for p, l in zip(avg_acc, lengths)]

            plt.errorbar(lengths, avg_acc, yerr=std, marker=MARKERS[i],linestyle=LINESTYLES[i], capsize=3,
                     label=f'{col} (r={corr:.2f})')

        plt.ylabel("Average Accuracy",fontsize='x-large')
        plt.legend( loc="best", fontsize='x-large')        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_folder / f"{task}-{get_short_model_name(model)}.png",
                dpi=300, bbox_inches="tight")


def analyze_number_of_tokens(df, models, output_folder):

    df = df[df['model'].isin(models)]

    plt.figure()

    for i,model in enumerate(models):
        model_df = df[df['model'] == model]
        error_df = model_df[~model_df['is_correct']]
        col = 'number_of_tokens'

        grouped = error_df.groupby(col)['question_idx'].apply(set)
        dist = model_df.groupby(col)['question_idx'].apply(set).to_dict()

        lengths = sorted([l for l in grouped.keys()])
        avg_acc = [(len(dist[l]) - len(grouped[l])) / len(dist[l]) for l in lengths]
        std = [((p * (1 - p)) / len(dist[l])) ** 0.5 for p, l in zip(avg_acc, lengths)]

        plt.errorbar(lengths, avg_acc, yerr=std, marker=MARKERS[i % len(MARKERS)],
                        linestyle=LINESTYLES[i % len(LINESTYLES)], capsize=3,
                        label=f'{get_short_model_name(model)}')

    plt.ylabel("Average Accuracy",fontsize='x-large')
    plt.xlabel("Number of Tokens",fontsize='x-large')
    plt.xticks(lengths)
    plt.title('count_unique_chars',fontsize='x-large')
    plt.legend( loc="best", fontsize='large')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_folder / f"number_of_tokens.png", dpi=300, bbox_inches="tight")


def analyze_word_length(df, models, output_folder):
    df = df[df['model'].isin(models)]

    for model in models:
        plt.figure()
        model_df = df[df['model'] == model]
        tasks = sorted(model_df['task'].unique())

        for i, task_name in enumerate(tasks):
            task_df = model_df[model_df['task'] == task_name]
            error_df = task_df[~task_df['is_correct']]
            col = 'word_length'

            grouped = error_df.groupby(col)['question_idx'].apply(set)
            dist = task_df.groupby(col)['question_idx'].apply(set).to_dict()

            lengths = sorted([l for l in grouped.keys()])
            avg_acc = [(len(dist[l]) - len(grouped[l])) / len(dist[l]) for l in lengths]
            std = [((p * (1 - p)) / len(dist[l])) ** 0.5 for p, l in zip(avg_acc, lengths)]

            x_vals = task_df[col].tolist()
            y_vals = task_df['is_correct'].astype(int).tolist()
            corr = pearsonr(x_vals, y_vals)[0]

            plt.errorbar(lengths, avg_acc, yerr=std, marker=MARKERS[i % len(MARKERS)],
                         linestyle=LINESTYLES[i % len(LINESTYLES)], capsize=3,
                         label=f'{task_name} (r={corr:.2f})')

        plt.ylabel("Average Accuracy",fontsize='x-large')
        plt.xlabel("Number of Characters",fontsize='x-large')
        plt.title(get_short_model_name(model))
        plt.legend( loc="best", fontsize='large')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_folder / f"{get_short_model_name(model)}.png", dpi=300, bbox_inches="tight")

def get_metric_correlations_table(df, args, task, output_folder):
    df = df.copy()
    metric_cols = ['word_length','gold_truth','gold_truth_devided_by_word_length','number_of_tokens','compression_rate','target_token_length',]
    tasks = ['find_first_occurrence','find_last_occurrence','count_character_frequency','count_unique_chars']
    rows = []
    for t in tasks:
        task_df = df[(df['task'] == t) & (df['model'].isin(args.models))]
        model_corr_matrix = []
        for model in args.models:
            model_df = task_df[task_df['model'] == model]
            corrs = []
            for col in metric_cols:
                x = model_df[col].tolist()
                y = model_df['is_correct'].astype(int).tolist()
                if len(set(x)) > 1 and len(set(y)) > 1:
                    v = pearsonr(x, y)[0]
                    v = round(v, 3)
                else:
                    v = np.nan
                corrs.append(v)
            row = {'task': t, 'model': model}
            row.update(dict(zip(metric_cols, corrs)))
            rows.append(row)
            model_corr_matrix.append(corrs)
        if model_corr_matrix:
            arr = np.array(model_corr_matrix, dtype=float)
            avg_vals = np.nanmean(arr, axis=0)
            avg_row = {'task': t, 'model': 'AVG'}
            avg_row.update({m: (round(v, 3) if not np.isnan(v) else np.nan) for m, v in zip(metric_cols, avg_vals)})
            rows.append(avg_row)
    table_df = pd.DataFrame(rows)
    out_path = Path(output_folder) / "all_tasks_correlation_table.csv"
    table_df.to_csv(out_path, index=False)
    return table_df

def get_errors_overlap_heatmap(df,args,output_folder):
    def calculate_overlap_matrix(models, model_errors):
        """Calculate the overlap matrix between model errors."""
        overlap_matrix = np.zeros((len(models), len(models)))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                intersection = len(model_errors[model1] & model_errors[model2])
                union = len(model_errors[model1] | model_errors[model2])
                if union > 0:
                    overlap_matrix[i, j] = intersection / union
        return overlap_matrix

    def create_heatmap(overlap_matrix, models, task, output_dir):
        plt.figure(figsize=(12, 10))  # wider helps long names
        ax = sns.heatmap(
            overlap_matrix,
            xticklabels=[get_short_model_name(m) for m in models],
            yticklabels=[get_short_model_name(m) for m in models],
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            vmin=0, vmax=1,
            annot_kws={"size": 18},
        )

        # Axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20)

        # Colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"error_overlap_heatmap-{task}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    tasks = ['find_first_occurrence','find_last_occurrence','count_character_frequency','count_unique_chars']
    for t in tasks:
        task_df = df[df['task']==t]
        models = args.models
        model_errors = {}
        
        for model in models:
            wrong_examples = set(task_df[
                (task_df['model'] == model) & 
                (task_df['is_correct'] == False)
            ]['question_idx'].unique())
            model_errors[model] = wrong_examples

        mat = calculate_overlap_matrix(models,model_errors)
        create_heatmap(mat,models,t,output_folder)


def analyze_mixed_case_performance(df, task, models, output_folder):
    output_folder = Path(output_folder) / task
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    task_df = df[(df['task'] == task) & (df['model'].isin(models))].copy()
    word_col = 'word'
    char_col = 'character'
    def has_mixed_case(word, ch, t):
        if t in ['find_first_occurrence','find_last_occurrence'] and char_col is not None:
            return (str(ch).upper() in str(word)) and (str(ch).lower() in str(word))
        return any(c.isupper() for c in str(word)) and any(c.islower() for c in str(word))
    task_df['has_mixed_case'] = task_df.apply(lambda r: has_mixed_case(r[word_col], r[char_col] if char_col in task_df.columns else None, r['task']), axis=1)
    acc = task_df.groupby(['model','has_mixed_case'])['is_correct'].mean().reset_index()
    acc['model_short'] = acc['model'].apply(get_short_model_name)
    plt.figure(figsize=(12,6))
    sns.barplot(data=acc, x='model_short', y='is_correct', hue='has_mixed_case')
    plt.ylabel("Accuracy", fontsize='x-large')
    plt.xlabel("Model", fontsize='x-large')
    plt.legend(loc="best", fontsize='large', title='')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_folder / "mixed_case_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    return acc

def analyze_counting_bias(df, task, models, output_folder):
    output_folder = Path(output_folder) / task
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    task_df = df[(df['task'] == task) & (df['model'].isin(models))].copy()
    pred_col = 'model_prediction'
    gold_col = 'correct_answer'
    task_df[pred_col] = pd.to_numeric(task_df[pred_col], errors='coerce')
    task_df[gold_col] = pd.to_numeric(task_df[gold_col], errors='coerce')
    task_df = task_df[task_df['is_correct'] == False]
    task_df = task_df[task_df[gold_col] != 0]
    task_df['count_bias'] = task_df[pred_col] - task_df[gold_col]
    bias = task_df.groupby('model')['count_bias'].agg(['mean','std','count']).reset_index()
    bias['model_short'] = bias['model'].apply(get_short_model_name)
    x = np.arange(len(bias))
    plt.figure(figsize=(12,6))
    plt.bar(bias['model_short'], bias['mean'])
    plt.errorbar(bias['model_short'], bias['mean'], yerr=bias['std'], fmt='none', capsize=5)
    plt.axhline(0, linestyle='--', alpha=0.3)
    plt.ylabel("Mean Prediction Bias", fontsize='x-large')
    plt.xlabel("Model", fontsize='x-large')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "counting_bias.png", dpi=300, bbox_inches="tight")
    plt.close()
    return bias
    
def main(args):
    output_folder = Path(args.output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    df = load_and_filter_data(args.input)
    df = add_inferred_columns(df)

    indexing_tasks = ['find_first_occurrence','find_last_occurrence']
    indexing_df = df.copy()
    indexing_df = indexing_df[indexing_df['task'].isin(indexing_tasks)]

    for task in indexing_tasks:
        task_df = indexing_df.copy()
        task_df = task_df[task_df['task'] == task]
        curr_output_folder = output_folder / 'word_length_vs_target_token'
        if not os.path.exists(curr_output_folder):
            os.mkdir(curr_output_folder)
        analyze_word_length_vs_target_token(task_df,task,args.models,curr_output_folder)

    # count_character_frequency
    ccf_df = df[df['task']=='count_character_frequency']
    curr_output_folder = output_folder / 'gold_truth'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    analyze_tokens_spread(ccf_df,'count_character_frequency',args.models,curr_output_folder)

    # count_unique_chars
    cuc_df = df[df['task']=='count_unique_chars']
    curr_output_folder = output_folder / 'number_of_tokens'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    analyze_number_of_tokens(cuc_df,args.models,curr_output_folder)

    # all tasks:

    curr_output_folder = output_folder / 'word_length'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    analyze_word_length(df,args.models,curr_output_folder)


    curr_output_folder = Path(output_folder) / 'correlation_table'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    get_metric_correlations_table(df,args,task,curr_output_folder)

    curr_output_folder = Path(output_folder) / 'error_overlap'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    get_errors_overlap_heatmap(df,args,curr_output_folder)

    curr_output_folder = Path(output_folder) / 'mixed_case'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    for t in ['find_first_occurrence','find_last_occurrence','count_character_frequency','count_unique_chars']:
        analyze_mixed_case_performance(df, t, args.models, curr_output_folder)

    curr_output_folder = Path(output_folder) / 'counting_bias'
    if not os.path.exists(curr_output_folder):
        os.mkdir(curr_output_folder)
    for t in ['count_character_frequency','count_unique_chars']:
        analyze_counting_bias(df, t, args.models, curr_output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis Parser")
    parser.add_argument("-i", "--input",default="evaluation_results.csv",help="Path to the results file")
    parser.add_argument("-o", "--output_folder",default='plots')
    parser.add_argument("-m", "--models",default=[
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo"
    ]
    ,help="Models to run the analysis on")
    args = parser.parse_args()
    main(args)
