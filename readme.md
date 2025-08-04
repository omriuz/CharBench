# CharBench

Character-level benchmark and analysis suite for LLMs.

The dataset is available [through HuggingFace.](https://huggingface.co/datasets/omriuz/CharBench)

## Installation

1. Use Python 3.10 or newer.
2. From the repo root, install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```
3. Create a `.env` file with your credentials:

   ```
   OPENAI_API_KEY=your_openai_key
   TOGETHER_API_KEY=your_together_key
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```

OpenAI and Together require valid API keys. Hugging Face tokenization requires an access token.

## Full reproducibility

Run the complete pipeline:

```bash
python main.py
```

This builds the benchmark, evaluates all models used in the paper, runs tokenization with each tokenizer, and produces analysis plots.

Expected outputs:

1. `benchmark.csv`
2. `evaluation_results.csv`
3. `plots/` (directory with figures)

## Modular reproducibility

Run individual steps as needed:

1. Construct the benchmark

   ```bash
   python benchmark/benchmark.py
   ```

2. Evaluate OpenAI models via API

   ```bash
   python evaluation/eval_openai.py
   ```

3. Evaluate Together.ai models via API

   ```bash
   python evaluation/eval_together.py
   ```

4. Tokenize benchmark words with Hugging Face and TikToken

   ```bash
   python evaluation/tokenize_results.py
   ```

5. Run analysis and generate plots

   ```bash
   python analysis/analyze.py
   ```

## Modifications
These are the possible args and flags for each script:

### `main.py`

```bash
python main.py [--steps ...] [-c CORPUS] [-n NUM] [-b PATH] [-s SEED] [-e PATH] [--task TASK] [-p DIR]
```

Args: `--steps` default "create\_benchmark eval\_openai eval\_together tokenize analyze", `-c/--corpus` `JeanKaddour/minipile`, `-n/--num_unique_words` `100000`, `-b/--benchmark_file` `benchmark.csv`, `-s/--seed` `42`, `-e/--evaluation_file` `evaluation_results.csv`, `--task` `all`, `-p/--output_plots_folder` `plots`.
### `evaluation/eval_openai.py`

```bash
python evaluation/eval_openai.py [-m MODELS...] [-i PATH] [-o PATH] [--batch_size N] [--task TASK]
```

Defaults: `-m` `gpt-4o gpt-4o-mini gpt-3.5-turbo`, `-i` `benchmark.csv`, `-o` `evaluation_results.csv`, `--batch_size` `100`, `--task` `all`.

### `evaluation/eval_together.py`

```bash
python evaluation/eval_together.py [-m MODELS...] [-i PATH] [-o PATH] [--task TASK] [--batch_size N]
```

Defaults: `-m` `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo meta-llama/Llama-3.3-70B-Instruct-Turbo deepseek-ai/DeepSeek-V3 mistralai/Mistral-7B-Instruct-v0.2`, `-i` `benchmark.csv`, `-o` `evaluation_results.csv`, `--task` `all`, `--batch_size` `20`.

### `evaluation/tokenize_results.py`

```bash
python evaluation/tokenize_results.py [-i PATH]
```

Defaults: `-i/--input` `valuation_results.csv`.

### `analysis/analyze.py`

```bash
python analysis/analyze.py [-i PATH] [-o DIR] [-m MODELS...]
```

Defaults: `-i/--input` `evaluation_results.csv`, `-o/--output_folder` `plots`, `-m/--models` `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo, meta-llama/Llama-3.3-70B-Instruct-Turbo, deepseek-ai/DeepSeek-V3, mistralai/Mistral-7B-Instruct-v0.2, gpt-4o, gpt-4o-mini, gpt-3.5-turbo`.
