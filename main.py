import argparse
import subprocess

def run_step(cmd):
    print(f"\n[RUNNING] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main(args):
    steps = {
        "create_benchmark": f"python benchmark/benchmark.py -c {args.corpus} -n {args.num_unique_words} -b {args.benchmark_file} -s {args.seed}",
        "eval_openai": f"python evaluation/eval_openai.py -i {args.benchmark_file} -o {args.evaluation_file}",
        "eval_together": f"python evaluation/eval_together.py -i {args.benchmark_file} -o {args.evaluation_file}",
        "tokenize": f"python evaluation/tokenize_results.py -i {args.evaluation_file}",
        "analyze": f"python analysis/analyze.py -i {args.evaluation_file} -o {args.output_plots_folder}"
    }

    for step in args.steps:
        run_step(steps[step])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=[
        "create_benchmark",
        "eval_openai",
        "eval_together",
        "tokenize",
        "analyze"
        ])
    parser.add_argument("-c", "--corpus",default="JeanKaddour/minipile")
    parser.add_argument("-n","--num_unique_words",default=100000,help="The number of unique words to extract from the corpus")
    parser.add_argument("-b", "--benchmark_file",default=f"benchmark.csv")
    parser.add_argument("-s","--seed",default=42)
    parser.add_argument("-e","--evaluation_file", type=str, default="evaluation_results.csv")
    parser.add_argument("--task", type=str, default="all", help="Task to evaluate")
    parser.add_argument("-p", "--output_plots_folder",default='plots')
    args = parser.parse_args()
    main(args)
