import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def generate_sample(
    s: int, correct_tools: list[str], tool_df_dedup: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    correct_rows = tool_df_dedup.loc[tool_df_dedup["name"].isin(correct_tools)]
    available_df = tool_df_dedup[~tool_df_dedup["name"].isin(correct_tools)]
    random_idx = rng.choice(len(available_df), size=s - len(correct_tools), replace=False)
    sample = available_df.iloc[random_idx]
    return pd.concat([sample, correct_rows])


def filter_tools(K: int, question_embedding, sample_df: pd.DataFrame):
    scores = cosine_similarity(
        [question_embedding], np.stack(sample_df["embeddings"].values)
    )
    top_k_idx = np.argsort(scores[0])[-K:][::-1]
    return sample_df.iloc[top_k_idx]


def evaluate_row(question_embedding, gt_tool_name, tool_df_dedup, s, k, seed):
    rng = np.random.default_rng(seed)
    sample_df = generate_sample(s, [gt_tool_name], tool_df_dedup, rng)
    filt_df = filter_tools(k, question_embedding, sample_df)
    return gt_tool_name in filt_df["name"].values


def embed_questions(client: OpenAI, questions: list[str], batch_size: int = 2048):
    all_embeddings = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Embedding questions"):
        batch = questions[i : i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-large", input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings


def run_config(eval_df, tool_df_dedup, s, k, seed, max_workers):
    futures = {}
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in eval_df.iterrows():
            row_seed = seed + idx
            fut = executor.submit(
                evaluate_row,
                row["question_embedding"],
                row["gt_tool_names"],
                tool_df_dedup,
                s, k, row_seed,
            )
            futures[fut] = idx

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"S={s}, K={k}"
        ):
            results.append(fut.result())

    acc = sum(results) / len(results) * 100
    return acc, sum(results), len(results)


def main():
    parser = argparse.ArgumentParser(description="RAG tool choice evaluation")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10], help="Top-K values to evaluate")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[200, 150], help="Sample pool sizes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    client = OpenAI()

    # Load data
    tool_df = pd.read_pickle("../data/pkls/ntokens_embeddings_tool_df.pkl")
    eval_df = pd.read_pickle("../data/pkls/multiple_tools.pkl")
    eval_df["gt_tool_names"] = eval_df["ground_truth"].apply(lambda x: list(x[0].keys())[0])

    # Validate that all gt tools exist
    tool_df_dedup = tool_df.drop_duplicates(subset=["name"])
    known_tools = set(tool_df_dedup["name"].values)
    unknown = set(eval_df["gt_tool_names"]) - known_tools
    if unknown:
        print(f"Warning: {len(unknown)} unknown ground-truth tools will be skipped: {unknown}")
        eval_df = eval_df[eval_df["gt_tool_names"].isin(known_tools)]

    # Batch-embed all questions upfront
    questions = eval_df["question"].astype(str).tolist()
    eval_df["question_embedding"] = embed_questions(client, questions)

    # Evaluate all (S, K) configs
    configs = list(product(args.sample_sizes, args.top_k))
    print(f"\nRunning {len(configs)} configs: {configs}\n")

    results = {}
    for s, k in configs:
        acc, correct, total = run_config(eval_df, tool_df_dedup, s, k, args.seed, args.workers)
        results[(s, k)] = acc
        print(f"  S={s}, K={k} -> Accuracy: {acc:.2f}% ({correct}/{total})\n")

    # Summary table
    print("=" * 40)
    print(f"{'S':>6} {'K':>4} {'Accuracy':>10}")
    print("-" * 40)
    for (s, k), acc in results.items():
        print(f"{s:>6} {k:>4} {acc:>9.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
