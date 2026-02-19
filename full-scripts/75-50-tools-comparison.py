import pandas as pd
import numpy as np
import json
import time
import os
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel
from typing import Literal
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "datasets", "BFCL_v4_live_multiple.json")
ANSWERS_PATH = os.path.join(DATA_DIR, "datasets", "possible_answer", "BFCL_v4_live_multiple.json")

K = 5    # top-k for semantic retrieval
MAX_WORKERS = 8
EMBEDDING_BATCH_SIZE = 100

client = OpenAI()  # uses OPENAI_API_KEY env var


# ── 1. Data Collection ───────────────────────────────────────────────────────

def change_gt_name(tools: list[dict]) -> list[dict]:
    for d in tools:
        old_key = next(iter(d))
        value = d.pop(old_key)
        new_key = old_key.replace(".", "_")
        d[new_key] = value
    return tools


def change_function_name(tools: list[dict]) -> list[dict]:
    for d in tools:
        d["name"] = d["name"].replace(".", "_")
    return tools


def merge_list(l: list[dict]) -> list[dict]:
    merged = []
    for d in l:
        if d in merged:
            continue
        merged.append(d)
    return merged


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("── Data Collection ─────────────────────────────────")
    df = pd.read_json(DATASET_PATH, lines=True)
    answers = pd.read_json(ANSWERS_PATH, lines=True)

    df["ground_truth"] = answers["ground_truth"].apply(change_gt_name)
    df["function"] = df["function"].apply(change_function_name)

    tools = df["function"].apply(merge_list).sum()
    tool_df = pd.DataFrame(tools)

    print(f"  Loaded {len(df)} eval rows, {len(tool_df)} tool entries "
          f"({tool_df['name'].nunique()} unique names)")
    return df, tool_df


# ── 2. Prepare Test Set (embeddings, tokens, OAI format) ─────────────────────

def batch(l: list, n: int):
    for i in range(0, len(l), n):
        yield l[i : i + n]


TYPE_MAP = {"dict": "object", "float": "number", "any": "string", "tuple": "array"}


def fix_types(schema: dict) -> dict:
    schema = schema.copy()
    if schema.get("type") in TYPE_MAP:
        schema["type"] = TYPE_MAP[schema["type"]]
    if "properties" in schema:
        schema["properties"] = {k: fix_types(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = fix_types(schema["items"])
    return schema


def to_oai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": fix_types(tool["parameters"]),
    }


def prepare_test_set(tool_df: pd.DataFrame) -> pd.DataFrame:
    print("── Preparing Test Set ──────────────────────────────")
    pkl_path = os.path.join(DATA_DIR, "pkls", "ntokens_embeddings_tool_df.pkl")
    try:
        tool_df = pd.read_pickle(pkl_path)
        print(f"  READ TOOL_DF PICKLE SUCCESSFUL ({pkl_path})")
        return tool_df
    except FileNotFoundError:

        # Embeddings
        tool_df["embedding_input"] = (
            "Name: " + tool_df["name"] + "\n" + "Description: " + tool_df["description"]
        )

        embeddings = []
        for x_b in tqdm(
            batch(tool_df["embedding_input"].tolist(), EMBEDDING_BATCH_SIZE),
            desc="  Embedding tools",
        ):
            response = client.embeddings.create(model="text-embedding-3-large", input=x_b)
            embeddings.extend([item.embedding for item in response.data])

        tool_df["embeddings"] = embeddings

        # Token counts
        encoding = tiktoken.encoding_for_model("gpt-5")
        tool_df["formatted"] = tool_df.apply(
            lambda row: row.loc[["name", "description", "parameters"]].to_dict(), axis=1
        )
        tool_df["n_tokens"] = tool_df["formatted"].apply(
            lambda x: len(encoding.encode(str(x)))
        )
        tool_df["name"] = tool_df["name"].str.replace(".", "_", regex=False)

        # OAI tool format
        tool_df["oai_format"] = tool_df["formatted"].apply(to_oai_tool)

        print(f"  Token stats — mean: {tool_df['n_tokens'].mean():.1f}, "
              f"median: {tool_df['n_tokens'].median():.1f}, "
              f"max: {tool_df['n_tokens'].max()}")
        return tool_df


# ── 3. Experiment ─────────────────────────────────────────────────────────────

class EvalOutput(BaseModel):
    correct: bool
    reason: Literal[
        "wrong parameters",
        "required tool call was not made",
        "unnecessary tool call was made",
        "this situation did not require a tool call",
        "correct"
    ]


def generate_sample(
    s: int, correct_tools: list[str], tool_df: pd.DataFrame
) -> pd.DataFrame:
    if not all(tool in tool_df["name"].values for tool in correct_tools):
        raise ValueError(
            "Unknown tool(s) being referenced: ", correct_tools
        )

    tool_df_dedup = tool_df.drop_duplicates(subset=["name"])
    correct_rows = tool_df_dedup.loc[tool_df_dedup["name"].isin(correct_tools)]

    available_df = tool_df_dedup[~tool_df_dedup["name"].isin(correct_tools)]
    random_idx = np.random.choice(
        len(available_df), size=s - len(correct_tools), replace=False
    )
    sample = available_df.iloc[random_idx]

    return pd.concat([sample, correct_rows])


def check_answer(
    response, ground_truth: list[dict], tool_schemas: list[dict]
) -> tuple[bool, str | None]:
    llm_called_tools = []
    items = []
    ground_truth_names = [key for tool in ground_truth for key in tool.keys()]

    for item in response.output:
        if item.type == "function_call":
            llm_called_tools.append(item.name)
            items.append(item)

    correct = Counter(llm_called_tools) == Counter(ground_truth_names)

    if correct:
        # Build schema lookup for the GT tools only
        gt_schemas = [s for s in tool_schemas if s["name"] in ground_truth_names]
        prompt = [
            {"role": "system", "content": (
                "Evaluate whether the LLM output matches the Ground Truth. "
                "The tool schemas are provided so you can check default values. "
                "Default values might be passed by the LLM but not included in the Ground Truth. "
                "That is not incorrect."
            )},
            {"role": "user", "content": (
                f"Tool Schemas:\n{json.dumps(gt_schemas, indent=2)}\n\n"
                f"Ground Truth:\n{ground_truth}\n\n"
                f"LLM Output:\n{items}"
            )},
        ]
        resp = client.responses.parse(
            model="gpt-5-mini",
            input=prompt,
            text_format=EvalOutput,
        )
        return resp.output_parsed.correct, resp.output_parsed.reason
    else:
        is_subset = Counter(ground_truth_names) <= Counter(llm_called_tools)
        if is_subset:
            return False, "unnecessary tool call was made"
        else:
            return False, "required tool call was not made"


def filter_tools(K: int, row, sample_df: pd.DataFrame):
    response = client.embeddings.create(model="text-embedding-3-large", input=str(row['question']))
    question_embedding = response.data[0].embedding
    scores = cosine_similarity([question_embedding], np.stack(sample_df['embeddings'].values))
    top_k_idx = np.argsort(scores[0])[-K:][::-1]
    return sample_df.iloc[top_k_idx]



def run_row(row, tool_df: pd.DataFrame, s: int, use_rag: bool = True):
    sample_df = generate_sample(
        s=s, correct_tools=row["correct_tool"], tool_df=tool_df
    )

    if use_rag:
        filtered_tools = filter_tools(K, row, sample_df)
        tools = filtered_tools["oai_format"].tolist()
    else:
        tools = sample_df["oai_format"].tolist()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.responses.create(
                model="gpt-5-mini",
                input=[{"role": "system", "content": "Only parameterize the fields specified in a query. Do not overwrite defaults unless required."}] + row["question"][0],
                tools=tools,
                tool_choice="required",
            )
            latency = time.time() - start_time
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(10)
                continue
            raise ValueError(f"Error: {e}\n\n{tools[2]}")

    correct, reason = check_answer(response, row["ground_truth"], tools)
    return correct, reason, latency, response


def run_experiment(eval_df: pd.DataFrame, tool_df: pd.DataFrame, s: int, use_rag: bool = True) -> pd.DataFrame:
    mode = "RAG" if use_rag else "No-RAG"
    print(f"── Experiment [{mode}] (s={s}, workers={MAX_WORKERS}) ──────────────────")

    eval_df["correct_tool"] = eval_df["ground_truth"].apply(
        lambda x: [s for item in x for s in item.keys()]
    )

    results = [None] * len(eval_df)

    print(f"  Eval rows: {len(eval_df)}")
    print(f"  Unique tools: {tool_df['name'].nunique()}")
    print(f"  Sample size (S): {s}, Top-K filter: {K if use_rag else 'N/A'}, Workers: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_row, row, tool_df, s, use_rag): i
            for i, (_, row) in enumerate(eval_df.iterrows())
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Evaluating"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = (None, f"error: {e}", None, None)

    results_df = eval_df.copy()
    results_df["correct"], results_df["reason"], results_df["latency"], results_df["response"] = zip(
        *results
    )

    # Normalize reasons
    results_df.loc[results_df["correct"] == True, "reason"] = "Correct"
    results_df.loc[
        results_df["reason"] == "required tool call was not made", "reason"
    ] = "missing required tool call"

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_single(eval_df: pd.DataFrame, tool_df: pd.DataFrame, s: int, use_rag: bool):
    mode = "RAG" if use_rag else "No-RAG"
    tag = "rag" if use_rag else "norag"

    results_df = run_experiment(eval_df, tool_df, s=s, use_rag=use_rag)

    avg_tokens = tool_df['n_tokens'].mean() * (K if use_rag else s)
    print(f"\n── Results [{mode}, s={s}] ─────────────────────────────────")
    print(results_df["reason"].value_counts().to_string())
    print(f"\n  Accuracy: {results_df['correct'].mean():.2%}")
    print(f"  Mean latency: {results_df['latency'].mean():.2f}s")
    print(f"  Avg tool tokens per sample: {avg_tokens:.0f}")

    out_path = os.path.join(DATA_DIR, f"{s}-tools-gpt-5-{tag}-results.pkl")
    results_df.to_pickle(out_path)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    # 1 ─ Data collection (shared across all runs)
    eval_df, tool_df = collect_data()

    # 2 ─ Prepare test set (shared across all runs)
    tool_df = prepare_test_set(tool_df)

    # 3 ─ Run all 6 experiments back-to-back
    S_VALUES = [100, 75, 50]

    for s in S_VALUES:
        for use_rag in [False, True]:
            mode = "RAG" if use_rag else "No-RAG"
            print(f"\n{'='*60}")
            print(f"  STARTING: s={s}, mode={mode}")
            print(f"{'='*60}\n")
            run_single(eval_df.copy(), tool_df, s=s, use_rag=use_rag)

    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
