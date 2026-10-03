import os
import ast
import asyncio
import inspect
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    BaseMetric,
    ContextRelevance,
    Faithfulness,
)

from rag import rag


HERE = Path(__file__).resolve().parent
eval_dataset = HERE / "eval_dataset.csv"
after_rag_dataset = HERE / "after_rag_dataset.csv"
metrics_report_path = HERE / "metrics_report.csv"
metrics_summary_path = HERE / "metrics_summary.csv"


def run_rag(eval_dataset_pth, after_rag_dataset_pth) -> None:
    df = pd.read_csv(eval_dataset_pth)
    responses = []
    retrieved_contexts = []
    for user_input in df["user_input"]:
        result = rag(user_input)
        responses.append(result["answer"])
        retrieved_contexts.append(result["retrieved_contexts"])
    df["response"] = responses
    df["retrieved_contexts"] = retrieved_contexts
    df.to_csv(after_rag_dataset_pth, index=False)


def load_after_rag_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    for row in rows:
        contexts = row["retrieved_contexts"]
        if isinstance(contexts, str):
            row["retrieved_contexts"] = ast.literal_eval(contexts)
    return rows


def ensure_after_rag_rows() -> list[dict]:
    if not after_rag_dataset.exists():
        print(f"{after_rag_dataset.name} not found, running RAG...")
        run_rag(eval_dataset, after_rag_dataset)
    else:
        print(f"using existing {after_rag_dataset.name}")
    return load_after_rag_rows(after_rag_dataset)


def build_judge():
    openai_client = AsyncOpenAI(
        api_key=os.getenv("AI_TUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/",
    )
    judge_llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
    judge_embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-3-small",
    )
    return judge_llm, judge_embeddings


def build_metrics(llm, embeddings) -> list[BaseMetric]:
    return [
        # ContextRelevance(llm=llm),
        Faithfulness(llm=llm),
        # AnswerRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]


def required_fields(metric: BaseMetric) -> list[str]:
    """Метрика сама объявляет свои входы в сигнатуре ascore()."""
    params = inspect.signature(metric.ascore).parameters
    return [name for name in params if name != "self"]


async def score_dataset(
    rows: list[dict],
    metrics: list[BaseMetric],
    max_concurrency: int = 8,
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score_one(row: dict, metric: BaseMetric) -> float:
        fields = {name: row[name] for name in required_fields(metric)}
        async with semaphore:
            try:
                result = await metric.ascore(**fields)
                return float(result.value)
            except Exception as exc:
                print(f"  {metric.name} failed: {exc}")
                return float("nan")

    tasks = [score_one(row, metric) for row in rows for metric in metrics]
    values = await asyncio.gather(*tasks)

    report = pd.DataFrame(rows)
    for i, metric in enumerate(metrics):
        report[metric.name] = values[i :: len(metrics)]
    return report


def save_aggregation(report: pd.DataFrame, metric_names: list[str], save_pth: Path) -> None:
    overall = report[metric_names].mean(skipna=True).to_frame().T
    overall.insert(0, "scope", "all")
    overall.insert(1, "n", len(report))

    by_synth = (
        report.groupby("synthesizer_name", dropna=False)[metric_names]
        .mean(skipna=True)
        .reset_index()
        .rename(columns={"synthesizer_name": "scope"})
    )
    counts = report.groupby("synthesizer_name", dropna=False).size()
    by_synth.insert(1, "n", by_synth["scope"].map(counts).astype(int))

    summary = pd.concat([overall, by_synth], ignore_index=True)
    summary.to_csv(save_pth, index=False)
    print(summary.to_string(index=False))


async def main() -> None:
    judge_llm, judge_embeddings = build_judge()
    rows = ensure_after_rag_rows()
    metrics = build_metrics(judge_llm, judge_embeddings)
    metric_names = [m.name for m in metrics]
    print(f"scoring {len(rows)} rows x {metric_names}")

    report = await score_dataset(rows, metrics)
    report.to_csv(metrics_report_path, index=False)
    print(f"saved {metrics_report_path.name}")

    save_aggregation(report, metric_names, metrics_summary_path)
    print(f"saved {metrics_summary_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
