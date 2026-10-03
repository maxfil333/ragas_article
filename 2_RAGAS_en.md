[[RAG. Theory]]

# Evaluation metrics

> Written and verified against `ragas 0.4.3`.
>
> This is part two. In part one we built `eval_dataset.csv`, a synthetic dataset of questions and reference answers. Here we finally get the numbers we can use to compare RAG configurations.

## What we have, and what is missing

Recall where part one left us. The dataset has four substantive columns; the rest are generation metadata:

| column               | contents                                              |
|----------------------|-------------------------------------------------------|
| `user_input`         | the generated question                                |
| `reference`          | the reference answer                                  |
| `reference_contexts` | the chunks the question was generated from            |
| `synthesizer_name`   | question type: single-hop / multi-hop specific/abstract |

What the dataset does not contain, and cannot contain, is **your system's answer** (`response`) and **the context your retriever fetched** (`retrieved_contexts`). The dataset knows nothing about your configuration. It describes only the task. The first step of evaluation is therefore not the metrics. It is **running your RAG over every `user_input`**.

## How the metrics are computed: the full picture

As in part one, we start by listing everything we prepare ourselves:

```
eval_dataset.csv             from part 1: user_input, reference, reference_contexts, ...
rag(question)                your RAG pipeline: returns response and retrieved_contexts
judge_llm, judge_embeddings  the judge: llm_factory(...) and OpenAIEmbeddings(...)
metrics                      the list of metrics from ragas.metrics.collections
max_concurrency              how many judge requests to keep in flight at once
```

Block numbering continues from part one, which ended at `[4]`:

```
eval_dataset.csv
   │
   ▼
[5] run_rag(eval_dataset.csv, after_rag_dataset.csv)
       if after_rag_dataset.csv is missing, call your RAG on every row
       and append two columns: response and retrieved_contexts
       if the file already exists, read it and do not rerun RAG
       this is the only step that depends on your configuration
   │
   │  rows: a list of dicts with every field the metrics need
   ▼
[6] metrics = build_metrics(judge_llm, judge_embeddings)
       each metric declares, in the ascore() signature, which fields it needs
   │
   │  metrics: a list of BaseMetric objects
   ▼
[7] scores = await score_dataset(rows, metrics, max_concurrency)
       our own runner: for each (row, metric) pair we call
       metric.ascore(**fields) under a semaphore
       ragas does not ship a ready-made runner for these metrics
   │
   │  scores: a table of rows × metrics
   ▼
[8] aggregation
       column means via nanmean, a breakdown by synthesizer_name,
       and, necessarily, a noise estimate from a repeated run
   │
   ▼
metrics_report.csv:  user_input | context_relevance | faithfulness | ... | synthesizer_name
metrics_summary.csv: scope | n | faithfulness | ...   (all + a breakdown by synthesizer_name)
```

> Code blocks below that carry a path comment (for example `# ragas/metrics/base.py`) show **library internals**. You do not write these yourself. They are here so you can see where the constraints we adapt to come from.

## Two metric systems, and why we use the second

In 0.4.3 the metrics live in two places at once, and the two should be kept apart.

`ragas.metrics` is the older system. It works with the stock `evaluate()` function, but the import warns you outright:

```
DeprecationWarning: Importing Faithfulness from 'ragas.metrics' is deprecated
and will be removed in v1.0. Please use 'ragas.metrics.collections' instead.
```

`ragas.metrics.collections` is the current system. That is the one we use from here on.

Metrics in `collections` inherit from `SimpleBaseMetric`, not from `Metric`.
We do not call `evaluate()`. We write the runner ourselves. That is less work than it sounds: the runner is twenty lines, and we walk through it in section `[7]`. The payoff is that `EvaluationDataset` and `SingleTurnSample` drop out along the way. They exist only for `evaluate()`. The `collections` metrics take ordinary dictionaries.

## The metric map: what each one measures

It helps to lay RAG metrics out on two axes at once: **which layer** they check, and **whether they need a reference**. The second decides whether you can compute the metric from production logs, or whether you need the dataset from part one.

| Metric                             | `ascore()` inputs                                           | layer       | needs `reference` |
|------------------------------------|-------------------------------------------------------------|-------------|-------------------|
| `ContextRelevance`                 | `user_input`, `retrieved_contexts`                          | retrieval   | no                |
| `ContextPrecisionWithoutReference` | `user_input`, `response`, `retrieved_contexts`              | retrieval   | no                |
| `ContextPrecisionWithReference`    | `user_input`, `reference`, `retrieved_contexts`             | retrieval   | yes               |
| `ContextRecall`                    | `user_input`, `retrieved_contexts`, `reference`             | retrieval   | yes               |
| `ContextEntityRecall`              | `reference`, `retrieved_contexts`                           | retrieval   | yes               |
| `Faithfulness`                     | `user_input`, `response`, `retrieved_contexts`              | generation  | no                |
| `ResponseGroundedness`             | `response`, `retrieved_contexts`                            | generation  | no                |
| `AnswerRelevancy`                  | `user_input`, `response` (+ embeddings)                     | generation  | no                |
| `AnswerCorrectness`                | `user_input`, `response`, `reference` (+ embeddings)        | end-to-end  | yes               |
| `FactualCorrectness`               | `response`, `reference`                                     | end-to-end  | yes               |
| `NoiseSensitivity`                 | `user_input`, `response`, `reference`, `retrieved_contexts` | diagnostics | yes               |


_The “inputs” column is the signature of `ascore()`. The signature does not always match intuition, and `Faithfulness` is exactly that case. Conceptually the metric compares the answer with the context, so the question looks unnecessary. In the implementation it is **required**: an empty `user_input` makes the metric raise `ValueError`, and the question itself is inserted into the prompt of the first step, where the answer is split into statements. The practical consequence is that you cannot compute `Faithfulness` from an export that saved only the answer and the context. `ResponseGroundedness`, by contrast, measures almost the same thing and does not require the question._

That yields the practical set we use below. One metric for each question we want to ask of the system:

- **How relevant is the retrieved context to the user's question?** → `ContextRelevance`. The only metric in the set that does not look at the answer at all.
- **Does the answer actually follow from the retrieved context?** → `Faithfulness`.
- **Did the user get an answer to their question?** → `AnswerRelevancy`.
- **How closely does the answer match the reference?** → `AnswerCorrectness`. The end-to-end metric.

## Mechanics: what happens inside a single score

Here is how the scores are actually computed.

### ContextRelevance

_Estimates how relevant the retrieved context passed to the LLM (`retrieved_contexts`) is to the original user query (`user_input`). It shows whether retrieval gave the model genuinely useful information, or a large volume of irrelevant documents._

It is built differently from the other three, which is why we take it first. There is no split into statements. All `retrieved_contexts` are concatenated into one string, and **two independent judges** — two different prompts — score that string's relevance to the question. Each assigns an integer: 0, 1, or 2. The ratings are divided by 2 and averaged:

```python
# ragas/metrics/collections/context_relevance/metric.py — library internals
judge1_rating = await self._get_judge_rating(self.judge1_prompt, user_input, context_str)
judge2_rating = await self._get_judge_rating(self.judge2_prompt, user_input, context_str)

score = self._average_scores(judge1_rating / 2.0, judge2_rating / 2.0)
```

That is its defining property: the score can take only **five** values — 0, 0.25, 0.5, 0.75, and 1. The scale is coarse, and that has to be kept in mind when you interpret it. The metric answers well the question “does the retriever fetch on-topic material, or does it bring back junk?” It is a poor detector of small improvements.

Two details from the code. If a judge returns a rating outside `[0, 1, 2]`, the metric retries up to `max_retries=5` times and, once those are exhausted, returns `NaN`. There are also a few degenerate cases in which the metric returns 0 outright: an empty context, a context identical to the question, or a context wholly contained in the question text.

### Faithfulness

_Estimates how far the answer (`response`) follows from the retrieved context (`retrieved_contexts`). An answer counts as faithful when every one of its statements is supported by the retrieved context._

Two LLM calls per row:

1. The answer is split into atomic statements.
2. All `retrieved_contexts` are concatenated into **one** string, and each statement is checked against it NLI-style: supported, or not.
3. The result: Faithfulness = (number of supported statements) / (total number of statements).

### AnswerRelevancy

_Did the answer (`response`) address the user's question (`user_input`)?_

Three LLM calls (`strictness=3`) plus two embedding requests. The logic runs opposite to intuition. From the **answer**, the metric generates a question that this answer would have settled — three times. It then takes the cosine between the original question and each of the three generated questions, and averages them.

Two things to note. First, the metric **does not look at the context at all**. It asks “did we answer the question?”, not “is this true?”. Second, there is a separate `noncommittal` flag. If all three runs judged the answer noncommittal (“the context contains no information about this”), the score is multiplied by zero. An honest “I don't know” is penalized all the way to zero. Keep that in mind when you compare configurations that use different system prompts.

### AnswerCorrectness

_How closely does the answer (`response`) match the reference answer (`reference`)?_

It requires ground truth and combines two independent sub-metrics as a weighted sum:

- Factual correctness — an F-measure on statement overlap. The generated answer and the ground truth are split into statements, classified as `TP` (present in both), `FP` (present only in the answer), or `FN` (present only in the ground truth). Then `F1 = TP / (TP + 0.5×(FP + FN))`.
- Semantic similarity — the cosine similarity of the embeddings of the full answer text and the ground truth. This reuses the separate `answer_similarity` metric.
- The final score is a weighted average of semantic similarity and the factual score. The default weights are `[0.75, 0.25]`, so factual overlap (F1) weighs three times as much as raw semantic closeness of the text.

### What this costs

Add up the calls per dataset row for our set:

| Metric              | LLM calls | embedding requests |
|---------------------|-----------|--------------------|
| `ContextRelevance`  | 2         | —                  |
| `Faithfulness`      | 2         | —                  |
| `AnswerRelevancy`   | 3         | 2                  |
| `AnswerCorrectness` | 3         | 2                  |
| **total per row**   | **10**    | **4**              |

On top of that, the RAG run itself: one generation call and one embedding request per row. That is **11** LLM calls per row (10 from the metrics, 1 from the generator). On the dataset from part one (8 rows) that is `8 × 11 = 88` LLM calls. The figure is small only because the dataset is tiny. On a thousand rows the same set is already `1000 × 11 = 11` thousand calls, and “which metrics do I actually need?” becomes a question of cost. (Eight rows demonstrate the method. They are not a production-config choice.)

## Building the evaluation

What follows is our code. The full file is `pipeline_calculate_metrics.py`.

### 1. The judge

```python
openai_client = AsyncOpenAI(api_key=os.getenv("YOUR_API_KEY"))
judge_llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
judge_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

This is the same `llm_factory` as in part one, and that is deliberate. Metrics from `collections` accept only this wrapper.

The choice of judge model deserves its own note. In our setup, `gpt-5-mini` both generates the answers and scores them. That is the cheapest option, and the most exposed one: models tend to score their own wording more highly. If the choice of configuration is a high-stakes one, take a judge stronger than the generator, and cross-check its scores against manual labels at least once.

### 2. Running RAG over the dataset ([5])

```python
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
```

If `after_rag_dataset.csv` is already present, we skip this step and read the finished file. Otherwise we run RAG over every `user_input` and save the result. The dataset describes the task; your system appends `response` and `retrieved_contexts`. Everything below is independent of the RAG configuration.

### 3. The metric set ([6])

```python
def build_metrics(llm, embeddings) -> list[BaseMetric]:
    return [
        # ContextRelevance(llm=llm),
        Faithfulness(llm=llm),
        # AnswerRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
```

`ContextRelevance` and `AnswerRelevancy` are left disabled for now. That saves 5 LLM calls and 2 embedding requests per row. When you want the full set, uncomment them.

Metrics that need only LLM verdicts are constructed with a single argument. `AnswerRelevancy` and `AnswerCorrectness` also compute cosines, so they are given the embeddings.

### 4. Our own runner ([7])

Ragas gives the metrics an `abatch_score` method, and at first glance it does the job. This is what it actually does:

```python
# ragas/metrics/base.py — library internals
async_tasks = []
for input_dict in inputs:
    async_tasks.append(self.ascore(**input_dict))

return await asyncio.gather(*async_tasks)
```

This is a bare `gather` with no concurrency limit. Eight rows will go through. A thousand rows means a thousand concurrent requests, and a refusal from the provider. That is why the runner is ours. It adds exactly three things the library version lacks: a semaphore, tolerance for a single score failing, and discovery of the required fields.

```python
def required_fields(metric: BaseMetric) -> list[str]:
    """The metric declares its own inputs in the ascore() signature."""
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
```

`required_fields` is worth a closer look. We do not maintain our own table of which metric needs which fields. The metric declares that itself in the `ascore()` signature, and we simply read the signature. Adding a new metric to the list then requires no change to the runner, and a typo in a field name surfaces immediately as a `KeyError`, not as a silent `NaN`.

Catching exceptions per score is deliberate as well. With LLM-as-judge metrics it is normal for one row in a hundred to fail on invalid structured output. Without the `try`, the whole `gather` is cancelled and you lose the results of the entire run.

### 5. Aggregation ([8])

```python
async def main() -> None:
    ...
    report = await score_dataset(rows, metrics)
    report.to_csv(metrics_report_path, index=False)
    save_aggregation(report, metric_names, metrics_summary_path)


def save_aggregation(report, metric_names, save_pth) -> None:
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

    pd.concat([overall, by_synth], ignore_index=True).to_csv(save_pth, index=False)
```

`skipna=True` is load-bearing here. `Faithfulness` returns `NaN` by design when no statement could be extracted from the answer, and `ContextRelevance` does the same when both judges exhaust their retries. A single call to `score_dataset` produces a single `report`. Per-row scores are written to `metrics_report.csv`, and those same numbers are averaged into `metrics_summary.csv`. There is no extra judge call between the two files.

The split by `synthesizer_name` is the main analytical device of this part. On a small corpus, single-hop questions are almost always answered well, and they mark the ceiling. Multi-hop questions require retrieving **two different** chunks, and with `top_k=3` the second chunk can easily miss the result list. A difference between configurations, if one exists, will show up there.

## Results of the run

The baseline is the configuration in `rag.py`: `top_k=3`, chunks of 1000 / 200, embeddings `text-embedding-3-small`, generator and judge both `gpt-5-mini`. Two of the four metrics in the set are enabled: `Faithfulness` and `AnswerCorrectness`. `ContextRelevance` and `AnswerRelevancy` are commented out.

The dataset from part one has 8 rows: 3 single-hop, 3 multi-hop specific, 2 multi-hop abstract. Each row takes 5 judge calls and 2 embedding requests, plus the RAG generation itself.

The means from this run are in `metrics_summary.csv`. The same scores, row by row, are in `metrics_report.csv`.

| scope | n | faithfulness | answer_correctness |
|-------|---|--------------|--------------------|
| all | 8 | 0.901 | 0.677 |
| `multi_hop_abstract_query_synthesizer` | 2 | 0.812 | 0.536 |
| `multi_hop_specific_query_synthesizer` | 3 | 0.899 | 0.633 |
| `single_hop_specific_query_synthesizer` | 3 | 0.963 | 0.815 |

Two things stand out at once.

**Faithfulness is high; AnswerCorrectness is noticeably lower.** The answers barely invent facts beyond the retrieved context (0.90), but they fall short of the reference (0.68). That is expected. The system prompt asks for a short answer grounded strictly in the context, while `reference` is an expanded paraphrase of the source chunks. The metric penalizes both omissions and surplus facts from neighboring chunks that the retriever returned and that the dataset generator left out of the reference.

**A ceiling on single-hop, a dip on multi-hop.** Single-hop: 0.96 / 0.82. Multi-hop specific holds its faithfulness (0.90), but correctness is already 0.63. Abstract is the worst of the three, especially on correctness (0.54). Exactly what the end of the previous section said: on eight rows, a difference between configurations, if one exists, lives in multi-hop.

## Judge noise, without which the conclusions mean nothing

This section matters in practice. An LLM judge is non-deterministic, so every metric carries its own noise, and **any difference between configurations smaller than that noise means nothing**.

To measure the noise, we ran `score_dataset` once more, separately, on the same `after_rag_dataset.csv`. The RAG answers were left as they were; only the judge invocation changed. This is not a pipeline step, and it is not the report/summary pair. Inside a single `main()` those two files come from one `report`. Here we have two independent scorings, run one after the other. Below are the means of the first and the second.

| scope | n | F₁ | F₂ | ΔF | AC₁ | AC₂ | ΔAC |
|-------|---|-----|-----|------|------|------|-------|
| all | 8 | 0.901 | 0.885 | −0.016 | 0.677 | 0.663 | −0.013 |
| abstract | 2 | 0.812 | 0.684 | −0.128 | 0.536 | 0.513 | −0.023 |
| specific | 3 | 0.899 | 0.970 | +0.071 | 0.633 | 0.634 | +0.001 |
| single-hop | 3 | 0.963 | 0.934 | −0.029 | 0.815 | 0.793 | −0.022 |

On the full dataset the metrics move within **0.02**. That is the threshold: a configuration delta below two hundredths, on eight rows, means nothing.

## Comparing configurations

We change exactly one thing: `top_k` from 3 to 5. Chunking, embeddings, prompt, and model stay as they were.

| scope | n | F @ k=3 | F @ k=5 | ΔF | AC @ k=3 | AC @ k=5 | ΔAC |
|-------|---|---------|---------|------|----------|----------|-------|
| all | 8 | 0.901 | 0.898 | −0.003 | 0.677 | 0.753 | **+0.076** |
| abstract | 2 | 0.812 | 0.825 | +0.013 | 0.536 | 0.679 | **+0.143** |
| specific | 3 | 0.899 | 0.903 | +0.004 | 0.633 | 0.736 | **+0.103** |
| single-hop | 3 | 0.963 | 0.941 | −0.022 | 0.815 | 0.819 | +0.004 |

Faithfulness on `all` barely moved (−0.003), which is inside judge noise. `AnswerCorrectness` rose by 0.076 — three times the noise on the full set. 
The conclusion that holds up: **on this corpus, `top_k=5` beats `top_k=3` end to end**, and the gain is not judge noise. 

## Pitfalls

**`evaluate()` does not work with the current metrics.** Metrics from `collections` do not inherit from `Metric`, and `evaluate()` rejects them with a `TypeError`. You write the runner yourself.

**`abatch_score` does not limit concurrency.** It is a bare `asyncio.gather` over every input. Past a hundred rows you need your own semaphore.

**`NaN` is an expected result, not a failure.** `Faithfulness` returns `NaN` when the LLM extracted no statements. `ContextRelevance` returns `NaN` when the judges exhaust their retries. Aggregate only with `skipna=True`.

**`AnswerCorrectness` is a composite, not a single measurement.** The default weights are `[0.75, 0.25]`, and the second quarter is a raw cosine that is almost never low.

**`ContextRelevance` has only five possible values.** Two judges on a 0/1/2 scale produce 0, 0.25, 0.5, 0.75, or 1. On a small dataset the metric will not show a small retrieval improvement. The improvement may be real; the metric does not have the resolution to show it.

**No metric in the set looks at context order.** Every one of them concatenates `retrieved_contexts` into a single string. If you add a reranker, this set will not measure it. You need `ContextPrecisionWithReference`, in which a context contributes more the higher it ranks in the retrieved list.

**`AnswerRelevancy` does not see the context, and it penalizes “I don't know.”** It measures how well the answer matches the question. A noncommittal answer scores zero, so a system prompt of the form “if you do not know, say so” lowers this metric while raising `Faithfulness`.

**No RAG metric reads `reference_contexts` from part one.** In the current API only `SummaryScore` and the rubrics consume this field. The deterministic metrics that compared retrieved contexts with reference contexts by string (`NonLLMContextPrecisionWithReference`, `NonLLMContextRecall`) and by identifier (`IDBasedContextPrecision`) remain in the deprecated API. For us, `reference_contexts` is a manual diagnostic: a way to check by eye whether the retriever found what the dataset generator treated as the source.

___

That closes the loop. Part one produced a dataset that describes the task and does not depend on the implementation. Part two produces the numbers by which configurations can be compared, and a sense of which difference in those numbers is real and which is noise.
