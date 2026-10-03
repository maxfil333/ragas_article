[[RAG. Теория]]

# Evaluation metrics

> Версия, на которой всё написано и проверено: `ragas 0.4.3`.
>
> Это вторая часть. В первой мы собрали `eval_dataset.csv` — синтетический датасет с вопросами и эталонными ответами. Здесь мы наконец получаем числа, по которым можно сравнивать конфигурации RAG.

## Что у нас есть и чего не хватает

Напомним, с чем мы вышли из первой части. Датасет даёт четыре содержательные колонки (остальные — метаданные генерации):

| колонка              | что внутри                                            |
|----------------------|-------------------------------------------------------|
| `user_input`         | сгенерированный вопрос                                |
| `reference`          | эталонный ответ                                       |
| `reference_contexts` | чанки, по которым этот вопрос был сгенерирован        |
| `synthesizer_name`   | тип вопроса: single-hop / multi-hop specific/abstract |

Чего в датасете нет и быть не может: **ответа вашей системы** (`response`) и **контекста, который достал ваш retriever** (`retrieved_contexts`). Датасет ничего не знает про вашу конфигурацию — он описывает только задачу. Значит первый шаг оценки это не метрики, а **прогон вашего RAG по всем `user_input`**.

## Как считаются метрики: схема целиком

Как и в первой части, сначала перечислим всё, что готовим сами:

```
eval_dataset.csv             из части 1: user_input, reference, reference_contexts, ...
rag(question)                ваш RAG пайплайн: возвращает response и retrieved_contexts
judge_llm, judge_embeddings  судья: llm_factory(...) и OpenAIEmbeddings(...)
metrics                      список метрик из ragas.metrics.collections
max_concurrency              сколько запросов к судье держать одновременно
```

Нумерацию блоков продолжаем с первой части, где закончили на `[4]`:

```
eval_dataset.csv
   │
   ▼
[5] run_rag(eval_dataset.csv, after_rag_dataset.csv)
       если after_rag_dataset.csv нет — для каждой строки вызываем свой RAG
       и дописываем две колонки: response и retrieved_contexts
       если файл уже есть, читаем его и RAG повторно не гоняем
       это единственный шаг, который зависит от вашей конфигурации
   │
   │  rows: список dict с полным набором полей для метрик
   ▼
[6] metrics = build_metrics(judge_llm, judge_embeddings)
       каждая метрика объявляет в сигнатуре ascore(), какие поля ей нужны
   │
   │  metrics: список объектов BaseMetric
   ▼
[7] scores = await score_dataset(rows, metrics, max_concurrency)
       свой раннер: для каждой пары (строка, метрика) вызываем
       metric.ascore(**поля) под семафором
       ragas готового раннера для этих метрик не даёт
   │
   │  scores: таблица строки × метрики
   ▼
[8] агрегация
       среднее по колонкам через nanmean, разрез по synthesizer_name,
       и обязательно — оценка шума повторным прогоном
   │
   ▼
metrics_report.csv:  user_input | context_relevance | faithfulness | ... | synthesizer_name
metrics_summary.csv: scope | n | faithfulness | ...   (all + разрез по synthesizer_name)
```

> Ниже в блоках кода с комментарием-путём (например `# ragas/metrics/base.py`) показаны **внутренности библиотеки**. Их не нужно писать у себя — они приведены, чтобы было видно, откуда берутся ограничения, под которые мы подстраиваемся.

## Две системы метрик, и почему мы берём вторую

В 0.4.3 метрики живут в двух местах одновременно, стоит их различать.

`ragas.metrics` — первая система. Работает с готовой функцией `evaluate()`, но при импорте честно предупреждает:

```
DeprecationWarning: Importing Faithfulness from 'ragas.metrics' is deprecated
and will be removed in v1.0. Please use 'ragas.metrics.collections' instead.
```

`ragas.metrics.collections` — вторая, актуальная. Её мы и используем дальше. 

Метрики из `collections` наследуются от `SimpleBaseMetric`, а не от `Metric`.   
`evaluate()` мы не используем, а раннер пишем сами. Это не так страшно, как звучит — раннер занимает двадцать строк, и в разделе `[7]` мы его разберём. Зато по пути исчезают `EvaluationDataset` и `SingleTurnSample`: они нужны только `evaluate()`, а `collections` принимают обычные словари.

## Карта метрик: что чем измеряется

Метрики RAG удобно разложить по двум осям сразу: **какой слой** они проверяют и **нужен ли им эталон**. Вторая определяет, можете ли вы считать метрику по логам прода или обязаны иметь датасет из первой части.

| Метрика                            | входы `ascore()`                                            | слой        | нужен `reference` |
|------------------------------------|-------------------------------------------------------------|-------------|-------------------|
| `ContextRelevance`                 | `user_input`, `retrieved_contexts`                          | retrieval   | нет               |
| `ContextPrecisionWithoutReference` | `user_input`, `response`, `retrieved_contexts`              | retrieval   | нет               |
| `ContextPrecisionWithReference`    | `user_input`, `reference`, `retrieved_contexts`             | retrieval   | да                |
| `ContextRecall`                    | `user_input`, `retrieved_contexts`, `reference`             | retrieval   | да                |
| `ContextEntityRecall`              | `reference`, `retrieved_contexts`                           | retrieval   | да                |
| `Faithfulness`                     | `user_input`, `response`, `retrieved_contexts`              | generation  | нет               |
| `ResponseGroundedness`             | `response`, `retrieved_contexts`                            | generation  | нет               |
| `AnswerRelevancy`                  | `user_input`, `response` (+ эмбеддинги)                     | generation  | нет               |
| `AnswerCorrectness`                | `user_input`, `response`, `reference` (+ эмбеддинги)        | end-to-end  | да                |
| `FactualCorrectness`               | `response`, `reference`                                     | end-to-end  | да                |
| `NoiseSensitivity`                 | `user_input`, `response`, `reference`, `retrieved_contexts` | диагностика | да                |


_Колонка «входы» — это сигнатура метода `ascore()`. Сигнатура не всегда совпадает с интуицией и `Faithfulness` — как раз такой случай. По смыслу метрика сравнивает ответ с контекстом, и вопрос ей вроде бы не нужен, но в реализации он **обязателен**: при пустом `user_input` метрика бросает `ValueError`, а сам вопрос идёт в промпт первого шага, где ответ разбивается на утверждения. Практическое следствие: посчитать `Faithfulness` по выгрузке, в которой сохранён только ответ и контекст, не получится. И наоборот, `ResponseGroundedness` меряет почти то же самое, но вопрос не требует._

Отсюда практический набор, который мы возьмём дальше. По одной метрике на каждый вопрос, который мы хотим задать системе:

- **Насколько найденный контекст относится к вопросу пользователя?** → `ContextRelevance`. Единственная в наборе, которая не смотрит на ответ вообще.
- **Правда ли ответ следует из найденного контекста?** → `Faithfulness`.
- **Получен ли ответ на вопрос пользователя?** → `AnswerRelevancy`.
- **Насколько ответ совпадает с эталонным?** → `AnswerCorrectness`. Итоговая end-to-end метрика.

## Механика: что происходит внутри одного score

Рассмотрим на примере, как происходит расчет метрик.

### ContextRelevance

_Оценивает, насколько найденный и переданный LLM контекст (`retrieved_contexts`) **релевантен исходному запросу пользователя** (`user_input`). Показывает, удалось ли retrieval-компоненту предоставить модели действительно полезную информацию, а не большое количество нерелевантных документов._

Устроена не так, как остальные три, и это стоит разобрать первым. Здесь нет никакого разбиения на утверждения: все `retrieved_contexts` склеиваются в одну строку, и её релевантность вопросу оценивают **два независимых судьи** — два разных промпта. Каждый выставляет целое число: 0, 1 или 2. Оценки делятся на 2 и усредняются:

```python
# ragas/metrics/collections/context_relevance/metric.py — внутренности библиотеки
judge1_rating = await self._get_judge_rating(self.judge1_prompt, user_input, context_str)
judge2_rating = await self._get_judge_rating(self.judge2_prompt, user_input, context_str)

score = self._average_scores(judge1_rating / 2.0, judge2_rating / 2.0)
```

Отсюда её главное свойство: скор может принимать всего **пять** значений — 0, 0.25, 0.5, 0.75 и 1. Это грубая шкала, и её надо учитывать при интерпретации. Метрика хорошо отвечает на вопрос «retriever достаёт по теме или приносит мусор», но плохо ловит небольшие улучшения.

Две детали из кода. Если судья вернул оценку вне `[0, 1, 2]`, метрика повторяет попытку до `max_retries=5` раз, а исчерпав их — отдаёт `NaN`. И есть несколько вырожденных случаев, где метрика возвращает 0 не глядя: пустой контекст, контекст, совпадающий с вопросом, или контекст, целиком содержащийся в тексте вопроса.

### Faithfulness

_Оценивает насколько ответ (`response`) следует из найденного контекста (`retrieved_contexts`). Ответ считается достоверным, если все его утверждения подтверждаются полученным контекстом._

Два вызова LLM на строку:

1. ответ разбивается на атомарные утверждения;
2. все `retrieved_contexts` склеиваются в **одну** строку, и каждое утверждение проверяется против неё на NLI-манер: поддержано или нет.
3. итог: Faithfulness = (число подтверждённых утверждений) / (общее число утверждений).

### AnswerRelevancy

_Получен ли ответ (`response`) на вопрос пользователя (`user_input`)?_

Три вызова LLM (параметр `strictness=3`) плюс два запроса к эмбеддингам. Логика обратная интуитивной: по **ответу** генерируется вопрос, который этот ответ закрывал бы, — и так три раза. Затем считается косинус между исходным вопросом и тремя сгенерированными, берётся среднее.

Два момента. Во-первых, метрика **вообще не смотрит на контекст**: она про «ответили ли на вопрос», а не про «правда ли это». Во-вторых, есть отдельный флаг `noncommittal`: если все три прогона сочли ответ уклончивым («в контексте нет информации»), скор умножается на ноль. То есть честное «не знаю» тут штрафуется в пол — это надо держать в голове, сравнивая конфигурации с разными системными промптами.

### AnswerCorrectness

_Насколько ответ (`response`) совпадает с эталонным ответом (`reference`)?_

Требует ground truth, комбинирует две независимые подметрики взвешенной суммой:

- Factual correctness — F-мера по перекрытию утверждений: генерируемый ответ и ground truth разбиваются на утверждения, которые классифицируются как `TP` (есть в обоих), `FP` (есть только в ответе), `FN` (есть только в ground truth), затем `F1 = TP / (TP + 0.5×(FP + FN))`. 
- Semantic similarity — косинусное сходство эмбеддингов полного текста ответа и ground truth (переиспользуется отдельная метрика answer_similarity). 
- Финал: взвешенное среднее семантического сходства и факт-скора, по умолчанию веса [0.75, 0.25] — то есть факт-совпадение (F1) весит втрое больше, чем просто семантическая близость текста. 

### Сколько это стоит

Сложим вызовы LLM на одну строку датасета для нашего набора:

| Метрика             | вызовов LLM | запросов к эмбеддингам |
|---------------------|-------------|------------------------|
| `ContextRelevance`  | 2           | —                      |
| `Faithfulness`      | 2           | —                      |
| `AnswerRelevancy`   | 3           | 2                      |
| `AnswerCorrectness` | 3           | 2                      |
| **итого на строку** | **10**      | **4**                  |

Плюс сам прогон RAG: один вызов генерации и один запрос к эмбеддингам на строку. Итого на строку **11** вызовов LLM (10 у метрик + 1 у генератора). На датасете из первой части (8 строк) это `8 × 11 = 88` вызовов LLM. Цифра маленькая только потому, что датасет крошечный: на тысяче строк тот же набор — уже `1000 × 11 = 11` тысяч вызовов, и вопрос «какие метрики мне действительно нужны» становится финансовым. (Восемь строк — демонстрация метода, а не выбор продакшен-конфига.)

## Собираем оценку

Дальше — наш код. Полный файл: `pipeline_calculate_metrics.py`.

### 1. Судья

```python
openai_client = AsyncOpenAI(api_key=os.getenv("YOUR_API_KEY"))
judge_llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
judge_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

Тот же `llm_factory`, что и в первой части, и это не случайно: метрики из `collections` принимают только такую обёртку.

Отдельно стоит сказать про выбор модели судьи. У нас и ответы генерирует `gpt-5-mini`, и оценивает их тоже `gpt-5-mini` — это самый дешёвый вариант и самый уязвимый: модели склонны выше оценивать собственные формулировки. Если решение по конфигурации дорогое, судью стоит брать сильнее генератора и хотя бы раз сверить его оценки с ручной разметкой.

### 2. Прогон RAG по датасету ([5])

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

Если `after_rag_dataset.csv` уже лежит рядом — этот шаг пропускаем и читаем готовый файл. Иначе прогоняем RAG по всем `user_input` и сохраняем результат: датасет описывает задачу, а `response` и `retrieved_contexts` дописывает ваша система. Всё, что ниже, от конфигурации RAG уже не зависит.

### 3. Набор метрик ([6])

```python
def build_metrics(llm, embeddings) -> list[BaseMetric]:
    return [
        # ContextRelevance(llm=llm),
        Faithfulness(llm=llm),
        # AnswerRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
```

`ContextRelevance` и `AnswerRelevancy` пока выключены: на строку это минус 5 вызовов LLM и 2 запроса к эмбеддингам. Когда понадобится полный набор — достаточно раскомментировать.

Метрики, которым нужны только вердикты LLM, собираются одним аргументом; `AnswerRelevancy` и `AnswerCorrectness` дополнительно считают косинусы, поэтому им передаются эмбеддинги.

### 4. Свой раннер ([7])

Ragas даёт метрикам метод `abatch_score`, и на первый взгляд он решает задачу. Но вот что у него внутри:

```python
# ragas/metrics/base.py — внутренности библиотеки
async_tasks = []
for input_dict in inputs:
    async_tasks.append(self.ascore(**input_dict))

return await asyncio.gather(*async_tasks)
```

Это голый `gather` без ограничения конкурентности. На восьми строках всё пройдёт, а на тысяче вы отправите тысячу одновременных запросов и получите от провайдера отказ. Поэтому раннер свой, и в нём есть ровно три вещи, которых нет в библиотечном: семафор, устойчивость к падению отдельного скора и определение нужных полей.

```python
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
```

Функция `required_fields` заслуживает отдельного слова. Мы не держим у себя таблицу «какой метрике какие поля» — метрика объявляет это сама в сигнатуре `ascore()`, и мы просто её читаем. Благодаря этому добавление новой метрики в список не требует правок в раннере, а опечатка в имени поля обнаруживается сразу, как `KeyError`, а не как тихий `NaN`.

Ловить исключение на уровне одного скора тоже осознанно: у LLM-метрик нормально, когда одна строка из сотни падает на невалидном structured output. Без `try` весь `gather` отменится и вы потеряете результаты всего прогона.

### 5. Агрегация ([8])

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

`skipna=True` здесь не декоративный: `Faithfulness` штатно возвращает `NaN`, когда из ответа не выделилось ни одного утверждения, а `ContextRelevance` — когда оба судьи исчерпали повторы. Один вызов `score_dataset` даёт один `report`: построчные скоры пишем в `metrics_report.csv`, те же числа усредняем в `metrics_summary.csv`. Отдельного вызова судьи между файлами нет.

Разрез по `synthesizer_name` — главный аналитический приём этой части. Single-hop вопросы на маленьком корпусе почти всегда отвечаются хорошо и показывают потолок, а вот multi-hop требуют достать **два разных** чанка, и при `top_k=3` второй вполне может не попасть в выдачу. Разница между конфигурациями, если она есть, проявится именно там.

## Результаты прогона

Базовая конфигурация — та, что в `rag.py`: `top_k=3`, чанки 1000 / 200, эмбеддинги `text-embedding-3-small`, генератор и судья — `gpt-5-mini`. Из четырёх метрик набора включены две: `Faithfulness` и `AnswerCorrectness`. `ContextRelevance` и `AnswerRelevancy` закомментированы.

Датасет из первой части: 8 строк — 3 single-hop, 3 multi-hop specific, 2 multi-hop abstract. На строку уходит 5 вызовов судьи и 2 запроса к эмбеддингам плюс сама генерация RAG.

Средние этого прогона — `metrics_summary.csv`. Построчно те же скоры лежат в `metrics_report.csv`.

| scope | n | faithfulness | answer_correctness |
|-------|---|--------------|--------------------|
| all | 8 | 0.901 | 0.677 |
| `multi_hop_abstract_query_synthesizer` | 2 | 0.812 | 0.536 |
| `multi_hop_specific_query_synthesizer` | 3 | 0.899 | 0.633 |
| `single_hop_specific_query_synthesizer` | 3 | 0.963 | 0.815 |

Две вещи видны сразу.

**Faithfulness высокий, AnswerCorrectness заметно ниже.** Ответы почти не выдумывают фактов сверх найденного контекста (0.90), но до эталона недотягивают (0.68). Это ожидаемо: системный промпт просит краткий ответ строго по контексту, а `reference` — развёрнутый пересказ исходных чанков. Метрика режет и пропуски, и лишние факты из соседних чанков, которые retriever принёс, а генератор датасета в эталон не клал.

**Потолок на single-hop, просадка на multi-hop.** Single-hop: 0.96 / 0.82. Multi-hop specific держит faithfulness (0.90), но correctness уже 0.63. Abstract — хуже всех, особенно correctness (0.54). Ровно то, что обещали в конце предыдущего раздела: на восьми строках разница конфигураций, если она есть, живёт в multi-hop.

## Шум судьи, без которого выводы бессмысленны

Важный раздел с практической точки зрения. LLM-судья недетерминирован, поэтому у каждой метрики есть собственный шум, и **любая разница между конфигурациями меньше этого шума не означает ничего**.

Чтобы измерить шум, мы отдельно прогнали `score_dataset` ещё раз на том же `after_rag_dataset.csv`: ответы RAG не трогали, менялся только вызов судьи. Это не шаг пайплайна и не пара файлов report/summary — внутри одного `main()` они из одного `report`. Здесь два независимых скоринга подряд. Ниже — средние первого и второго.

| scope | n | F₁ | F₂ | ΔF | AC₁ | AC₂ | ΔAC |
|-------|---|-----|-----|------|------|------|-------|
| all | 8 | 0.901 | 0.885 | −0.016 | 0.677 | 0.663 | −0.013 |
| abstract | 2 | 0.812 | 0.684 | −0.128 | 0.536 | 0.513 | −0.023 |
| specific | 3 | 0.899 | 0.970 | +0.071 | 0.633 | 0.634 | +0.001 |
| single-hop | 3 | 0.963 | 0.934 | −0.029 | 0.815 | 0.793 | −0.022 |

На полном датасете метрики меняются в пределах **0.02**. Это и есть порог: дельта конфигурации меньше двух сотых на восьми строках ничего не значит.


## Сравнение конфигураций

Меняем ровно одну вещь: `top_k` 3 → 5. Чанкинг, эмбеддинги, промпт, модель — как были.

| scope | n | F @ k=3 | F @ k=5 | ΔF | AC @ k=3 | AC @ k=5 | ΔAC |
|-------|---|---------|---------|------|----------|----------|-------|
| all | 8 | 0.901 | 0.898 | −0.003 | 0.677 | 0.753 | **+0.076** |
| abstract | 2 | 0.812 | 0.825 | +0.013 | 0.536 | 0.679 | **+0.143** |
| specific | 3 | 0.899 | 0.903 | +0.004 | 0.633 | 0.736 | **+0.103** |
| single-hop | 3 | 0.963 | 0.941 | −0.022 | 0.815 | 0.819 | +0.004 |

Faithfulness по `all` почти не сдвинулся (−0.003) — внутри шума судьи. `AnswerCorrectness` вырос на 0.076 — втрое выше шума на полном наборе. 
Вывод, который можно защищать: **на этом корпусе `top_k=5` лучше `top_k=3` по end-to-end**, и это не шум судьи.

## Ловушки

**`evaluate()` не работает с актуальными метриками.** Метрики из `collections` не наследуются от `Metric`, и `evaluate()` отвергает их с `TypeError`. Раннер пишется руками.

**`abatch_score` не ограничивает конкурентность.** Голый `asyncio.gather` по всем входам. На датасете больше сотни строк нужен свой семафор.

**`NaN` — это штатный результат, а не сбой.** `Faithfulness` возвращает `NaN`, когда LLM не выделила ни одного утверждения, `ContextRelevance` — когда судьи исчерпали повторы. Агрегировать только через `skipna=True`.

**`AnswerCorrectness` — композит, а не измерение.** Веса по умолчанию `[0.75, 0.25]`, и вторая четверть это сырой косинус, который почти никогда не бывает низким. 

**У `ContextRelevance` всего пять возможных значений.** Два судьи со шкалой 0/1/2 дают 0, 0.25, 0.5, 0.75 или 1. На маленьком датасете метрика не покажет небольшое улучшение retrieval — не потому, что улучшения нет, а потому, что у неё нет разрешения.

**Ни одна метрика набора не смотрит на порядок контекстов.** Все склеивают `retrieved_contexts` в одну строку. Если вы вводите reranker, этим набором вы его не измерите: нужна `ContextPrecisionWithReference`, у которой вклад контекста тем выше, чем выше он стоит в выдаче.

**`AnswerRelevancy` не видит контекста и штрафует «не знаю».** Она про соответствие ответа вопросу. Уклончивый ответ получает ноль, поэтому системный промпт «если не знаешь, так и скажи» ухудшает эту метрику, одновременно улучшая `Faithfulness`.

**`reference_contexts` из первой части не читает ни одна RAG-метрика.** В актуальном API это поле берут только `SummaryScore` и рубрики. Детерминированные метрики, которые сравнивали найденные контексты с эталонными по строкам (`NonLLMContextPrecisionWithReference`, `NonLLMContextRecall`) и по идентификаторам (`IDBasedContextPrecision`), остались в устаревшем API. Поэтому `reference_contexts` у нас служит инструментом ручной диагностики: глазами сверить, нашёл ли retriever то, что генератор датасета считал источником.

___

На этом цикл закрывается. В первой части мы получили датасет, который описывает задачу и не зависит от реализации; во второй — числа, по которым конфигурации можно сравнивать, и понимание того, какая разница в этих числах реальна, а какая является шумом.
