# Введение

Задача:
> У нас есть RAG. Мы меняем embeddings / vector store / reranker / chunking и хотим объективно понимать, какая конфигурация лучше.

Для решения данной задачи будем использовать библиотеку Ragas.
Ragas в данном случает фундаментально выполняет две функции:

- Генерирует evaluation dataset
- Оценивает RAG-пайплайн (metrics)

Генерация датасета и оценка пайплайна — два полностью независимых процесса. Каждый из них выполняет свою роль в решении исходной задачи по выбору лучшей конфигурации.

> Версия, на которой всё написано и проверено: `ragas 0.4.3`.

# Evaluation dataset

## Описание проблемы

Классический RAG пайплайн в упрощенном виде принимает на вход вопрос пользователя 
(`user_input`), а на выходе возвращает ответ (`response`),  на основе контекста 
(`retrieved_contexts`), полученного в результате retrieval по базе знаний.  

Хорошая новость в том, что на основе уже этих трех сущностей (`user_input, response, retrieved_contexts`) мы можем оценить некоторые метрики, например **Faithfulness** и **Answer Relevancy** (подробнее о метриках будет ниже). Эти данные можно получить обычной выгрузкой логов с работающего агента.

При этом некоторые метрики требуют дополнительных данных. Например, для оценки **Context Precision** и **Answer Correctness** нужна эталонная информация, то есть "идеальный ответ" (`reference`). Для получения этих данных с эталонной информацией можно воспользоваться следующими подходами:

**1) Генерация датасета при помощи экспертов**

Берем документацию, вопросы с production, поручаем эксперту дать на них ответы.
Данный подход имеет высокое качество, однако является долгим, дорогим и неавтоматизируемым.

**2) Генерация датасета с помощью LLM**

LLM для каждого документа или чанка генерирует вопрос (`user_input`) и эталонный ответ 
(`reference`). Использование этого подхода в сыром виде приведет к тому, что датасет получится слишком искусственным, так как реальный пользователь часто может не знать термины, ошибаться, задавать слишком короткие вопросы, иметь разный уровень знаний, а для ответа на некоторые вопросы может потребоваться информация сразу из нескольких документов.

Для решения описанных выше проблем и генерации более реалистичного датасета можно применить ragas.

## Как ragas генерирует датасет: схема целиком

Сначала — что мы готовим сами, до всякого ragas-пайплайна. Это ровно шесть вещей.

```
docs                 list[str] — тексты документов
llm, embeddings      клиенты моделей: llm_factory(...) и OpenAIEmbeddings(...)
transforms           список шагов обогащения графа        → нужен шагу [2]
personas             список Persona: от чьего лица спрашивать → нужен шагу [3]
query_distribution   какие типы вопросов и в какой пропорции  → нужен шагу [3]
testset_size         сколько строк хотим получить            → нужен шагу [3]
```

Теперь сам маршрут. Он линейный: слева от отступа — что мы вызываем, на стрелках — что получаем на выходе.

```
docs
   │
   ▼
[1] kg = build_knowledge_graph(docs)
       создаём KnowledgeGraph: по одному узлу DOCUMENT на документ,
       единственное свойство узла — page_content
   │
   │  kg: граф из документов, ещё без чанков и без ребер
   ▼
[2] apply_transforms(kg, transforms)
       прогоняем конвейер трансформаций графа из 4 типов шагов (порядок задаём мы):
         Splitters   →  режут DOCUMENT на CHUNK
         Extractors  →  добавляют свойства узлам (headlines, keyphrases, summary, ...)
         Relations   →  строят ребра между узлами
         Filters     →  удаляют бесполезные чанки
       граф меняется in place: это всё тот же объект kg
   │
   │  kg: чанки + свойства + ребра
   ▼
generator = TestsetGenerator(llm, embeddings, kg, persona_list=personas)
testset = generator.generate(testset_size, query_distribution)
   │   один вызов generate(), внутри которого две фазы подряд:
   │
   ├─ [3] ОТБОР СЦЕНАРИЕВ — вопросов ещё нет.
   │      query_distribution задаёт, сколько сценариев какого типа набрать,
   │      personas — от чьего лица они будут заданы,
   │      testset_size — сколько их всего.
   │      Сценарий это «условия задачи»: какой узел (или пара узлов),
   │      про какую тему, от какого персонажа, каким стилем, какой длины.
   │      В консоли: прогресс-бар "Generating Scenarios"
   │
   └─ [4] ГЕНЕРАЦИЯ ВОПРОСОВ — здесь LLM пишет текст.
          На каждый сценарий один запрос к LLM, который возвращает
          вопрос (user_input) и эталонный ответ (reference) по нему.
          Один сценарий = один сэмпл = одна строка будущего датасета.
          В консоли: прогресс-бар "Generating Samples"
   │
   │  testset: объект Testset
   ▼
testset.to_pandas().to_csv("eval_dataset.csv")
   │
   ▼
eval_dataset.csv:  user_input | reference_contexts | reference | ...
```

Тот же маршрут кодом (схематично) — сначала готовим все шесть входов, потом пять строк самого пайплайна:

```python
# готовим входы
openai_client = AsyncOpenAI(api_key=..., base_url=...)
docs = load_hf_documents()
llm = llm_factory("gpt-5-mini", client=openai_client)
embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
transforms = build_transforms(llm, embeddings)
personas = [Persona(name="student", role_description="curious university student")]
query_distribution = build_query_distribution(llm)
testset_size = 6

# маршрут
kg = build_knowledge_graph(docs)                                        # [1]
apply_transforms(kg, transforms=transforms)                             # [2]
generator = TestsetGenerator(llm, embeddings, kg, persona_list=personas)
testset = generator.generate(testset_size, query_distribution)          # [3] + [4]
testset.to_pandas().to_csv("eval_dataset.csv")
```

Каждую из функций `build_*` мы напишем сами — они разбираются по шагам в разделе «Собираем пайплайн».

Зачем вообще граф, а не просто список чанков? Список чанков даёт только вопросы «по одному фрагменту». Ребра позволяют взять **пару связанных чанков** и попросить вопрос, ответ на который лежит сразу в двух местах — это и есть multi-hop.

Дальше разбираем блоки по порядку:

- **[1] и [2]** — из чего состоит `KnowledgeGraph` и какие бывают трансформации (Splitters / Extractors / Relations / Filters);
- **[3] и [4]** — персонажи, стили и длина вопросов, single-hop vs multi-hop, и обе фазы генерации;
- после этого — рабочий код целиком в разделе «Собираем пайплайн».

> Ниже в блоках кода с комментарием-путём (например `# ragas/testset/graph.py`) показаны **внутренности библиотеки**. Их не нужно писать у себя — они приведены только чтобы было видно, откуда берутся имена свойств и типов, на которые мы опираемся в конфигурации.

### [1] KnowledgeGraph

KnowledgeGraph - базовая структура данных, хранящая корпус документов в виде графа. Узлами (nodes) являются фрагменты текста (чанки), обогащенные выделенными сущностями, ключевыми словами, заголовками и прочей вспомогательной информацией. 
Ребрами (edges) — семантические или логические связи между ними (например, «относится к», «использует», «следствие из»).

Структура — это два списка:

```python
# ragas/testset/graph.py — внутренности библиотеки, реализовывать не нужно
class KnowledgeGraph:
    nodes: list[Node]                    # Node(type: NodeType, properties: dict)
    relationships: list[Relationship]    # Relationship(type: str, source, target, properties)
```

_Граф можно сохранить и переиспользовать: `kg.save("kg.json")` / `KnowledgeGraph.load("kg.json")`. Построение графа — самая дорогая часть (много вызовов LLM), поэтому при экспериментах граф строят один раз, а потом гоняют по нему разные `query_distribution`._

### [2] transforms

Конвейер (пайплайн) обработки данных, который обогощает информацией `KnowledgeGraph`. Трансформации прогоняют тексты через LLM и эмбеддинги.

`transforms` — это просто список, и порядок в нём и есть логика построения графа. Типов трансформаций четыре (Splitters, Extractors, Relationship builders, Filters), и у всех есть общий параметр `filter_nodes: Callable[[Node], bool]` — на каких узлах работать (по умолчанию на всех).

#### Splitters

> *создают/разбивают Nodes*

- Разбивают исходный массивный текст на базовые узлы (`Node` / чанки).
- **Пример:** **`HeadlineSplitter`** нарезает документ не по фиксированному количеству символов, а по логическим заголовкам (H1, H2, H3), сохраняя контекстную целостность разделов (у Nodes уже должны быть свойства `headlines` и `page_content`)


_**Важная и неочевидная особенность:**_  
_Вопреки описанному выше порядку (сначала Splitters, затем Extractors),_  
_мы обязаны **сначала** извлечь заголовки (HeadlinesExtractor)_  
_и только **затем** делить на чанки по этим заголовкам (HeadlineSplitter)._  
_Иначе HeadlineSplitter пытается прочитать несуществующий `headlines`_  
_и выдает ошибку `ValueError: 'headlines' property not found in this node`._


Что делает `HeadlineSplitter(min_tokens=300, max_tokens=1000)`:

- режет текст по позициям заголовков из `headlines`; слишком длинные секции дорезает по словам, слишком короткие **склеивает с соседними**, чтобы не появлялись чанки-огрызки;
- если документ целиком короче `min_tokens` или заголовки не нашлись — возвращает исходный узел без изменений;
- кроме узлов создаёт ребра `child` (document → chunk) и `next` (chunk → chunk).

#### Extractors

> *добавляют свойства в Node*

| Extractor                   | property_name       | что кладёт                             | лимит по умолчанию    |
|-----------------------------|---------------------|----------------------------------------|-----------------------|
| `HeadlinesExtractor`        | `headlines`         | список заголовков                      | `max_num=5`           |
| `KeyphrasesExtractor`       | `keyphrases`        | ключевые фразы                         | `max_num=5`           |
| `NERExtractor`              | `entities`          | именованные сущности                   | `max_num_entities=10` |
| `ThemesExtractor`           | `themes`            | темы и концепции                       | `max_num_themes=10`   |
| `SummaryExtractor`          | `summary`           | краткое содержание (до 10 предложений) | —                     |
| `TitleExtractor`            | `title`             | заголовок документа                    | —                     |
| `TopicDescriptionExtractor` | `topic_description` | описание основной темы                 | —                     |
| `EmbeddingExtractor`        | `embedding`         | вектор текста из `embed_property_name` | —                     |

Все, кроме `EmbeddingExtractor`, — это LLM-вызовы со строгим structured output (Pydantic-модель + instructor), поэтому результат всегда типизирован. Проверить экстрактор в отрыве от пайплайна можно так (это иллюстрация для консоли, в пайплайне вызывать вручную не нужно):

```python
extractor = NERExtractor()
await extractor.extract(node)

> ('entities', ['Einstein', 'theory of relativity', 'space', 'time' ...])
```

У `EmbeddingExtractor` два параметра, которые легко перепутать: `embed_property_name` — **что** векторизуем (`page_content`, `summary`), а `property_name` — **куда** кладём результат (`embedding`, `summary_embedding`). Отсюда стандартная связка «эмбеддинг саммари документа»: `EmbeddingExtractor(embed_property_name="summary", property_name="summary_embedding")`.

#### Relationship builders

> *создают ребра между Nodes*

Важная часть в построении графа знаний, обеспечивающая **связи между узлами**. Без Relations ноды графа будут обогащены после трансформаций (иметь entities, title и т.д.), но ноды не будут связаны друг с другом, т.е. получится граф без ребер.

Builder читает **одно свойство узла** - `property_name`. Значение должно совпадать с тем, что положил экстрактор на предыдущем шаге. В нашем пайплайне цепочка такая:

```python
# экстрактор кладёт в каждый чанк свойство "keyphrases"
KeyphrasesExtractor(llm=llm, property_name="keyphrases", filter_nodes=_is_chunk)
# → node.properties["keyphrases"] = ["International Atomic Time", "atomic clock", "UTC", ...]

# builder читает ровно это же свойство "keyphrases"
OverlapScoreBuilder(property_name="keyphrases", new_property_name="overlap_score",
                    distance_threshold=0.9, threshold=0.01, filter_nodes=_is_chunk)
```

Дальше builder берёт **все пары чанков** и для каждой пары считает, насколько сильно у них пересекаются ключевые фразы. Буквально по шагам, на примере двух чанков:

```
чанк A  keyphrases: ["International Atomic Time", "atomic clock", "leap second", "UTC", "geoid"]
чанк B  keyphrases: ["atomic clocks", "caesium", "UTC", "Circular T", "BIPM"]
```

1. Сравнивает каждую фразу A с каждой фразой B — здесь 5 × 5 = **25 сравнений**.
2. Пару считает совпавшей, если строки похожи не меньше чем на `distance_threshold=0.9`. Совпадут две: `atomic clock` / `atomic clocks` (≈ 0.98) и `UTC` / `UTC` (1.0).
3. Считает score = совпавшие пары / все сравнения = 2 / 25 = **0.08**.
4. Сравнивает score с `threshold=0.01`. Больше порога — создаёт ребро. Меньше — эти два чанка остаются несвязанными.

Ребро для нашей пары получится такое:

```python
Relationship(
    source=<чанк A>,
    target=<чанк B>,
    type="keyphrases_overlap",                   # имя связи
    properties={
        "keyphrases_overlap_score": 0.08,        # результат шага 3
        "overlapped_items": [                    # что именно совпало на шаге 2
            ("atomic clock", "atomic clocks"),
            ("UTC", "UTC"),
        ],
    },
)
```

Оба поля потом читает синтезатор multi-hop вопросов, но для разного:

- **`type`** — чтобы **найти** нужные рёбра: синтезатор берёт только те, у которых `type` совпал с его параметром `relation_type`. Не совпало — рёбер для него нет, и генерация падает с `No clusters found in the knowledge graph`.
- **`properties["overlapped_items"]`** — чтобы понять, **о чём спрашивать**: совпавшие фразы становятся темой вопроса. Поэтому multi-hop вопрос получается осмысленным, а не «расскажи про два случайных текста».

Отсюда два требования к ребру: его `type` должен совпадать с `relation_type` синтезатора, и оно обязано нести `overlapped_items`. Косинусные рёбра второму требованию не удовлетворяют, так что multi-hop specific по ним не работает.

Остальные builder-ы устроены так же, отличается только способ сравнения. В колонке — значение `property_name` по умолчанию, то есть какое свойство узла builder возьмёт, если не указать своё:

| Builder                    | `property_name` по умолчанию | как сравнивает                | `type` ребра                  |
|----------------------------|------------------------------|-------------------------------|-------------------------------|
| `CosineSimilarityBuilder`  | `embedding`                  | косинус между векторами       | `new_property_name`           |
| `JaccardSimilarityBuilder` | `entities`                   | Жаккар по множествам строк    | `new_property_name`           |
| `OverlapScoreBuilder`      | `entities`                   | нечёткое сравнение строк      | **`{property_name}_overlap`** |

Последняя строка — **главная ловушка** ragas: имя типа формируется не так, как у остальных. Косинусный builder берёт его прямо из `new_property_name`, а `OverlapScoreBuilder` `new_property_name` в типе игнорирует и склеивает имя из `property_name`. Поэтому при `property_name="keyphrases"` тип ребра — `keyphrases_overlap`, и ровно эту строку надо передать синтезатору.

**Ещё деталь `OverlapScoreBuilder`:**

- перед перебором builder выбрасывает топ-5% самых частотных фраз по всему корпусу — иначе граф превратился бы в клику из-за слова, которое есть в каждом чанке.

**Про `threshold` в `CosineSimilarityBuilder`** — это обычный порог косинуса, но подобрать его «по интуиции» не получается: абсолютные значения зависят и от модели эмбеддингов, и от того, насколько однороден корпус. Дефолт самого класса — `0.9`, в `default_transforms` библиотеки для длинных документов стоит `0.7`.

На наших трёх статьях косинусы между эмбеддингами их саммари оказались такими:

```
International Atomic Time  <->  Agricultural science : 0.132
International Atomic Time  <->  Arithmetic mean      : 0.257
Agricultural science       <->  Arithmetic mean      : 0.202
```

То есть с библиотечным порогом `0.7` не возникло бы ни одного ребра, и синтезатор, который по ним ходит, упал бы с `No relationships match the provided condition`. Практический вывод простой: пороги нужно мерить на своём корпусе, а не переносить из примеров. Мы поставили `0.25` — так остаётся единственное самое осмысленное ребро (среднее арифметическое ↔ атомное время, которое буквально считается как взвешенное среднее по часам), а случайные пары вроде «сельское хозяйство ↔ атомное время» не связываются.

#### Filters

> *удаляют Nodes*

- `CustomNodeFilter` — LLM-фильтр «пригодности чанка для вопроса». Для каждого чанка он берёт `summary` **родительского** документа (через ребро `child`), просит LLM оценить чанк по рубрике 1–5 («насколько содержимое соответствует теме документа и есть ли о чём спрашивать») и удаляет узел при score ≤ `min_score=2`.
- Отсюда важное следствие: фильтр **работает только если у документа есть `summary`**. Если `SummaryExtractor` в пайплайне нет, фильтр просто напишет warning и ничего не отфильтрует.

#### filter_nodes и порядок трансформаций

`filter_nodes` — способ не сжигать деньги и не ломать граф. Два типичных мотива:

1. **Экономия.** После сплиттера в графе лежат и документы, и чанки. Если у `KeyphrasesExtractor` не указать `filter_nodes=is_chunk`, он честно вызовет LLM ещё и на каждом полном документе — то есть на тексте, который уже покрыт чанками.
2. **Корректность.** `CosineSimilarityBuilder` бросает `ValueError`, если у **любого** узла в графе нет свойства, по которому он считает косинус. Мы считаем эмбеддинги только для документов, поэтому без `filter_nodes=_is_document` билдер упадёт на первом же чанке.

Про порядок: `HeadlinesExtractor` обязан идти **до** `HeadlineSplitter`, а `SummaryExtractor` — до `EmbeddingExtractor`, который эти саммари векторизует. Builder-ы идут последними, потому что читают свойства, положенные экстракторами. Общее правило: **трансформация читает только то, что положила предыдущая**.

#### Parallel

`Parallel(KeyphrasesExtractor(...), EmbeddingExtractor(...))` — обёртка, которая группирует трансформации, не зависящие друг от друга. Внутри одной трансформации корутины по узлам и так выполняются конкурентно, так что основной выигрыш по времени вы получаете и без `Parallel` — но группировка делает список `transforms` читаемым и сразу показывает, между какими шагами зависимости нет.

### [3] Моделирование персонажей

Персонаж (`Persona`) — это структура из двух полей, которую мы задаём сами: `Persona(name="student", role_description="curious university student")`.

Работает это так: перед генерацией вопроса ragas берёт темы узла (ключевые фразы / сущности) и через промпт `ThemesPersonasMatchingPrompt` просит LLM **сопоставить персонажей с темами**. Получается словарь вида `{"student": ["arithmetic mean", "median"], "professor": [...]}`. Вопрос генерируется только для тех пар (персонаж, тема), которые LLM признала осмысленными, а сам персонаж попадает в промпт — поэтому студент и профессор про один и тот же чанк спросят по-разному.

Два практических момента:

- Если `persona_list` не передать в `TestsetGenerator`, ragas **сгенерирует персонажей сам** (`generate_personas_from_kg`). Но эта функция требует у узлов `summary` и `summary_embedding`, то есть без `SummaryExtractor` + `EmbeddingExtractor(embed_property_name="summary")` она упадёт. В нашем пайплайне персонажи заданы вручную.
- Параметр `num_personas: int = 3` в `generate()` **режет список**. Если вы передали 5 персонажей, по умолчанию будут использованы только 3 (список предварительно перемешивается).

### [3] Стили и длина вопросов

Стиль и длина — два enum-а библиотеки; выбирать их вручную не нужно, ragas сам перебирает все 4 × 3 = 12 комбинаций и семплирует их так, чтобы датасет был разнообразным. Знать их значения полезно, потому что они попадают в колонки итогового CSV:

```python
# ragas/testset/synthesizers/base.py — внутренности библиотеки, реализовывать не нужно
QueryStyle:  MISSPELLED | PERFECT_GRAMMAR | POOR_GRAMMAR | WEB_SEARCH_LIKE
QueryLength: SHORT | MEDIUM | LONG
```

Главное: стиль и длина не «постобрабатывают» готовый вопрос — они **передаются в промпт как условия генерации** вместе с персонажем, темой и контекстом. Именно так в датасете появляются строки вроде `Wht is TAI?` — это не баг, а `MISSPELLED` + `SHORT`, то есть ровно тот тип запроса, на котором реальный retrieval и ломается.

### [3] Single-hop и Multi-hop queries

Терминология:

- **single-hop** — для ответа достаточно одного чанка;
- **multi-hop** — нужно склеить информацию из ≥ 2 чанков;
- **specific** — вопрос про конкретный термин / сущность («что такое TAI?»);
- **abstract** — вопрос про идею или связь между темами («как X влияет на Y?»).

В ragas 0.4.3 доступны три синтезатора:

| Синтезатор                          | как выбирает узлы                                                         | что требует в графе                               |
|-------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------|
| `SingleHopSpecificQuerySynthesizer` | все узлы, у которых есть `property_name`                                  | свойство узла (`entities` / `keyphrases`)         |
| `MultiHopSpecificQuerySynthesizer`  | пары узлов, соединённые ребром типа `relation_type`                       | ребра `*_overlap` + `overlapped_items` в ребре    |
| `MultiHopAbstractQuerySynthesizer`  | кластеры узлов по ребрам со свойством `summary_similarity` (глубина до 3) | `summary_similarity` на ребрах + `themes` у узлов |

Обратите внимание на дефолты: у обоих `Specific`-синтезаторов `property_name = "entities"`, а у multi-hop ещё и `relation_type = "entities_overlap"`. Это дефолты «под `NERExtractor`». Если вы, как в нашем пайплайне, строите граф на `keyphrases`, **оба параметра нужно переопределить**.

##### Что делает синтезатор и зачем ему эти параметры

Задача синтезатора одна: найти в графе, **о чём** и **по какому тексту** можно задать вопрос. Он не пишет вопросы — он готовит сценарии (текст появится в фазе [4]).

`SingleHopSpecificQuerySynthesizer(property_name="keyphrases")` работает так:

1. берёт все чанки, у которых заполнено свойство `keyphrases`;
2. из каждого чанка достаёт список фраз — это **кандидаты в тему вопроса** (в коде поле называется `term`);
3. на отобранные темы заводит сценарии, а сам текст чанка кладёт в будущий `reference_contexts`.

То есть `property_name` отвечает на вопрос «откуда брать термин, вокруг которого строится вопрос». Промахнётесь с именем — генерация упадёт с `No nodes found with the 'entities' property`.

`MultiHopSpecificQuerySynthesizer(property_name="keyphrases", relation_type="keyphrases_overlap")` использует **оба** параметра, и на разных шагах:

1. **`relation_type`** — отбирает рёбра: в работу идут только пары чанков, соединённые ребром с таким типом. Не нашлось ни одного — `No clusters found in the knowledge graph`;
2. из ребра достаёт `overlapped_items` — фразы, общие для обоих чанков. Это и есть темы;
3. **`property_name`** — проверяет темы по узлам: в сценарий попадают только те чанки, у которых тема действительно лежит в `node.properties["keyphrases"]`;
4. текст обоих чанков идёт в `reference_contexts` с маркерами `<1-hop>` и `<2-hop>` — по ним потом удобно отличать multi-hop-строки в датасете.

Коротко: `relation_type` выбирает **пары чанков**, `property_name` — **темы внутри этих чанков**.

`MultiHopAbstractQuerySynthesizer(relation_property="summary_similarity", abstract_property_name="themes")` устроен иначе, и в двух местах не так, как можно ожидать:

1. **`relation_property`** — ребра он ищет **не по `type`, а по наличию свойства** с таким именем в `properties` ребра (`rel.get_property("summary_similarity")`). Это единственный синтезатор, который смотрит в `properties`, а не в `type`;
2. кластеры он собирает не из пар, а из **путей длиной до 3 узлов**. У нас эти рёбра соединяют документы, поэтому кластер — это группа тематически близких документов;
3. дальше он спускается от документов к их чанкам по рёбрам `child` (их создаёт сплиттер) — именно текст чанков попадёт в `reference_contexts`;
4. **`abstract_property_name`** — у этих чанков он берёт свойство `themes` и через промпт `ConceptCombinationPrompt` просит LLM собрать из тем разных документов **комбинацию концепций**, по которой имеет смысл задать один общий вопрос.

Отсюда требования к графу: `themes` у чанков (то есть `ThemesExtractor`) и рёбра со свойством `summary_similarity` между документами. И здесь есть отдельная ловушка: в библиотеке существует готовый `SummaryCosineSimilarityBuilder`, но он кладёт свойство с именем `summary_cosine_similarity` — на одно слово длиннее, чем ищет синтезатор. Поэтому имя приходится задавать руками через `new_property_name="summary_similarity"` у обычного `CosineSimilarityBuilder`.

### [3] и [4] Сценарии и генерация вопросов

Ключевая идея, которая делает датасет разнообразным: генерация **разделена на две фазы**.

**Фаза [3] — сценарии.** Сценарий — это ещё не вопрос, а «условия задачи»: набор узлов (один для single-hop, пара для multi-hop) плюс тема, персонаж, стиль и длина. Строится он так: отобрать подходящие узлы → посчитать, сколько вопросов нужно с одного узла (`ceil(n / len(nodes))`) → сопоставить темы с персонажами через LLM → построить **все** комбинации `(узел, тема, персонаж, стиль, длина)` → перемешать и отобрать нужное количество, стараясь не повторять пару «узел + тема».

Multi-hop делает то же самое, но стартует от пар узлов, соединённых ребром, а темами берёт `overlapped_items` из ребра — ровно те ключевые фразы, которые есть в обоих чанках. Это гарантирует, что вопрос действительно «про пересечение», а не про два случайных текста.

**Фаза [4] — сэмплы, то есть сами вопросы.** Сэмпл в терминологии ragas — это одна готовая строка датасета: `user_input` + `reference` + `reference_contexts`. Каждый сценарий уходит в промпт `QueryAnswerGenerationPrompt`, и LLM возвращает structured output `{query, answer}`. В промпте жёстко зашито требование **faithfulness к контексту**: «Do not add any information not included in or inferable from the context». Именно поэтому сгенерированный `reference` годится как эталон — он по построению не выходит за пределы `reference_contexts`.

Зачем такое разделение? Оно превращает «сгенерируй 100 вопросов» в контролируемую задачу семплирования: разнообразие по чанкам, персонажам и стилям обеспечивается **до** обращения к LLM, а не надеждой на то, что модель сама не выдаст 100 однотипных «What is X?».

**Про `testset_size`.** Он делится по `query_distribution` через `math.ceil`, а внутри синтезаторов округляется вверх ещё раз. Итоговое количество строк — **примерно** `testset_size`, обычно чуть больше.

---

## Собираем пайплайн

Теперь тот же маршрут со схемы, но кодом. Всё ниже — уже наш код, его нужно писать; полный файл: `full_pipeline.py`.

Соответствие со схемой: шаги 0–2 — это блок **[1]**, шаг 3 — блок **[2]**, шаги 4–5 — настройка блоков **[3]/[4]**, шаг 6 — сам прогон.

Для начала выполним все необходимые импорты:

```python
import os
import asyncio
from time import perf_counter
from dotenv import load_dotenv

from datasets import load_dataset
from openai import AsyncOpenAI

import ragas.async_utils as ragas_async_utils
import ragas.executor as ragas_executor
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    HeadlineSplitter,
    HeadlinesExtractor,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
    apply_transforms,
)
from ragas.testset.transforms.extractors.llm_based import ThemesExtractor

load_dotenv()
```

### 0. Документы

На вход ragas нужен просто `list[str]`. Берём три статьи из английской Википедии через HuggingFace-датасет в streaming-режиме (чтобы не качать десятки гигабайт):

```python
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"

def load_hf_documents() -> list[str]:
    doc_titles = ["International Atomic Time", "Agricultural science", "Arithmetic mean"]
    found: dict[str, str] = {}

    stream = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    for row in stream:
        if row["title"] in doc_titles:
            found[row["title"]] = row["text"]
            if len(found) == len(doc_titles):
                break

    missing = set(doc_titles) - found.keys()
    if missing:
        raise RuntimeError(f"Titles not found: {sorted(missing)}")

    return [found[title] for title in doc_titles]
```

Статьи выбраны неполитические и достаточно длинные — важно, чтобы документ был заметно больше `min_tokens`, иначе сплиттер вернёт его одним куском.

### 1. LLM и эмбеддинги

```python
openai_client = AsyncOpenAI(
    api_key=os.getenv("AI_TUNNEL_API_KEY"),
    base_url="https://api.aitunnel.ru/v1/",
)
llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

`llm_factory` возвращает обёртку с structured output (через instructor), поэтому все промпты ragas отдают готовые Pydantic-объекты, а не текст, который надо парсить. Клиент передаётся снаружи — значит подойдёт любой OpenAI-совместимый провайдер, достаточно поменять `base_url`. `max_tokens=8192` здесь не роскошь: `reference` для multi-hop получается длинным, и при маленьком лимите ответ обрежется и structured output развалится.

### 2. Создание графа

```python
def build_knowledge_graph(documents: list[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(Node(type=NodeType.DOCUMENT, properties={"page_content": doc}))
    return kg
```

Никакой магии: граф стартует как список узлов-документов с единственным свойством `page_content`. Всё остальное появится в результате трансформаций.

### 3. Трансформации

```python
def build_transforms(llm, embedding_model):
    def _is_document(node: Node) -> bool:
        return node.type == NodeType.DOCUMENT

    def _is_chunk(node: Node) -> bool:
        return node.type == NodeType.CHUNK

    return [
        # First, we extract the `headlines` feature.
        HeadlinesExtractor(llm=llm, filter_nodes=_is_document),

        # Based on the `headlines` feature, we split the original documents into chunks.
        HeadlineSplitter(min_tokens=300, max_tokens=1000),

        # `summary` must exist before we can embed it in the next step.
        SummaryExtractor(llm=llm, filter_nodes=_is_document),

        # Extractors
        Parallel(
            KeyphrasesExtractor(llm=llm, property_name="keyphrases", filter_nodes=_is_chunk),
            ThemesExtractor(llm=llm, property_name="themes", filter_nodes=_is_chunk),
            EmbeddingExtractor(
                embedding_model=embedding_model,
                property_name="summary_embedding",
                embed_property_name="summary",
                filter_nodes=_is_document,
            ),
        ),

        # Relations
        Parallel(
            # MultiHopAbstractQuerySynthesizer looks up relations by the property name
            # "summary_similarity", so `new_property_name` has to match it exactly.
            # Our three articles are topically unrelated: measured summary similarities
            # are 0.13-0.26, so the ragas default of 0.7 would yield zero relations.
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.25,
                filter_nodes=_is_document,
            ),
            OverlapScoreBuilder(
                property_name="keyphrases",
                new_property_name="overlap_score",
                threshold=0.01,
                distance_threshold=0.9,
                filter_nodes=_is_chunk,
            ),
        ),
    ]
```

Здесь важно видеть, что пайплайн работает **на двух уровнях сразу**, и путать их нельзя:

- **уровень чанков** — `keyphrases` и `themes`, поверх `keyphrases` строятся ребра `keyphrases_overlap`. Это питает single-hop и multi-hop specific;
- **уровень документов** — `summary`, его эмбеддинг `summary_embedding` и косинусные ребра `summary_similarity` между документами. Это питает multi-hop abstract, который потом сам спустится от документов к их чанкам.

Отсюда и `filter_nodes` в каждом шаге: он не только экономит LLM-вызовы, но и удерживает каждое свойство на своём уровне графа.

#### Почему `summary` считается по документам, а не по чанкам

Соблазнительно сделать проще: посчитать эмбеддинги прямо по тексту чанков (`embed_property_name="page_content"`, `filter_nodes=_is_chunk`) и строить косинусные ребра между чанками. Так делать не надо, и вот четыре причины — от самой важной к самой скучной.

**1. Косинус между чанками одного документа всегда выше, чем между документами.** Все чанки статьи про TAI написаны про атомное время, их эмбеддинги почти совпадают. Значит самые сильные косинусные связи окажутся **внутри** одного документа, и кластеры соберутся там же. Abstract-вопрос выродится в «свяжи два абзаца одной статьи» — а это ровно то, что уже умеет multi-hop specific через `keyphrases_overlap`. Уровень документов даёт единственное, чего нет ни у одного другого синтезатора: связь **между разными документами**.

**2. Синтезатор рассчитан на документы и сам спускается к чанкам.**

```python
# ragas/testset/synthesizers/multi_hop/abstract.py — внутренности библиотеки
for node in cluster:
    child_nodes = [rel.target for rel in child_relationships if rel.source == node]
    if child_nodes:
        nodes.extend(child_nodes)   # узел кластера → все его чанки
    else:
        nodes.append(node)          # fallback: если детей нет, берём сам узел
```

То есть кластеризация задумана на документах, а в `reference_contexts` попадают их чанки. Побочный эффект приятный: кластер получает весь набор тем документа, и `ConceptCombinationPrompt` есть из чего выбирать комбинацию концепций. Если в кластере лежат чанки, срабатывает `else` (эта ветка нужна для заранее нарезанных корпусов — см. `default_transforms_for_prechunked`), и выбирать приходится из двух-трёх чанков.

**3. Саммари чанка — это почти сам чанк.** Промпт `SummaryExtractorPrompt` просит «Summarize the given text in less than 10 sentences», а наш чанк — 300–1000 токенов, то есть и так примерно десяток предложений. Сжатия не происходит, вектор такого саммари практически равен вектору `page_content` — мы платим лишний LLM-вызов за то, что получается бесплатно. На документе в несколько тысяч токенов сжатие настоящее, и вектор саммари — это «вектор темы документа».

**4. Цена.** Три LLM-вызова и три эмбеддинга вместо двенадцати (по числу чанков). На реальном корпусе разница уже в десятки раз.

Есть и пятая причина, которая проявится, если вы добавите в пайплайн `CustomNodeFilter`: он умеет читать **только** саммари документа. Для чанка он идёт к родителю через `get_parent_nodes` и берёт `properties["summary"]` у него, потому что рубрика оценивает «насколько чанк соответствует теме документа» — саммари самого чанка для этого бесполезно. Без документного `summary` фильтр напишет warning и не отфильтрует ничего.

> Оговорка про очень длинные документы: `SummaryExtractor.extract` режет текст по `max_token_limit = 32000` и суммаризует только первый кусок. Для статей Википедии это не проблема, но для книги саммари опишет лишь её начало.

### 4. Генератор и персонажи

```python
def build_generator(llm, embedding_model, kg: KnowledgeGraph) -> TestsetGenerator:
    personas = [
        Persona(name="student", role_description="curious university student"),
        Persona(name="professor", role_description="university professor"),
    ]
    return TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
        persona_list=personas,
    )
```

Персонажи заданы вручную — так мы не платим за их автогенерацию и не зависим от наличия `summary`/`summary_embedding` в графе.

### 5. Распределение типов вопросов

```python
def build_query_distribution(llm):
    return [
        (
            SingleHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases",   # тема, вокруг которой строится вопрос
            ),
            0.4,
        ),
        (
            MultiHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases",
                relation_type="keyphrases_overlap",
                # OverlapScoreBuilder: relation.type = f"{property_name}_overlap"
                # "keyphrases" + "_overlap" = "keyphrases_overlap"
            ),
            0.4,
        ),
        (
            MultiHopAbstractQuerySynthesizer(
                llm=llm,
                relation_property="summary_similarity",  # ребра от CosineSimilarityBuilder
                abstract_property_name="themes",         # свойство от ThemesExtractor
            ),
            0.2,
        ),
    ]
```

Это то место, где сходится вся конфигурация графа. Две цепочки, которые стоит держать в голове, — по одной на каждый уровень:

```
KeyphrasesExtractor(property_name="keyphrases")
        ↓  кладёт node.properties["keyphrases"] чанкам
OverlapScoreBuilder(property_name="keyphrases")
        ↓  создаёт Relationship(type="keyphrases_overlap", properties={... "overlapped_items": [...]})
MultiHopSpecificQuerySynthesizer(property_name="keyphrases", relation_type="keyphrases_overlap")
```

```
SummaryExtractor  →  ThemesExtractor(property_name="themes")
        ↓  кладёт summary документам, themes чанкам
EmbeddingExtractor(embed_property_name="summary", property_name="summary_embedding")
        ↓  кладёт вектор саммари документам
CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity")
        ↓  создаёт Relationship(properties={"summary_similarity": 0.257})
MultiHopAbstractQuerySynthesizer(relation_property="summary_similarity", abstract_property_name="themes")
```

Если хоть одно звено рассинхронизировано — вы получите либо `No nodes found with the 'entities' property`, либо `No clusters found in the knowledge graph`, либо `No relationships match the provided condition`. Все три сообщения означают одно и то же: **синтезатор ищет в графе то, чего вы туда не положили**.

Веса `0.4 / 0.4 / 0.2` — это доли от `testset_size`, а не вероятности при семплировании: ragas просто делит размер датасета между синтезаторами.

### 6. Запуск

```python
kg = build_knowledge_graph(docs)
transforms = build_transforms(llm, embeddings)
generator = build_generator(llm, embeddings, kg)

with asyncio.Runner() as runner:
    _share_event_loop_across_ragas_runs(runner)

    apply_transforms(kg, transforms=transforms)
    print(f"KG after transforms: nodes={len(kg.nodes)} relationships={len(kg.relationships)}")

    testset = generator.generate(
        testset_size=6,
        query_distribution=build_query_distribution(llm),
    )

testset.to_pandas().to_csv("eval_dataset.csv", index=False)
```

Два шага — `apply_transforms` (построить граф) и `generator.generate` (сгенерировать вопросы) — специально разделены. Между ними полезно вставить `kg.save("kg.json")`: граф строится дорого, а генерировать по нему можно сколько угодно раз с разными распределениями.

Самая полезная диагностика — распечатать, сколько и каких ребер получилось, **до** генерации. Для трёх статей вывод выглядит так:

```
KG before transforms: KnowledgeGraph(nodes: 3, relationships: 0)
KG after transforms: nodes=15 relationships=29
  rel child: 12                # документ → его чанки (создаёт сплиттер)
  rel next: 9                  # соседние чанки внутри документа (создаёт сплиттер)
  rel summary_similarity: 1    # документ ↔ документ (CosineSimilarityBuilder)
  rel keyphrases_overlap: 7    # чанк ↔ чанк (OverlapScoreBuilder)
```

Если напротив `summary_similarity` или `keyphrases_overlap` стоит `0` — соответствующий синтезатор упадёт, и порог (`threshold`) надо снижать. Проверить это за секунды дешевле, чем поймать исключение через три минуты прогона.

<small>Техническая деталь: `_share_event_loop_across_ragas_runs` подменяет `ragas.async_utils.run` и `ragas.executor.run` так, чтобы обе фазы работали в одном event loop. Без этого `AsyncOpenAI`-клиент, созданный вне цикла ragas, на Windows натыкается на закрытый loop. К логике генерации это отношения не имеет, но без такого шима код падает.</small>

### 7. Результат

`testset.to_pandas()` даёт таблицу со следующими колонками:

| колонка              | что внутри                                                            |
|----------------------|-----------------------------------------------------------------------|
| `user_input`         | сгенерированный вопрос                                                |
| `reference_contexts` | список чанков, из которых он сгенерирован (для multi-hop с `<N-hop>`) |
| `reference`          | эталонный ответ, построенный **только** по этим контекстам            |
| `persona_name`       | имя персонажа                                                         |
| `query_style`        | `MISSPELLED` / `PERFECT_GRAMMAR` / `POOR_GRAMMAR` / `WEB_SEARCH_LIKE` |
| `query_length`       | `SHORT` / `MEDIUM` / `LONG`                                           |
| `synthesizer_name`   | какой синтезатор породил строку                                       |

Примеры вопросов из реального прогона (3 статьи, `testset_size=6` → 8 строк: 3 + 3 + 2):

```
why there is leap seconds?
  → student / POOR_GRAMMAR / SHORT / single_hop_specific_query_synthesizer

How did gravitational time dilation affect the International Atomic Time (TAI)?
  → professor / PERFECT_GRAMMAR / SHORT / single_hop_specific_query_synthesizer

How did early experiments by Johann Friedrich Mayer and the long-term Rothamsted
trials, together with US policies such as the Hatch Act, contribute to the
develpment of Agricultural science, and how does Agricultural science diffr
from agronomy and agriculture?
  → multi_hop_specific_query_synthesizer   (reference_contexts: <1-hop> + <2-hop>)

What is a weighted average and how is the weighted average of atomic clocks used
in International Atomic Time (over 450 clocks in 80+ national laboratories)?
  → multi_hop_abstract_query_synthesizer   (reference_contexts: <1-hop> + <2-hop>)
```

Ровно то, что мы хотели получить вместо «идеальных» LLM-вопросов: опечатки (`develpment`, `diffr`), сломанная грамматика, короткие запросы и вопросы, которые физически нельзя закрыть одним чанком.

Отдельно стоит посмотреть на последний пример — он показывает, **зачем вообще нужен был косинус по саммари**. Ребро `summary_similarity` связало статьи *Arithmetic mean* и *International Atomic Time*, синтезатор спустился к их чанкам, нашёл в темах общее понятие «weighted average» и построил вопрос, который склеивает определение из статистики с его применением в метрологии. Ни один из чанков по отдельности на такой вопрос не отвечает, и `keyphrases_overlap` эту пару не нашёл бы — статьи написаны разным словарём, буквальных пересечений фраз между ними почти нет. Это и есть разница между *specific* (общий термин) и *abstract* (общая идея).

Две вещи, которые бросаются в глаза в CSV и обе являются **особенностями ragas 0.4.3**, а не ошибками пайплайна:

1. У multi-hop строк **пустые `persona_name`, `query_style`, `query_length`**. Персонаж, стиль и длина в сценарии есть и в промпт передаются — но `MultiHopQuerySynthesizer._generate_sample` возвращает `SingleTurnSample` только с `user_input` / `reference` / `reference_contexts` и не прокидывает эти поля дальше. У single-hop они прокидываются.
2. Просьба про «как бы от лица персонажа» иногда **протекает в текст вопроса**: `"ok so as a student i wanna know in detail why the arithmetic mean is..."`. Это следствие того, что `role_description` подаётся в промпт как часть условий. Лечится более нейтральными описаниями персонажей или кастомизацией промпта через `PromptMixin`.

### Чек-лист перед запуском на своих данных

- Документы длиннее `min_tokens`, иначе чанков не будет.
- `HeadlinesExtractor` до `HeadlineSplitter`, `SummaryExtractor` до `EmbeddingExtractor`.
- У всех экстракторов и builder-ов выставлен `filter_nodes` (`_is_chunk` / `_is_document`).
- `property_name` синтезатора совпадает с тем, что положил экстрактор (`entities` — дефолт!).
- `relation_type` совпадает с типом ребра, который реально создал builder (`{property_name}_overlap`).
- Для abstract-синтезатора: ребра несут свойство именно `summary_similarity`, а `threshold` подобран по замеренным на своём корпусе значениям косинуса.
- Счётчик ребер по типам распечатан после `apply_transforms` — нулей быть не должно.
- `kg.save(...)` после `apply_transforms` — чтобы не платить за граф повторно.
- Итоговый датасет **просмотрен глазами**. Синтетика — это черновик разметки, а не готовый эталон: строки с «протёкшим» персонажем или слишком общим вопросом лучше выкинуть до того, как по ним начнут сравниваться конфигурации RAG.

На этом первая часть заканчивается: у нас есть `eval_dataset.csv` с колонками `user_input`, `reference_contexts`, `reference`. Во второй части подключим к нему собственно метрики — и посмотрим, как по этому датасету сравнивать конфигурации RAG.

___

# Evaluation metrics


## Context precision


