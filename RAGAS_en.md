# Introduction

The task:
> We have a RAG system. We swap embeddings / vector store / reranker / chunking, and we want an objective answer to the question "which configuration is better?".

To solve this we will use the Ragas library.
In this context Ragas fundamentally does two things:

- generates an evaluation dataset;
- evaluates the RAG pipeline (metrics).

Dataset generation and pipeline evaluation are two completely independent processes. Each of them plays its own part in solving the original problem of picking the best configuration.

> Everything here is written and verified against `ragas 0.4.3`.

# Evaluation dataset

## The problem

A classic RAG pipeline, in simplified form, takes a user question
(`user_input`) and returns an answer (`response`) based on the context
(`retrieved_contexts`) obtained by retrieval over the knowledge base.

The good news is that these three entities alone (`user_input, response, retrieved_contexts`) are already enough to compute some metrics — for example **Faithfulness** and **Answer Relevancy** (more on metrics below). This data can be obtained by simply exporting the logs of a running agent.

Some metrics, however, require additional data. **Context Precision** and **Answer Correctness**, for instance, need ground-truth information — an "ideal answer" (`reference`). There are two common ways to obtain data with such ground truth:

**1) Building the dataset with human experts**

Take the documentation and real production questions, and have an expert write the answers.
This approach gives high quality, but it is slow, expensive, and impossible to automate.

**2) Generating the dataset with an LLM**

For each document or chunk, an LLM generates a question (`user_input`) and a reference answer
(`reference`). Applied naively, this produces a dataset that is far too artificial: real users often do not know the terminology, make typos, ask overly short questions, and have wildly different levels of background knowledge — and some questions require information from several documents at once.

Ragas exists to solve exactly these problems and to produce a more realistic dataset.

## How ragas generates a dataset: the full picture

First, what we prepare ourselves, before any ragas pipeline gets involved. It is exactly six things.

```
docs                 list[str] — the document texts
llm, embeddings      model clients: llm_factory(...) and OpenAIEmbeddings(...)
transforms           list of graph enrichment steps      → used by step [2]
personas             list of Persona: whose voice to use → used by step [3]
query_distribution   which question types, in what ratio → used by step [3]
testset_size         how many rows we want               → used by step [3]
```

Now the route itself. It is linear: to the left of the indentation is what we call, and on the arrows is what we get out.

```
docs
   │
   ▼
[1] kg = build_knowledge_graph(docs)
       create a KnowledgeGraph: one DOCUMENT node per document,
       whose only property is page_content
   │
   │  kg: a graph of documents, still without chunks and without edges
   ▼
[2] apply_transforms(kg, transforms)
       run the graph transformation pipeline, 4 kinds of steps (we choose the order):
         Splitters   →  cut DOCUMENT into CHUNK
         Extractors  →  add properties to nodes (headlines, keyphrases, summary, ...)
         Relations   →  build edges between nodes
         Filters     →  drop useless chunks
       the graph is mutated in place: it is still the very same kg object
   │
   │  kg: chunks + properties + edges
   ▼
generator = TestsetGenerator(llm, embeddings, kg, persona_list=personas)
testset = generator.generate(testset_size, query_distribution)
   │   a single generate() call, with two consecutive phases inside:
   │
   ├─ [3] SCENARIO SELECTION — there are no questions yet.
   │      query_distribution decides how many scenarios of each type to collect,
   │      personas — whose voice they will be asked in,
   │      testset_size — how many of them in total.
   │      A scenario is a "problem statement": which node (or pair of nodes),
   │      about which theme, from which persona, in which style, at which length.
   │      In the console: the "Generating Scenarios" progress bar
   │
   └─ [4] QUESTION GENERATION — this is where the LLM writes text.
          One LLM call per scenario, returning the question (user_input)
          and the reference answer (reference) for it.
          One scenario = one sample = one row of the future dataset.
          In the console: the "Generating Samples" progress bar
   │
   │  testset: a Testset object
   ▼
testset.to_pandas().to_csv("eval_dataset.csv")
   │
   ▼
eval_dataset.csv:  user_input | reference_contexts | reference | ...
```

The same route in code (schematically) — first we prepare all six inputs, then five lines make up the pipeline itself:

```python
# prepare the inputs
openai_client = AsyncOpenAI(api_key=..., base_url=...)
docs = load_hf_documents()
llm = llm_factory("gpt-5-mini", client=openai_client)
embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
transforms = build_transforms(llm, embeddings)
personas = [Persona(name="student", role_description="curious university student")]
query_distribution = build_query_distribution(llm)
testset_size = 6

# the route
kg = build_knowledge_graph(docs)                                        # [1]
apply_transforms(kg, transforms=transforms)                             # [2]
generator = TestsetGenerator(llm, embeddings, kg, persona_list=personas)
testset = generator.generate(testset_size, query_distribution)          # [3] + [4]
testset.to_pandas().to_csv("eval_dataset.csv")
```

Each of the `build_*` functions is ours to write — they are broken down step by step in the "Assembling the pipeline" section.

Why a graph at all, rather than just a list of chunks? A list of chunks only gives you "single fragment" questions. Edges let you take a **pair of related chunks** and ask for a question whose answer lives in two places at once — and that is exactly what multi-hop means.

From here we go through the blocks in order:

- **[1] and [2]** — what a `KnowledgeGraph` consists of and what kinds of transformations exist (Splitters / Extractors / Relations / Filters);
- **[3] and [4]** — personas, query styles and lengths, single-hop vs multi-hop, and both generation phases;
- after that — the complete working code in "Assembling the pipeline".

> Code blocks below that carry a path comment (for example `# ragas/testset/graph.py`) show **library internals**. You do not need to write any of it yourself — it is included only to make visible where the property and type names we rely on in our configuration come from.

### [1] KnowledgeGraph

A KnowledgeGraph is the core data structure that stores a document corpus as a graph. Its nodes are fragments of text (chunks) enriched with extracted entities, keyphrases, headlines and other auxiliary information.
Its edges are semantic or logical connections between them (for example "relates to", "uses", "follows from").

Structurally it is just two lists:

```python
# ragas/testset/graph.py — library internals, nothing to implement here
class KnowledgeGraph:
    nodes: list[Node]                    # Node(type: NodeType, properties: dict)
    relationships: list[Relationship]    # Relationship(type: str, source, target, properties)
```

_The graph can be saved and reused: `kg.save("kg.json")` / `KnowledgeGraph.load("kg.json")`. Building the graph is the most expensive part (lots of LLM calls), so when experimenting people build the graph once and then run many different `query_distribution` settings over it._

### [2] transforms

The data processing pipeline that enriches the `KnowledgeGraph` with information. Transformations push texts through the LLM and the embedding model.

`transforms` is simply a list, and the order in it *is* the logic of how the graph gets built. There are four kinds of transformations (Splitters, Extractors, Relationship builders, Filters), and all of them share one parameter — `filter_nodes: Callable[[Node], bool]` — which decides which nodes to work on (all of them by default).

#### Splitters

> *create / split Nodes*

- They break the original bulk text into base nodes (`Node` / chunks).
- **Example:** **`HeadlineSplitter`** cuts a document not by a fixed number of characters but along logical headlines (H1, H2, H3), keeping sections contextually intact (the nodes must already carry the `headlines` and `page_content` properties).


_**An important and non-obvious detail:**_  
_contrary to the order described above (Splitters first, Extractors second),_  
_we are obliged to extract the headlines **first** (HeadlinesExtractor)_  
_and only **then** split into chunks along those headlines (HeadlineSplitter)._  
_Otherwise HeadlineSplitter tries to read a `headlines` property that does not exist_  
_and raises `ValueError: 'headlines' property not found in this node`._


What `HeadlineSplitter(min_tokens=300, max_tokens=1000)` actually does:

- cuts the text at the positions of the headlines listed in `headlines`; sections that are too long get cut further along word boundaries, and sections that are too short are **merged with their neighbours** so that no stub chunks appear;
- if the whole document is shorter than `min_tokens`, or no headlines were found, it returns the original node untouched;
- besides nodes it also creates `child` edges (document → chunk) and `next` edges (chunk → chunk).

#### Extractors

> *add properties to a Node*

| Extractor                   | property_name       | what it writes                  | default limit         |
|-----------------------------|---------------------|---------------------------------|-----------------------|
| `HeadlinesExtractor`        | `headlines`         | list of headlines               | `max_num=5`           |
| `KeyphrasesExtractor`       | `keyphrases`        | keyphrases                      | `max_num=5`           |
| `NERExtractor`              | `entities`          | named entities                  | `max_num_entities=10` |
| `ThemesExtractor`           | `themes`            | themes and concepts             | `max_num_themes=10`   |
| `SummaryExtractor`          | `summary`           | a summary (under 10 sentences)  | —                     |
| `TitleExtractor`            | `title`             | the document title              | —                     |
| `TopicDescriptionExtractor` | `topic_description` | description of the main topic   | —                     |
| `EmbeddingExtractor`        | `embedding`         | vector of `embed_property_name` | —                     |

All of them except `EmbeddingExtractor` are LLM calls with strict structured output (a Pydantic model plus instructor), so the result is always typed. You can try an extractor outside the pipeline like this (an illustration for the console; there is no need to call it manually in the pipeline):

```python
extractor = NERExtractor()
await extractor.extract(node)

> ('entities', ['Einstein', 'theory of relativity', 'space', 'time' ...])
```

`EmbeddingExtractor` has two parameters that are easy to mix up: `embed_property_name` is **what** we vectorize (`page_content`, `summary`), while `property_name` is **where** we put the result (`embedding`, `summary_embedding`). Hence the standard "embedding of the document summary" combination: `EmbeddingExtractor(embed_property_name="summary", property_name="summary_embedding")`.

#### Relationship builders

> *create edges between Nodes*

A crucial part of building the knowledge graph, the one that provides **connections between nodes**. Without Relations the graph nodes will be enriched after the transformations (they will have entities, title and so on), but they will not be connected to each other — you end up with a graph with no edges.

A builder reads **one node property** — `property_name`. Its value must match whatever the extractor wrote in the previous step. In our pipeline the chain looks like this:

```python
# the extractor writes a "keyphrases" property into every chunk
KeyphrasesExtractor(llm=llm, property_name="keyphrases", filter_nodes=_is_chunk)
# → node.properties["keyphrases"] = ["International Atomic Time", "atomic clock", "UTC", ...]

# the builder reads exactly that same "keyphrases" property
OverlapScoreBuilder(property_name="keyphrases", new_property_name="overlap_score",
                    distance_threshold=0.9, threshold=0.01, filter_nodes=_is_chunk)
```

The builder then takes **every pair of chunks** and, for each pair, computes how strongly their keyphrases overlap. Literally step by step, on an example of two chunks:

```
chunk A  keyphrases: ["International Atomic Time", "atomic clock", "leap second", "UTC", "geoid"]
chunk B  keyphrases: ["atomic clocks", "caesium", "UTC", "Circular T", "BIPM"]
```

1. It compares every phrase in A with every phrase in B — here that is 5 × 5 = **25 comparisons**.
2. A pair counts as matched if the strings are at least `distance_threshold=0.9` similar. Two will match: `atomic clock` / `atomic clocks` (≈ 0.98) and `UTC` / `UTC` (1.0).
3. It computes score = matched pairs / all comparisons = 2 / 25 = **0.08**.
4. It compares the score against `threshold=0.01`. Above the threshold it creates an edge; below it, these two chunks stay unconnected.

The edge for our pair comes out like this:

```python
Relationship(
    source=<chunk A>,
    target=<chunk B>,
    type="keyphrases_overlap",                   # the name of the connection
    properties={
        "keyphrases_overlap_score": 0.08,        # the result of step 3
        "overlapped_items": [                    # what exactly matched in step 2
            ("atomic clock", "atomic clocks"),
            ("UTC", "UTC"),
        ],
    },
)
```

The multi-hop question synthesizer later reads both fields, but for different purposes:

- **`type`** — to **find** the right edges: the synthesizer only takes edges whose `type` matches its `relation_type` parameter. No match means it has no edges to work with, and generation fails with `No clusters found in the knowledge graph`.
- **`properties["overlapped_items"]`** — to work out **what to ask about**: the matched phrases become the theme of the question. That is why a multi-hop question comes out meaningful rather than "tell me about these two random texts".

So an edge has to satisfy two requirements: its `type` must match the synthesizer's `relation_type`, and it must carry `overlapped_items`. Cosine edges fail the second requirement, so multi-hop specific cannot work on them.

The other builders are arranged the same way; only the comparison method differs. The column shows the default `property_name`, i.e. which node property the builder will read if you do not specify your own:

| Builder                    | default `property_name` | how it compares          | edge `type`                   |
|----------------------------|-------------------------|--------------------------|-------------------------------|
| `CosineSimilarityBuilder`  | `embedding`             | cosine between vectors   | `new_property_name`           |
| `JaccardSimilarityBuilder` | `entities`              | Jaccard over string sets | `new_property_name`           |
| `OverlapScoreBuilder`      | `entities`              | fuzzy string comparison  | **`{property_name}_overlap`** |

That last row is ragas' **biggest gotcha**: the type name is composed differently from all the others. The cosine builder takes it straight from `new_property_name`, whereas `OverlapScoreBuilder` ignores `new_property_name` for the type builds the name by concatenating `property_name`. So with `property_name="keyphrases"` the edge type is `keyphrases_overlap`, and that exact string is what you must pass to the synthesizer.

**One more `OverlapScoreBuilder` detail:**

- before iterating, the builder throws away the top 5% most frequent phrases across the whole corpus — otherwise the graph would collapse into a clique because of a word that appears in every chunk.

**About `threshold` in `CosineSimilarityBuilder`** — it is an ordinary cosine threshold, but you cannot pick it "by intuition": the absolute values depend both on the embedding model and on how homogeneous the corpus is. The class default is `0.9`; the library's `default_transforms` uses `0.7` for long documents.

For our three articles, the cosines between their summary embeddings turned out to be:

```
International Atomic Time  <->  Agricultural science : 0.132
International Atomic Time  <->  Arithmetic mean      : 0.257
Agricultural science       <->  Arithmetic mean      : 0.202
```

In other words, with the library threshold of `0.7` not a single edge would appear, and the synthesizer that walks those edges would fail with `No relationships match the provided condition`. The practical takeaway is simple: thresholds must be measured on your own corpus, not copied from examples. We settled on `0.25`, which keeps the single most meaningful edge (arithmetic mean ↔ atomic time, which is literally computed as a weighted average over clocks) while leaving accidental pairs such as "agriculture ↔ atomic time" unconnected.

#### Filters

> *remove Nodes*

- `CustomNodeFilter` is an LLM filter for "is this chunk suitable for a question at all". For every chunk it takes the `summary` of the **parent** document (via the `child` edge), asks the LLM to score the chunk on a 1–5 rubric ("how well does the content match the document's topic, and is there anything worth asking about"), and deletes the node if the score is ≤ `min_score=2`.
- An important consequence: the filter **only works if the document has a `summary`**. Without a `SummaryExtractor` in the pipeline, it just logs a warning and filters nothing.

#### filter_nodes and the order of transformations

`filter_nodes` is how you avoid burning money and breaking the graph. Two typical motivations:

1. **Cost.** After the splitter, the graph holds both documents and chunks. If you do not give `KeyphrasesExtractor` a `filter_nodes=is_chunk`, it will dutifully call the LLM on every full document as well — that is, on text that is already covered by the chunks.
2. **Correctness.** `CosineSimilarityBuilder` raises `ValueError` if **any** node in the graph lacks the property it computes the cosine over. We only compute embeddings for documents, so without `filter_nodes=_is_document` the builder fails on the very first chunk.

On ordering: `HeadlinesExtractor` must come **before** `HeadlineSplitter`, and `SummaryExtractor` must come before the `EmbeddingExtractor` that vectorizes those summaries. Builders come last, because they read properties written by extractors. The general rule: **a transformation only reads what the previous one wrote**.

#### Parallel

`Parallel(KeyphrasesExtractor(...), EmbeddingExtractor(...))` is a wrapper that groups transformations which do not depend on each other. Within a single transformation the per-node coroutines already run concurrently, so you get most of the wall-clock benefit even without `Parallel` — but the grouping keeps the `transforms` list readable and shows at a glance which steps have no dependencies between them.

### [3] Modelling personas

A persona (`Persona`) is a two-field structure that we define ourselves: `Persona(name="student", role_description="curious university student")`.

Here is how it works: before generating a question, ragas takes the node's themes (keyphrases / entities) and uses the `ThemesPersonasMatchingPrompt` prompt to ask the LLM to **match personas to themes**. The result is a mapping such as `{"student": ["arithmetic mean", "median"], "professor": [...]}`. A question is generated only for those (persona, theme) pairs the LLM considered sensible, and the persona itself goes into the prompt — which is why a student and a professor will ask about the same chunk differently.

Two practical notes:

- If you do not pass `persona_list` to `TestsetGenerator`, ragas will **generate the personas itself** (`generate_personas_from_kg`). That function, however, requires nodes to have `summary` and `summary_embedding`, so without `SummaryExtractor` + `EmbeddingExtractor(embed_property_name="summary")` it will fail. In our pipeline the personas are defined by hand.
- The `num_personas: int = 3` parameter of `generate()` **truncates the list**. If you passed 5 personas, only 3 will be used by default (the list is shuffled first).

### [3] Query styles and lengths

Style and length are two library enums; you do not need to choose them manually, since ragas iterates over all 4 × 3 = 12 combinations itself and samples them so that the dataset stays diverse. Knowing their values is still useful, because they end up as columns in the final CSV:

```python
# ragas/testset/synthesizers/base.py — library internals, nothing to implement here
QueryStyle:  MISSPELLED | PERFECT_GRAMMAR | POOR_GRAMMAR | WEB_SEARCH_LIKE
QueryLength: SHORT | MEDIUM | LONG
```

The key point: style and length do not "post-process" a finished question — they are **passed into the prompt as generation conditions**, together with the persona, the theme and the context. That is exactly how rows like `Wht is TAI?` appear in the dataset. It is not a bug, it is `MISSPELLED` + `SHORT` — precisely the kind of query on which real retrieval breaks.

### [3] Single-hop and multi-hop queries

Terminology:

- **single-hop** — one chunk is enough to answer;
- **multi-hop** — you need to combine information from ≥ 2 chunks;
- **specific** — a question about a concrete term or entity ("what is TAI?");
- **abstract** — a question about an idea or a connection between themes ("how does X affect Y?").

ragas 0.4.3 ships three synthesizers:

| Synthesizer                         | how it picks nodes                                            | what it requires in the graph                      |
|-------------------------------------|---------------------------------------------------------------|----------------------------------------------------|
| `SingleHopSpecificQuerySynthesizer` | every node that has `property_name`                           | a node property (`entities` / `keyphrases`)        |
| `MultiHopSpecificQuerySynthesizer`  | pairs of nodes joined by an edge of type `relation_type`      | `*_overlap` edges + `overlapped_items` in the edge |
| `MultiHopAbstractQuerySynthesizer`  | node clusters over `summary_similarity` edges (depth up to 3) | `summary_similarity` on edges + `themes` on nodes  |

Pay attention to the defaults: both `Specific` synthesizers have `property_name = "entities"`, and the multi-hop one additionally has `relation_type = "entities_overlap"`. Those are the "made for `NERExtractor`" defaults. If, like us, you build the graph on `keyphrases`, **both parameters must be overridden**.

##### What a synthesizer does, and why it needs these parameters

A synthesizer has exactly one job: to find in the graph **what** a question can be asked about and **which text** it should be grounded in. It does not write questions — it prepares scenarios (the text appears in phase [4]).

`SingleHopSpecificQuerySynthesizer(property_name="keyphrases")` works like this:

1. it takes every chunk that has the `keyphrases` property filled in;
2. from each chunk it pulls the list of phrases — these are the **candidate themes** for a question (in the code the field is called `term`);
3. it creates scenarios for the selected themes and puts the chunk text itself into the future `reference_contexts`.

So `property_name` answers the question "where do I take the term the question is built around from". Get the name wrong and generation fails with `No nodes found with the 'entities' property`.

`MultiHopSpecificQuerySynthesizer(property_name="keyphrases", relation_type="keyphrases_overlap")` uses **both** parameters, at different steps:

1. **`relation_type`** — selects the edges: only pairs of chunks joined by an edge of that type are considered. If none are found — `No clusters found in the knowledge graph`;
2. from the edge it pulls `overlapped_items` — the phrases common to both chunks. Those are the themes;
3. **`property_name`** — validates the themes against the nodes: only chunks whose `node.properties["keyphrases"]` actually contains the theme make it into a scenario;
4. the text of both chunks goes into `reference_contexts` with `<1-hop>` and `<2-hop>` markers, which later make multi-hop rows easy to spot in the dataset.

In short: `relation_type` picks the **pairs of chunks**, `property_name` picks the **themes inside those chunks**.

`MultiHopAbstractQuerySynthesizer(relation_property="summary_similarity", abstract_property_name="themes")` is built differently, and and differs from what you'd expect in two ways:

1. **`relation_property`** — it looks for edges **not by `type`, but by the presence of a property** with that name in the edge's `properties` (`rel.get_property("summary_similarity")`). It is the only synthesizer that looks into `properties` rather than `type`;
2. it assembles clusters not from pairs but from **paths of up to 3 nodes**. In our case these edges connect documents, so a cluster is a group of topically close documents;
3. it then descends from the documents to their chunks along the `child` edges (created by the splitter) — it is the chunk text that ends up in `reference_contexts`;
4. **`abstract_property_name`** — on those chunks it reads the `themes` property and, through the `ConceptCombinationPrompt` prompt, asks the LLM to assemble a **combination of concepts** out of the themes of different documents that is worth asking a single shared question about.

Hence the requirements on the graph: `themes` on chunks (i.e. a `ThemesExtractor`) and edges carrying a `summary_similarity` property between documents. And there is a separate gotcha here: the library does ship a ready-made `SummaryCosineSimilarityBuilder`, but it writes a property named `summary_cosine_similarity` — one word longer than what the synthesizer looks for. So the name has to be set by hand via `new_property_name="summary_similarity"` on a plain `CosineSimilarityBuilder`.

### [3] and [4] Scenarios and question generation

The key idea that makes the dataset diverse: generation is **split into two phases**.

**Phase [3] — scenarios.** A scenario is not a question yet, it is a "problem statement": a set of nodes (one for single-hop, a pair for multi-hop) plus a theme, a persona, a style and a length. It is built like this: select the suitable nodes → compute how many questions are needed per node (`ceil(n / len(nodes))`) → match themes to personas via the LLM → build **all** `(node, theme, persona, style, length)` combinations → shuffle and pick the required number, trying not to repeat any "node + theme" pair.

Multi-hop does the same thing, but starts from pairs of nodes joined by an edge and takes its themes from the edge's `overlapped_items` — exactly those keyphrases that appear in both chunks. This guarantees that the question really is "about the intersection" rather than about two random texts.

**Phase [4] — samples, i.e. the questions themselves.** A sample, in ragas terminology, is one finished dataset row: `user_input` + `reference` + `reference_contexts`. Every scenario goes into the `QueryAnswerGenerationPrompt` prompt, and the LLM returns the structured output `{query, answer}`. The prompt hard-codes a **faithfulness-to-context** requirement: "Do not add any information not included in or inferable from the context". That is precisely why the generated `reference` works as ground truth — by construction it does not go beyond `reference_contexts`.

Why split it this way? It turns "generate 100 questions" into a controllable sampling problem: diversity across chunks, personas and styles is guaranteed **before** the LLM is ever called, rather than being left to the hope that the model will not produce 100 identical "What is X?" questions.

**About `testset_size`.** It is divided across `query_distribution` using `math.ceil`, and then rounded up once more inside the synthesizers. The final row count is **approximately** `testset_size`, usually slightly more.

---

## Assembling the pipeline

Now the same route as in the diagram, but in code. Everything below is our own code, the code you have to write; the full file is `full_pipeline.py`.

Mapping onto the diagram: steps 0–2 are block **[1]**, step 3 is block **[2]**, steps 4–5 configure blocks **[3]/[4]**, and step 6 is the run itself.

Let us start with all the imports we need:

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

Note the last line: `ThemesExtractor` (like `NERExtractor`) is not re-exported from `ragas.testset.transforms`, so it has to be imported from `extractors.llm_based` directly.

### 0. Documents

All ragas needs as input is a plain `list[str]`. We take three English Wikipedia articles through the HuggingFace dataset in streaming mode (so as not to download tens of gigabytes):

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

The articles were chosen to be non-political and long enough — it matters that a document is noticeably larger than `min_tokens`, otherwise the splitter will hand it back as a single piece.

### 1. LLM and embeddings

```python
openai_client = AsyncOpenAI(
    api_key=os.getenv("AI_TUNNEL_API_KEY"),
    base_url="https://api.aitunnel.ru/v1/",
)
llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

`llm_factory` returns a wrapper with structured output (via instructor), so every ragas prompt hands back ready Pydantic objects instead of text you have to parse. The client is injected from outside, which means any OpenAI-compatible provider will do — changing `base_url` is enough. `max_tokens=8192` is not a luxury here: the `reference` for multi-hop questions gets long, and with a small limit the answer is truncated and the structured output falls apart.

### 2. Building the graph

```python
def build_knowledge_graph(documents: list[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(Node(type=NodeType.DOCUMENT, properties={"page_content": doc}))
    return kg
```

No magic at all: the graph starts life as a list of document nodes with a single `page_content` property. Everything else appears as a result of the transformations.

### 3. Transformations

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

The important thing to see here is that the pipeline works **on two levels at once**, and the two must not be confused:

- **the chunk level** — `keyphrases` and `themes`, with `keyphrases_overlap` edges built on top of `keyphrases`. This feeds single-hop and multi-hop specific;
- **the document level** — `summary`, its embedding `summary_embedding`, and the cosine `summary_similarity` edges between documents. This feeds multi-hop abstract, which will then descend from documents to their chunks on its own.

That is also why every step has a `filter_nodes`: it does not just save LLM calls, it keeps each property on its own level of the graph.

#### Why `summary` is computed on documents rather than on chunks

It is tempting to make things simpler: compute the embeddings straight from the chunk text (`embed_property_name="page_content"`, `filter_nodes=_is_chunk`) and build the cosine edges between chunks. Don't — and here are four reasons, from the most important to the most boring.

**1. The cosine between chunks of the same document is always higher than between documents.** Every chunk of the TAI article is about atomic time, so their embeddings nearly coincide. That means the strongest cosine connections will be **inside** a single document, and the clusters will form there too. The abstract question then degenerates into "connect two paragraphs of the same article" — which is exactly what multi-hop specific already does through `keyphrases_overlap`. The document level provides the one thing no other synthesizer offers: a connection **between different documents**.

**2. The synthesizer is designed for documents and descends to the chunks itself.**

```python
# ragas/testset/synthesizers/multi_hop/abstract.py — library internals
for node in cluster:
    child_nodes = [rel.target for rel in child_relationships if rel.source == node]
    if child_nodes:
        nodes.extend(child_nodes)   # a cluster node → all of its chunks
    else:
        nodes.append(node)          # fallback: no children, so take the node itself
```

So clustering is meant to happen on documents, while their chunks are what land in `reference_contexts`. There is a pleasant side effect: the cluster receives the document's entire set of themes, giving `ConceptCombinationPrompt` plenty to choose a concept combination from. If the cluster holds chunks instead, the `else` branch fires (that branch exists for pre-chunked corpora — see `default_transforms_for_prechunked`), and there are only two or three chunks to choose from.

**3. The summary of a chunk is very nearly the chunk itself.** The `SummaryExtractorPrompt` prompt asks to "Summarize the given text in less than 10 sentences", while our chunk is 300–1000 tokens, i.e. roughly a dozen sentences already. No compression happens, the vector of such a summary is practically the vector of `page_content` — we pay an extra LLM call for something that comes for free. On a document of several thousand tokens the compression is real, and the summary vector genuinely is "the vector of the document's topic".

**4. Cost.** Three LLM calls and three embeddings instead of twelve (one per chunk). On a real corpus the difference is already tens of times.

There is a fifth reason, which shows up if you add `CustomNodeFilter` to the pipeline: it can read **only** the document summary. For a chunk it walks up to the parent via `get_parent_nodes` and reads `properties["summary"]` there, because the rubric evaluates "how well this chunk matches the document's topic" — for which the chunk's own summary is useless. Without a document-level `summary` the filter logs a warning and filters nothing.

> A caveat about very long documents: `SummaryExtractor.extract` cuts the text at `max_token_limit = 32000` and summarizes only the first piece. For Wikipedia articles that is not a problem, but for a book the summary will describe only its beginning.

### 4. Generator and personas

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

The personas are defined by hand — this way we do not pay for auto-generating them and do not depend on `summary` / `summary_embedding` being present in the graph.

### 5. Query type distribution

```python
def build_query_distribution(llm):
    return [
        (
            SingleHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases",   # the theme the question is built around
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
                relation_property="summary_similarity",  # edges from CosineSimilarityBuilder
                abstract_property_name="themes",         # property from ThemesExtractor
            ),
            0.2,
        ),
    ]
```

This is where the entire graph configuration comes together. Two chains are worth keeping in your head — one per level:

```
KeyphrasesExtractor(property_name="keyphrases")
        ↓  writes node.properties["keyphrases"] on chunks
OverlapScoreBuilder(property_name="keyphrases")
        ↓  creates Relationship(type="keyphrases_overlap", properties={... "overlapped_items": [...]})
MultiHopSpecificQuerySynthesizer(property_name="keyphrases", relation_type="keyphrases_overlap")
```

```
SummaryExtractor  →  ThemesExtractor(property_name="themes")
        ↓  writes summary on documents, themes on chunks
EmbeddingExtractor(embed_property_name="summary", property_name="summary_embedding")
        ↓  writes the summary vector on documents
CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity")
        ↓  creates Relationship(properties={"summary_similarity": 0.257})
MultiHopAbstractQuerySynthesizer(relation_property="summary_similarity", abstract_property_name="themes")
```

If even one link is out of sync you will get either `No nodes found with the 'entities' property`, or `No clusters found in the knowledge graph`, or `No relationships match the provided condition`. All three messages mean the same thing: **the synthesizer is looking in the graph for something you never put there**.

The weights `0.4 / 0.4 / 0.2` are shares of `testset_size`, not sampling probabilities: ragas simply divides the dataset size between the synthesizers.

### 6. Running it

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

The two steps — `apply_transforms` (build the graph) and `generator.generate` (generate the questions) — are deliberately kept apart. It is worth slipping a `kg.save("kg.json")` in between: the graph is expensive to build, but you can generate from it as many times as you like with different distributions.

The single most useful piece of diagnostics is printing how many edges of which kind you ended up with, **before** generation. For three articles the output looks like this:

```
KG before transforms: KnowledgeGraph(nodes: 3, relationships: 0)
KG after transforms: nodes=15 relationships=29
  rel child: 12                # document → its chunks (created by the splitter)
  rel next: 9                  # adjacent chunks within a document (splitter)
  rel summary_similarity: 1    # document ↔ document (CosineSimilarityBuilder)
  rel keyphrases_overlap: 7    # chunk ↔ chunk (OverlapScoreBuilder)
```

If there is a `0` next to `summary_similarity` or `keyphrases_overlap`, the corresponding synthesizer will fail and the `threshold` needs lowering. Checking that in seconds is cheaper than catching an exception three minutes into a run.

<small>A technical detail: `_share_event_loop_across_ragas_runs` monkey-patches `ragas.async_utils.run` and `ragas.executor.run` so that both phases run in a single event loop. Without it, an `AsyncOpenAI` client created outside the ragas loop runs into a closed loop on Windows. It has nothing to do with generation logic, but without such a shim the code crashes.</small>

### 7. The result

`testset.to_pandas()` gives a table with the following columns:

| column               | what's inside                                                           |
|----------------------|-------------------------------------------------------------------------|
| `user_input`         | the generated question                                                  |
| `reference_contexts` | the chunks it was generated from (with `<N-hop>` markers for multi-hop) |
| `reference`          | the reference answer, built **only** from those contexts                |
| `persona_name`       | the persona's name                                                      |
| `query_style`        | `MISSPELLED` / `PERFECT_GRAMMAR` / `POOR_GRAMMAR` / `WEB_SEARCH_LIKE`   |
| `query_length`       | `SHORT` / `MEDIUM` / `LONG`                                             |
| `synthesizer_name`   | which synthesizer produced the row                                      |

Example questions from a real run (3 articles, `testset_size=6` → 8 rows: 3 + 3 + 2):

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

Exactly what we wanted instead of "perfect" LLM questions: typos (`develpment`, `diffr`), broken grammar, short queries, and questions that physically cannot be answered from a single chunk.

The last example deserves a closer look, because it shows **what the summary cosine was needed for in the first place**. A `summary_similarity` edge connected the *Arithmetic mean* and *International Atomic Time* articles, the synthesizer descended to their chunks, found the shared notion of "weighted average" among the themes, and built a question that welds a definition from statistics to its application in metrology. Neither chunk answers such a question on its own, and `keyphrases_overlap` would never have found this pair — the articles are written in different vocabularies, with almost no literal phrase overlap between them. That is precisely the difference between *specific* (a shared term) and *abstract* (a shared idea).

Two things jump out in the CSV, and both are **quirks of ragas 0.4.3** rather than bugs in the pipeline:

1. Multi-hop rows have **empty `persona_name`, `query_style`, `query_length`**. The persona, style and length are present in the scenario and are passed into the prompt — but `MultiHopQuerySynthesizer._generate_sample` returns a `SingleTurnSample` with only `user_input` / `reference` / `reference_contexts` and does not carry those fields any further. For single-hop they are carried through.
2. The "speak as if you were this persona" instruction sometimes **leaks into the question text**: `"ok so as a student i wanna know in detail why the arithmetic mean is..."`. This follows from `role_description` being fed into the prompt as part of the conditions. The cure is more neutral persona descriptions, or customizing the prompt through `PromptMixin`.

### Checklist before running on your own data

- Documents are longer than `min_tokens`, otherwise there will be no chunks.
- `HeadlinesExtractor` before `HeadlineSplitter`, `SummaryExtractor` before `EmbeddingExtractor`.
- Every extractor and builder has `filter_nodes` set (`_is_chunk` / `_is_document`).
- The synthesizer's `property_name` matches what the extractor actually wrote (`entities` is the default!).
- `relation_type` matches the edge type the builder really created (`{property_name}_overlap`).
- For the abstract synthesizer: edges carry the property named exactly `summary_similarity`, and `threshold` was chosen from cosine values measured on your own corpus.
- The per-type edge counter is printed after `apply_transforms` — there should be no zeros.
- `kg.save(...)` after `apply_transforms`, so you do not pay for the graph twice.
- The resulting dataset has been **read by a human**. Synthetic data is a draft of your labels, not finished ground truth: rows with a leaked persona or an overly vague question are better dropped before people start comparing RAG configurations against them.

That concludes the first part: we have an `eval_dataset.csv` with `user_input`, `reference_contexts` and `reference` columns. In the second part we will plug the actual metrics into it — and see how to compare RAG configurations using this dataset.

___

# Evaluation metrics


## Context precision

