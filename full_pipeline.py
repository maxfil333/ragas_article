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

# load dataset from huggingface

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"


def load_hf_documents() -> list[str]:
    """Load full Wikipedia articles by title (streaming, stops when all found)."""

    doc_titles = [
        "International Atomic Time",
        "Agricultural science",
        "Arithmetic mean",
    ]

    found: dict[str, str] = {}

    stream = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    for row in stream:
        title = row["title"]
        if title in doc_titles:
            found[title] = row["text"]
            if len(found) == len(doc_titles):
                break

    missing = set(doc_titles) - found.keys()
    if missing:
        raise RuntimeError(f"Titles not found in {HF_DATASET}/{HF_CONFIG}: {sorted(missing)}")

    return [found[title] for title in doc_titles]


def _share_event_loop_across_ragas_runs(runner: asyncio.Runner) -> None:
    """Keep one event loop for transforms + generation (Windows / AsyncOpenAI)."""

    def run_on_shared_loop(async_func, allow_nest_asyncio: bool = True):
        coro = async_func() if callable(async_func) else async_func
        return runner.run(coro)

    ragas_async_utils.run = run_on_shared_loop
    ragas_executor.run = run_on_shared_loop


def _count_rel_types(kg: KnowledgeGraph) -> dict[str, int]:
    """Helper function for displaying logs."""
    counts: dict[str, int] = {}
    for rel in kg.relationships:
        counts[rel.type] = counts.get(rel.type, 0) + 1
    return counts


def build_knowledge_graph(documents: list[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc},
            )
        )
    return kg


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


def build_query_distribution(llm):
    return [
        (
            SingleHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases"  # Term / keyword / NER entity for which the question will be generated
            ),
            0.4,
        ),
        (
            MultiHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases",
                relation_type="keyphrases_overlap",
                # from OverlapScoreBuilder class:
                # relation.type = property_name + "_overlap"
                # "keyphrases" + "_overlap" = keyphrases_overlap
            ),
            0.4,
        ),
        (
            MultiHopAbstractQuerySynthesizer(
                llm=llm,
                relation_property="summary_similarity",  # relations built by CosineSimilarityBuilder
                abstract_property_name="themes",  # node property filled by ThemesExtractor
            ),
            0.2,
        ),
    ]


def create_testset(docs: list[str]) -> None:
    start = perf_counter()

    openai_client = AsyncOpenAI(
        api_key=os.getenv("AI_TUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/",
    )
    llm_kwargs = {"max_tokens": 8192}

    llm = llm_factory("gpt-5-mini", client=openai_client, **llm_kwargs)
    embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

    kg = build_knowledge_graph(docs)
    print(f"KG before transforms: {kg}")

    transforms = build_transforms(llm, embeddings)
    generator = build_generator(llm, embeddings, kg)

    with asyncio.Runner() as runner:
        _share_event_loop_across_ragas_runs(runner)
        apply_transforms(kg, transforms=transforms)
        print(
            f"KG after transforms: nodes={len(kg.nodes)} "
            f"relationships={len(kg.relationships)}"
        )
        for rel_type, count in _count_rel_types(kg).items():
            print(f"  rel {rel_type}: {count}")

        testset = generator.generate(
            testset_size=6,
            query_distribution=build_query_distribution(llm),
        )

    out_csv = "eval_dataset.csv"
    testset.to_pandas().to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(testset)} samples)")
    print(f"Elapsed: {perf_counter() - start:.1f}s")


if __name__ == "__main__":
    docs = load_hf_documents()
    create_testset(docs)
