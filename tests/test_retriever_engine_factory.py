from rag.retrieval.engine import RetrieverEngine
from rag.retrieval.plugins.dense import DenseRetrievalPlugin


def test_retriever_engine_from_name():
    # Must use a REAL registered embedding plugin → "cohere"
    embed_cfg = type("Cfg", (), {
        "plugin_name": "cohere",
        "api_key": "k",
        "model": "m",
        "output_dimension": None,
    })

    retriever_cfg = type("Cfg", (), {
        "collection": "col",
        "top_k": 2,
    })

    engine = RetrieverEngine.from_name(
        "dense",
        client="dummy",
        embed_cfg=embed_cfg,
        retriever_cfg=retriever_cfg,
        rerank_cfg=None,
    )

    assert isinstance(engine.plugin, DenseRetrievalPlugin)
    assert engine.plugin.client == "dummy"
