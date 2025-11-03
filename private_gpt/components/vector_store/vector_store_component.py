import logging
import typing
from functools import lru_cache

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def _doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> MetadataFilters:
    """
    Faster construction of filters: build list with comprehension instead of repeated append().
    Identical behavior to original.
    """
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        # list comprehension is marginally faster than a for-loop with append
        filters.filters = [MetadataFilter(key="doc_id", value=doc_id) for doc_id in context_filter.docs_ids]

    return filters


@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store: BasePydanticVectorStore
    _is_qdrant: bool

    @inject
    def __init__(self, settings: Settings) -> None:
        # cache local ref to avoid repeated attribute lookups
        self.settings = settings
        db = settings.vectorstore.database
        self._is_qdrant = db == "qdrant"

        match db:
            case "postgres":
                try:
                    from llama_index.vector_stores.postgres import (  # type: ignore
                        PGVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras vector-stores-postgres`"
                    ) from e

                if settings.postgres is None:
                    raise ValueError(
                        "Postgres settings not found. Please provide settings."
                    )

                # call model_dump once and reuse
                pg_params = settings.postgres.model_dump(exclude_none=True)
                self.vector_store = typing.cast(
                    BasePydanticVectorStore,
                    PGVectorStore.from_params(
                        **pg_params,
                        table_name="embeddings",
                        embed_dim=settings.embedding.embed_dim,
                    ),
                )

            case "chroma":
                try:
                    import chromadb  # type: ignore
                    from chromadb.config import (  # type: ignore
                        Settings as ChromaSettings,
                    )
                    from private_gpt.components.vector_store.batched_chroma import (
                        BatchedChromaVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "ChromaDB dependencies not found, install with `poetry install --extras vector-stores-chroma`"
                    ) from e

                # compute path once
                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_path = str((local_data_path / "chroma_db").absolute())
                chroma_client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chroma_settings,
                )
                chroma_collection = chroma_client.get_or_create_collection(
                    "make_this_parameterizable_per_api_call"
                )  # TODO

                self.vector_store = typing.cast(
                    BasePydanticVectorStore,
                    BatchedChromaVectorStore(
                        chroma_client=chroma_client, chroma_collection=chroma_collection
                    ),
                )

            case "qdrant":
                try:
                    from llama_index.vector_stores.qdrant import (  # type: ignore
                        QdrantVectorStore,
                    )
                    from qdrant_client import QdrantClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
                    ) from e

                if settings.qdrant is None:
                    logger.info(
                        "Qdrant config not found. Using default settings."
                        "Trying to connect to Qdrant at localhost:6333."
                    )
                    client = QdrantClient()
                else:
                    qdrant_params = settings.qdrant.model_dump(exclude_none=True)
                    client = QdrantClient(**qdrant_params)

                self.vector_store = typing.cast(
                    BasePydanticVectorStore,
                    QdrantVectorStore(
                        client=client,
                        collection_name="make_this_parameterizable_per_api_call",
                    ),
                )

            case "milvus":
                try:
                    from llama_index.vector_stores.milvus import (  # type: ignore
                        MilvusVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Milvus dependencies not found, install with `poetry install --extras vector-stores-milvus`"
                    ) from e

                if settings.milvus is None:
                    logger.info(
                        "Milvus config not found. Using default settings.\n"
                        "Trying to connect to Milvus at local_data/private_gpt/milvus/milvus_local.db "
                        "with collection 'make_this_parameterizable_per_api_call'."
                    )

                    self.vector_store = typing.cast(
                        BasePydanticVectorStore,
                        MilvusVectorStore(
                            dim=settings.embedding.embed_dim,
                            collection_name="make_this_parameterizable_per_api_call",
                            overwrite=True,
                        ),
                    )

                else:
                    milvus = settings.milvus
                    self.vector_store = typing.cast(
                        BasePydanticVectorStore,
                        MilvusVectorStore(
                            dim=settings.embedding.embed_dim,
                            uri=milvus.uri,
                            token=milvus.token,
                            collection_name=milvus.collection_name,
                            overwrite=milvus.overwrite,
                        ),
                    )

            case "clickhouse":
                try:
                    from clickhouse_connect import (  # type: ignore
                        get_client,
                    )
                    from llama_index.vector_stores.clickhouse import (  # type: ignore
                        ClickHouseVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "ClickHouse dependencies not found, install with `poetry install --extras vector-stores-clickhouse`"
                    ) from e

                if settings.clickhouse is None:
                    raise ValueError(
                        "ClickHouse settings not found. Please provide settings."
                    )

                clickhouse_client = get_client(
                    host=settings.clickhouse.host,
                    port=settings.clickhouse.port,
                    username=settings.clickhouse.username,
                    password=settings.clickhouse.password,
                )
                self.vector_store = ClickHouseVectorStore(
                    clickhouse_client=clickhouse_client
                )
            case _:
                raise ValueError(
                    f"Vectorstore database {db} not supported"
                )

    def get_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:
        """
        Use cached boolean self._is_qdrant instead of repeated string comparisons.
        Build the filters only when needed.
        """
        # localize for tiny speed-up
        is_qdrant = self._is_qdrant

        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter) if not is_qdrant else None
            ),
        )

    def close(self) -> None:
        # safer and slightly cheaper: get client once
        client = getattr(self.vector_store, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
