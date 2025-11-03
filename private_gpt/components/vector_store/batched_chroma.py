from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.chroma import ChromaVectorStore  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Mapping


def chunk_list(
    lst: Sequence[BaseNode], max_chunk_size: int
) -> Generator[Sequence[BaseNode], None, None]:
    """Yield successive max_chunk_size-sized chunks from lst."""
    # keep simple and clear; range with step is fine
    n = len(lst)
    for i in range(0, n, max_chunk_size):
        yield lst[i : i + max_chunk_size]


class BatchedChromaVectorStore(ChromaVectorStore):  # type: ignore
    """Chroma vector store, batching additions to avoid reaching the max batch limit."""

    chroma_client: Any | None

    def __init__(
        self,
        chroma_client: Any,
        chroma_collection: Any,
        host: str | None = None,
        port: str | None = None,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        collection_kwargs: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(
            chroma_collection=chroma_collection,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            collection_kwargs=collection_kwargs or {},
        )
        self.chroma_client = chroma_client

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes to index, batching the insertion to avoid issues."""
        if not self.chroma_client:
            raise ValueError("Client not initialized")

        if not self._collection:
            raise ValueError("Collection not initialized")

        # cache frequently used attributes / functions locally for speed
        max_chunk_size = self.chroma_client.max_batch_size
        collection = self._collection
        node_to_md = node_to_metadata_dict
        flat_metadata = self.flat_metadata
        metadata_mode = MetadataMode.NONE

        all_ids: list[str] = []
        n = len(nodes)

        # inline chunking to avoid generator overhead
        for i in range(0, n, max_chunk_size):
            node_chunk = nodes[i : i + max_chunk_size]

            # list comprehensions are faster than repeated .append()
            embeddings = [node.get_embedding() for node in node_chunk]
            metadatas = [
                node_to_md(node, remove_text=True, flat_metadata=flat_metadata)
                for node in node_chunk
            ]
            ids = [node.node_id for node in node_chunk]
            documents = [
                node.get_content(metadata_mode=metadata_mode) for node in node_chunk
            ]

            collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )

            all_ids.extend(ids)

        return all_ids
