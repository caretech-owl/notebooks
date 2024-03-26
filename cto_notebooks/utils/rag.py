from typing import Dict, Iterable, List, Protocol, Tuple, Union

import numpy as np
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings


class VectorStore(Protocol):
    def merge_from(self, vector_store: "VectorStore") -> None:
        pass

    def save_local(self, path: str) -> None:
        pass

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> List[str]:
        pass

    def search(self, query: str, search_type: str, k: int) -> List[Document]:
        pass

    def similarity_search_by_vector(
        self, vector: List[float], k: int
    ) -> List[Document]:
        pass


def average_vector(
    field: str,
    embeddings: HuggingFaceEmbeddings,
    data_dict: Iterable[Tuple[Dict[str, Union[str, List[str]]], VectorStore]],
    initial_query: str = "",
    k: int = 10,
) -> List[float]:
    vecs = []
    for dict, store in data_dict:
        values = dict[field]
        if isinstance(values, str):
            values = [values]
        for vec in store.search(
            f"{initial_query} {', '.join(values)}",
            search_type="similarity",
            k=k,
        ):
            if not any(val in vec.page_content for val in values):
                continue
            vecs.append(embeddings.embed_query(vec.page_content))
            break
        embeddings.embed_query(vec.page_content)
    return np.mean(vecs, axis=0)
