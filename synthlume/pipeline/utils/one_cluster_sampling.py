from langchain_core.embeddings import Embeddings
from typing import Optional
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger
from synthlume.pipeline.utils.stratified_sampler import StratifiedSampler
import random

logger = get_logger(__name__)


class OneClusterSampling(StratifiedSampler):
    """Chunk sampling for document chunks

    Parameters:
    - documents: list of input document chunks
    - sample_size: number of chunks to be sampled

    """

    name: str = "random_sampling"

    def __init__(
        self,
        documents: list[Document],
        sample_size: int,  # number of chunks to be sampled
    ):
        super().__init__(documents)
        self.sample_size = sample_size

        cluster_labels = self._stratification_method()

        self._shuffle_documents(cluster_labels)

    def _stratification_method(self) -> list[int]:
        logger.info("Assigning all document chunks into one cluster")
        return [0] * len(self.documents)

    def _shuffle_documents(self, cluster_labels: list[int]) -> list[Document]:
        """Shuffle documents"""
        # shuffle documents based on cluster labels
        clustered_docs = []
        for i in range(max(cluster_labels) + 1):
            indexes = [idx for idx, x in enumerate(cluster_labels) if x == i]
            cluster = [self.documents[i] for i in indexes]
            random.shuffle(cluster)
            cluster_sample = cluster[
                : min(len(cluster), self.sample_size // (max(cluster_labels) + 1))
            ]
            clustered_docs.append(cluster_sample)

        self.documents = clustered_docs

    def __iter__(self):
        for i in range(self.sample_size // len(self.documents)):
            if i < len(self.documents[0]):
                yield self.documents[0][i]
