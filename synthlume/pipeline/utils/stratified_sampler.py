from langchain_core.documents import Document
from abc import ABC, abstractmethod
from synthlume.logging.logging import get_logger
from typing import Any
from copy import deepcopy

logger = get_logger(__name__)


class StratifiedSampler(ABC):
    """Stratified chunk sampling interface. Stratified chunk sampler relies on stratification method to group similar document chunks together.
    The sampling strategy is implemented by the subclass.

    Parameters:
    - documents: list of input document chunks
    """

    name: str = "generic_stratified_sampler"

    def __init__(
        self,
        documents: list[Document],
        **kwargs: Any,
    ):
        self.documents = (
            deepcopy(documents)  # List of input documents that we want to sample from
        )

    @abstractmethod
    def _stratification_method(
        self,
        **kwargs: Any,
    ) -> list[int]:
        """Cluster (Stratifiy) documents into groups of chunks

        Returns:
        - list of cluster indices
        """

    @abstractmethod
    def __iter__(self):
        """Iterator that yields the next chunk of documents based on the implemented sampling strategy"""
