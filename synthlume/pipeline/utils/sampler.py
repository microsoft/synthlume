from langchain_core.documents import Document
from abc import ABC, abstractmethod
from synthlume.logging.logging import get_logger
from typing import Any

logger = get_logger(__name__)

class Sampler(ABC):
    """Chunk sampling interface. Chunk sampler relies on document clustering to group similar documents together. 
    The sampling strategy is implemented by the subclass.
    
    Parameters:
    - documents: list of input document chunks 
    """
    name: str = "generic sampler"

    def __init__(
        self,
        documents: list[Document],
        **kwargs: Any,
    ):
        self.documents = [documents] #Each nested list represents a cluster of document chunks. Initializaing with a single cluster
    
    @abstractmethod
    def _clustering(
        self, 
        **kwargs: Any,
    ) -> list[int]:
        """Cluster documents into chunks
        
        Returns:
        - list of cluster indices
        """

    @abstractmethod
    def __iter__(self):
        """Iterator that yields the next chunk of documents based on the implemented sampling strategy"""
        