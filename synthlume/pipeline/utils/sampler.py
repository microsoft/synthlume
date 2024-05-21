import os
from copy import deepcopy

from langchain_core.documents import Document
from abc import ABC
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)

class ChunkSampling(ABC):
    """Chunk sampling for document chunks interface
    
    Parameters:
    - documents: list of input document chunks 
    - sample_size: number of chunks to be sampled
    """
    name: str = "generic sampler"

    def __init__(
        self,
        documents: list[Document],
        sample_size: int, #number of chunks to be sampled
    ):
        self.sample_size = sample_size
        self.documents = documents[:sample_size]

    def __iter__(self):
        for i in range(len(self.documents)):
            yield self.documents[i]