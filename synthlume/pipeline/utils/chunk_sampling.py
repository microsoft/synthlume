import os
from copy import deepcopy

from langchain_core.embeddings import Embeddings
from typing import Optional
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger
import random

logger = get_logger(__name__)

#TODO: Implement GMM and Spectral Clustering
class ChunkSampling():
    """Chunk sampling for document chunks 
    
    Parameters:
    - documents: list of input document chunks 
    - sample_size: number of chunks to be sampled
    - sampling_method: method used for sampling chunks, possible choices: random, Gaussian Mixture Models, k-NN graph partitioning approximated with Spectral Clustering
    - embeddings: embeddings to be used for clustering
    - dimensionality_reduction: method used for dimensionality reduction that can be applied prior to clustering, possible choices: UMAP, Laplacian Eigenvectors of the k-NN graph
    - knn_dim: knn parameter for umap or k-NN graph 
    - dim: dimension for dimentionality reduction 
    - n_clusters: number of clusters, if value='None' BIC is used to determined optimal number of clustering

    """
    name: str = "chunk_sampling"

    def __init__(
        self,
        documents: list[Document],
        sample_size: int, #number of chunks to be sampled
        sampling_method: str = "random", #random, gmm, spectral_clustering
        embeddings: Optional[Embeddings] = None, #embeddings to be used for clustering
        dimensionality_reduction: Optional[str] = None, #umap, spectral_embeddings
        knn_dim: Optional[int] = None, #knn parameter for umap or spectral_embeddings
        dim: Optional[int] = None, #dimension for dimentionality reduction for umap or spectral_embeddings
        n_clusters: Optional[int] = None, #number of clusters, if value='None' BIC is used to determined optimal number of clustering
    ):
        super().__init__(documents, sample_size)

        self.documents = (
            self._shuffle_documents(documents, sample_size)
            if sampling_method == "random"
            else documents
        )        
        self.embeddings = embeddings
        self.sampling_method = sampling_method
        self.dimensionality_reduction = dimensionality_reduction
        self.knn_dim = knn_dim
        self.dim = dim
        self.n_clusters = n_clusters

    def _shuffle_documents(self, documents: list[Document], sample_size: int)->list[Document]:
        """Shuffle documents"""
        return random.shuffle(documents)[:sample_size]

    def __iter__(self):
        for i in range(len(self.documents)):
            yield self.documents[i]