from langchain_core.embeddings import Embeddings
from typing import Optional
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger
from synthlume.pipeline.utils.stratified_sampler import StratifiedSampler
import random

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.faiss import FAISS
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import SpectralClustering
import numpy as np
from scipy.sparse.linalg import eigsh

import faiss

logger = get_logger(__name__)


class SpectralSampling(StratifiedSampler):
    """Chunk sampling for document chunks

    Parameters:
    - documents: list of input document chunks
    - sample_size: number of chunks to be sampled
    - documents_exclude: clusters that contain any of the document chunks in this list will be excluded
    - documents_focus: clusters that do not contain document chunks in this list will be excluded
    - embeddings: embeddings to be used for clustering
    - knn_dim: list of candidate nearest neighbors to be considered for k-NN graph creation. Used for spectral_clustering method
    - n_clusters: list of possible values for number of clusters
    - n_clusters_criterion: criterion used to determine optimal number of clusters. Available options are BIC, eigen_gap
    - sample_size_for_BIC: sample size used for BIC computation
    - visualize_results: boolean variable that determines whether to visualize clustering results
    - sampling_method: string variable that determines how the iterator will sample the elements of the different groups
    - random_state: random seed

    """

    name: str = "spectral_sampling"

    def __init__(
        self,
        documents: list[Document],
        sample_size: int,  # number of chunks to be sampled
        documents_exclude: Optional[
            list[Document]
        ] = None,  # clusters that contain any of the document chunks in this list will be excluded
        documents_focus: Optional[
            list[Document]
        ] = None,  # clusters that do not contain document chunks in this list will be excluded
        embeddings: Optional[Embeddings] = None,  # embeddings to be used for clustering
        knn_dim: Optional[list[int]] = [
            100
        ],  # list of candidate nearest neighbors to be considered for k-NN graph creation. Used for spectral_clustering method
        n_clusters: Optional[list[int]] = [
            100
        ],  # list of possible values for number of clusters
        n_clusters_criterion: Optional[
            str
        ] = "eigen_gap",  # criterion used to determine optimal number of clusters. Available option is eigen_gap
        visualize_results: Optional[bool] = False,  # visualize clustering results
        sampling_method: Optional[
            str
        ] = "stratified_sequence",  # iterator samples different groups sequencially (aka sample_1 comes from group_1, sample_2 come from group_2, etc.). Available other option is 'cluster_sequence' that initially samples all elements of group_1, then moves to group_2 etc.
        random_state: Optional[int] = 42,  # random state
    ):
        super().__init__(documents)
        self.sample_size = sample_size
        self.random_state = random_state
        self.sampling_method = sampling_method
        cluster_labels = self._stratification_method(
            embeddings,
            knn_dim,
            n_clusters,
            n_clusters_criterion,
            visualize_results,
        )

        self._shuffle_documents(cluster_labels, documents_focus, documents_exclude)

    def _stratification_method(
        self,
        embeddings: Embeddings,
        knn_dim: list[int],
        n_clusters: list[int],
        n_clusters_criterion: str,
        visualize_results: bool = False,
    ) -> list[int]:
        """Cluster documents into chunks"""
        logger.info("Using spectral clustering to cluster the document chunks")
        # initially create vector store using the embeddings
        vectorstore = FAISS.from_documents(
            self.documents, embeddings, distance_strategy=DistanceStrategy.COSINE
        )

        num_docs = vectorstore.index.ntotal
        embedding_dimension = vectorstore.index.d
        doc_embeddings = faiss.rev_swig_ptr(
            vectorstore.index.get_xb(), num_docs * embedding_dimension
        ).reshape(num_docs, embedding_dimension)
        logger.info("Total number of chunks is %d", num_docs)
        _, indices = vectorstore.index.search(doc_embeddings, max(knn_dim))
        k_nn_graph = indices

        if (
            len(knn_dim) > 1 or len(n_clusters) > 1
        ) and n_clusters_criterion == "eigen_gap":
            best_gaps = []
            ncl_candidates = []
            for k_dim in knn_dim:
                row_indeces = [[i] * k_dim for i in range(len(k_nn_graph))]
                flat_row_indeces = [item for sublist in row_indeces for item in sublist]

                col_indeces = [k_nn_graph[i][:k_dim] for i in range(len(k_nn_graph))]
                flat_col_indeces = [item for sublist in col_indeces for item in sublist]

                row = np.array(flat_row_indeces)
                col = np.array(flat_col_indeces)
                data = np.ones((len(flat_row_indeces),))

                affinity_matrix = csr_matrix(
                    (data, (row, col)), shape=(len(k_nn_graph), len(k_nn_graph))
                )
                affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.transpose())
                D = diags(1 / np.sqrt(affinity_matrix.sum(axis=1).data).flatten())
                Laplacian = D.dot(affinity_matrix).dot(D)
                eigs = eigsh(Laplacian, max(n_clusters) + 1, return_eigenvectors=False)
                eigs = np.flip(eigs)
                the_gaps = [
                    eigs[n_clusters[i] - 1] - eigs[n_clusters[i]]
                    for i in range(len(n_clusters))
                ]
                best_gaps.append(max(the_gaps))
                ncl_candidates.append(n_clusters[the_gaps.index(max(the_gaps))])
                logger.info(
                    "Optimal number of clusters for k=%d based on eigengap heuristic is %d",
                    k_dim,
                    ncl_candidates[-1],
                )

            best_k = knn_dim[best_gaps.index(max(best_gaps))]
            best_ncl = ncl_candidates[best_gaps.index(max(best_gaps))]
            logger.info(
                "Optimal clustering parameters are k=%d and number of clusters=%d",
                best_k,
                best_ncl,
            )
        else:
            if len(knn_dim) > 1 or len(n_clusters) > 1:
                logger.warning(
                    "Invalid criterion for selecting optimal cluster parameters, using %d clusters and %d for k-nn graph",
                    n_clusters[0],
                    knn_dim[0],
                )

            best_k = knn_dim[0]
            best_ncl = n_clusters[0]

        row_indeces = [[i] * best_k for i in range(len(k_nn_graph))]
        flat_row_indeces = [item for sublist in row_indeces for item in sublist]

        col_indeces = [k_nn_graph[i][:best_k] for i in range(len(k_nn_graph))]
        flat_col_indeces = [item for sublist in col_indeces for item in sublist]

        row = np.array(flat_row_indeces)
        col = np.array(flat_col_indeces)
        data = np.ones(
            (
                len(
                    flat_col_indeces,
                )
            )
        )
        affinity_matrix = csr_matrix(
            (data, (row, col)), shape=(len(k_nn_graph), len(k_nn_graph))
        )

        labels_list = (
            SpectralClustering(
                n_clusters=best_ncl,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=best_k,
                assign_labels="discretize",
                random_state=self.random_state,
            )
            .fit(affinity_matrix)
            .labels_
        )

        # fix order of labels based on input document list
        labels_list_order_fix = labels_list
        for i in range(len(labels_list)):
            _id = vectorstore.index_to_docstore_id[i]
            doc = vectorstore.docstore.search(_id)
            idx_fix = self.documents.index(doc)
            self.documents[idx_fix].metadata["cluster_id"] = labels_list[i]
            labels_list_order_fix[idx_fix] = labels_list[i]

        if visualize_results:
            # TODO: implement visualization of BIC values or eigengap scores, as well as tSNE embeddings of clustering results
            pass

        return labels_list_order_fix

    def _shuffle_documents(
        self,
        cluster_labels: list[int],
        documents_focus: list[Document],
        documents_exclude: list[Document],
    ) -> list[Document]:
        """Shuffle documents"""
        # shuffle documents based on cluster labels
        clustered_docs = []
        for i in range(max(cluster_labels) + 1):
            indexes = [idx for idx, x in enumerate(cluster_labels) if x == i]
            cluster = [self.documents[i] for i in indexes]
            random.shuffle(cluster)
            to_append_cluster = True
            if documents_focus is not None:
                # only consider document clusters that contain documents in the focus list
                if not any(x in cluster for x in documents_focus):
                    to_append_cluster = False
                    logger.info(
                        "Cluster %d excluded since it does not contain focus documents ",
                        i,
                    )
            if documents_exclude is not None:
                # exclude document clusters that contain documents  in the exclude list
                if any(x in cluster for x in documents_exclude):
                    to_append_cluster = False
                    logger.info(
                        "Cluster %d excluded since it contains excluded documents ", i
                    )

            if to_append_cluster:
                cluster_sample = cluster[
                    : min(len(cluster), self.sample_size // (max(cluster_labels) + 1))
                ]
                clustered_docs.append(cluster_sample)
        self.documents = clustered_docs

    def __iter__(self):
        if self.sampling_method == "stratified_sequence":
            for i in range(self.sample_size // len(self.documents)):
                for j in range(len(self.documents)):
                    if i < len(self.documents[j]):
                        yield self.documents[j][i]
        elif self.sampling_method == "cluster_sequence":
            for j in range(len(self.documents)):
                for i in range(self.sample_size // len(self.documents)):
                    if i < len(self.documents[j]):
                        yield self.documents[j][i]
        else:
            logger.warning(
                "Invalid sampling method provided. Available sampling method options are ['stratified_sequence', 'cluster_sequence']. Defaulting to 'stratified_sequence'"
            )
            for i in range(self.sample_size // len(self.documents)):
                for j in range(len(self.documents)):
                    if i < len(self.documents[j]):
                        yield self.documents[j][i]
