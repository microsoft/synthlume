from langchain_core.embeddings import Embeddings
from typing import Optional
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger
from synthlume.pipeline.utils.stratified_sampler import StratifiedSampler
import random

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.faiss import FAISS
import numpy as np
from sklearn.mixture import GaussianMixture

import faiss

logger = get_logger(__name__)


class KMeansSampling(StratifiedSampler):
    """Chunk sampling for document chunks

    Parameters:
    - documents: list of input document chunks
    - sample_size: number of chunks to be sampled
    - documents_exclude: clusters that contain any of the document chunks in this list will be excluded
    - documents_focus: clusters that do not contain document chunks in this list will be excluded
    - embeddings: embeddings to be used for clustering
    - n_clusters: list of possible values for number of clusters
    - n_clusters_criterion: criterion used to determine optimal number of clusters. Available options are BIC, eigen_gap
    - sample_size_for_BIC: sample size used for BIC computation
    - visualize_results: boolean variable that determines whether to visualize clustering results
    - sampling_method: string variable that determines how the iterator will sample the elements of the different groups
    - random_state: random seed

    """

    name: str = "kmeans_sampling"

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
        n_clusters: Optional[list[int]] = [
            100
        ],  # list of possible values for number of clusters
        n_clusters_criterion: Optional[
            str
        ] = "eigen_gap",  # criterion used to determine optimal number of clusters. Available options are BIC, eigen_gap
        sample_size_for_BIC: Optional[int] = -1,  # sample size used for BIC computation
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
            n_clusters,
            n_clusters_criterion,
            sample_size_for_BIC,
            visualize_results,
        )

        self._shuffle_documents(cluster_labels, documents_focus, documents_exclude)

    def _stratification_method(
        self,
        embeddings: Embeddings,
        n_clusters: list[int],
        n_clusters_criterion: str,
        sample_size_for_BIC: int = -1,
        visualize_results: bool = False,
    ) -> list[int]:
        logger.info("Using k-means for clustering the document chunks")
        # initially create vector store using the embeddings
        vectorstore = FAISS.from_documents(
            self.documents, embeddings, distance_strategy=DistanceStrategy.COSINE
        )
        num_docs = vectorstore.index.ntotal
        logger.info("Total number of chunks is %d", num_docs)
        embedding_dimension = vectorstore.index.d
        doc_embeddings = faiss.rev_swig_ptr(
            vectorstore.index.get_xb(), num_docs * embedding_dimension
        ).reshape(num_docs, embedding_dimension)
        if len(n_clusters) > 1 and n_clusters_criterion == "BIC":
            bic_values = []
            for n in n_clusters:
                gm = GaussianMixture(n_components=n, random_state=self.random_state)
                if sample_size_for_BIC > 0:
                    sample = random.sample(
                        range(num_docs), min(sample_size_for_BIC, num_docs)
                    )
                    gm.fit(doc_embeddings[sample, :])
                else:
                    gm.fit(doc_embeddings)
                bic_values.append(gm.bic(doc_embeddings))
                logger.info(
                    "BIC value for candidate number of clusters %d is %f",
                    n,
                    bic_values[-1],
                )
            best_k = n_clusters[np.argmin(bic_values)]
            logger.info(
                "Optimal number of clusters based on BIC criterion is %d", best_k
            )
        elif len(n_clusters) > 1 and n_clusters_criterion == "eigen_gap":
            mat = faiss.PCAMatrix(embedding_dimension, max(n_clusters) + 1)
            mat.train(doc_embeddings)
            # from the range of available number of cluster options, select the one with the largest eigengap
            the_eigs = faiss.vector_to_array(mat.eigenvalues)
            the_gaps = [
                the_eigs[n_clusters[i] - 2] - the_eigs[n_clusters[i] - 1]
                for i in range(len(n_clusters))
            ]
            best_k = max(2, n_clusters[the_gaps.index(max(the_gaps))])
            logger.info(
                "Optimal number of clusters based on eigengap heuristic is %d", best_k
            )
        else:
            best_k = n_clusters[0]
            if len(n_clusters) > 1:
                logger.warning(
                    "Invalid criterion for selecting optimal number of clusters, using %d clusters",
                    best_k,
                )

        # now apply k-means with best_k
        kmeans = faiss.Kmeans(embedding_dimension, best_k)
        kmeans.train(doc_embeddings)

        # assign cluster labels
        _, cluster_labels = kmeans.index.search(doc_embeddings, 1)
        labels_list = [x[0] for x in cluster_labels]
        if visualize_results:
            # TODO: implement visualization of BIC values or eigengap scores, as well as tSNE embeddings of clustering results
            pass

        # fix order of labels based on input document list
        labels_list_order_fix = labels_list
        for i in range(len(labels_list)):
            _id = vectorstore.index_to_docstore_id[i]
            doc = vectorstore.docstore.search(_id)
            idx_fix = self.documents.index(doc)
            self.documents[idx_fix].metadata["cluster_id"] = labels_list[i]
            labels_list_order_fix[idx_fix] = labels_list[i]

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
