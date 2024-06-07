from copy import deepcopy

from langchain_core.embeddings import Embeddings
from langchain.chat_models import AzureChatOpenAI
from typing import Optional
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger
from synthlume.pipeline.utils.sampler import Sampler
import random

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.faiss import FAISS
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import SpectralClustering  
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import eigsh
from langchain.chains.summarize import load_summarize_chain

import faiss

logger = get_logger(__name__)


class ChunkSampling(Sampler):
    """Chunk sampling for document chunks 
    
    Parameters:
    - documents: list of input document chunks 
    - sample_size: number of chunks to be sampled
    - documents_exclude: clusters that contain any of the document chunks in this list will be excluded
    - documents_focus: clusters that do not contain document chunks in this list will be excluded
    - embeddings: embeddings to be used for clustering
    - llm: llm used for cluster summarization
    - sampling_method: random, spectral_clustering, kmeans
    - knn_dim: list of candidate nearest neighbors to be considered for k-NN graph creation. Used for spectral_clustering method
    - n_clusters: list of possible values for number of clusters
    - n_clusters_criterion: criterion used to determine optimal number of clusters. Available options are BIC, eigen_gap
    - sample_size_for_BIC: sample size used for BIC computation
    - visualize_results: boolean variable that determines whether to visualize clustering results
    - random_state: random seed

    """
    name: str = "chunk_sampling"

    def __init__(
        self,
        documents: list[Document],
        sample_size: int, #number of chunks to be sampled
        documents_exclude: Optional[list[Document]] = None, #clusters that contain any of the document chunks in this list will be excluded
        documents_focus: Optional[list[Document]] = None, #clusters that do not contain document chunks in this list will be excluded
        embeddings: Optional[Embeddings] = None, #embeddings to be used for clustering
        llm: Optional[AzureChatOpenAI] = None, #LLM used for cluster summarization
        limit_summarization: Optional[int] = 40, #number of chunks used to provide cluster summaries
        sampling_method: str = "random", #random, spectral_clustering, kmeans
        knn_dim: Optional[list[int]] = [100], #list of candidate nearest neighbors to be considered for k-NN graph creation. Used for spectral_clustering method
        n_clusters: Optional[list[int]] = [100], #list of possible values for number of clusters
        n_clusters_criterion: Optional[str] = 'eigen_gap', #criterion used to determine optimal number of clusters. Available options are BIC, eigen_gap
        sample_size_for_BIC: Optional[int] = -1, #sample size used for BIC computation
        visualize_results: Optional[bool] = False, #visualize clustering results
        random_state: Optional[int] = 42 #random state
    ):
        super().__init__(documents)
        self.sample_size = sample_size
        self.random_state = random_state

        if sampling_method == "spectral_clustering" or sampling_method == "kmeans":
            cluster_labels = self._clustering(sampling_method, embeddings, knn_dim, n_clusters, n_clusters_criterion, sample_size_for_BIC, visualize_results)    
        
        self._shuffle_and_summarize_documents(cluster_labels, documents_focus, documents_exclude, llm, limit_summarization)

    def _clustering(self, sampling_method:str, embeddings:Embeddings, knn_dim:list[int], n_clusters:list[int], n_clusters_criterion:str, sample_size_for_BIC:int = -1, visualize_results:bool = False) -> list[int]:
        """Cluster documents into chunks"""
        if sampling_method == "spectral_clustering":
            logger.info("Using spectral clustering to cluster the document chunks")
            return self._spectral_clustering(embeddings, knn_dim, n_clusters, n_clusters_criterion, visualize_results)
        elif sampling_method == "kmeans":
            logger.info("Using k-means for clustering the document chunks")
            return self._kmeans(embeddings, n_clusters, n_clusters_criterion, sample_size_for_BIC, visualize_results)
        
    def _kmeans(self, embeddings:Embeddings, n_clusters:list[int], n_clusters_criterion:str, sample_size_for_BIC:int = -1, visualize_results:bool = False)->list[int]:
        """Compute clustering structure for documents"""
        #initially create vector store using the embeddings 
        vectorstore = FAISS.from_documents(
            self.documents[0], embeddings, distance_strategy=DistanceStrategy.COSINE
            )
        num_docs = vectorstore.index.ntotal
        logger.info("Total number of chunks is %d", num_docs)
        embedding_dimension = vectorstore.index.d
        doc_embeddings = faiss.rev_swig_ptr(vectorstore.index.get_xb(), num_docs*embedding_dimension).reshape(num_docs, embedding_dimension)
        if len(n_clusters)>1 and n_clusters_criterion == 'BIC':
            bic_values = []
            for n in n_clusters:
                gm = GaussianMixture(n_components=n, random_state=self.random_state)
                if sample_size_for_BIC > 0:
                    sample = random.sample(range(num_docs), min(sample_size_for_BIC, num_docs))
                    gm.fit(doc_embeddings[sample,:])
                else:
                    gm.fit(doc_embeddings)
                bic_values.append(gm.bic(doc_embeddings))
                logger.info("BIC value for candidate number of clusters %d is %f", n, bic_values[-1])
            best_k = n_clusters[np.argmin(bic_values)]
            logger.info("Optimal number of clusters based on BIC criterion is %d", best_k)
        elif len(n_clusters)>1 and n_clusters_criterion == 'eigen_gap':
            mat = faiss.PCAMatrix(embedding_dimension, max(n_clusters)+1)
            mat.train(doc_embeddings)
            #from the range of available number of cluster options, select the one with the largest eigengap
            the_eigs = faiss.vector_to_array(mat.eigenvalues)
            the_gaps = [the_eigs[n_clusters[i]-2] - the_eigs[n_clusters[i]-1] for i in range(len(n_clusters))]
            best_k = max(2, n_clusters[the_gaps.index(max(the_gaps))])
            logger.info("Optimal number of clusters based on eigengap heuristic is %d", best_k)
        else:
            best_k = n_clusters[0]
            if len(n_clusters)>1:
                logger.warning("Invalid criterion for selecting optimal number of clusters, using %d clusters", best_k)

        #now apply k-means with best_k
        kmeans = faiss.Kmeans(embedding_dimension, best_k)
        kmeans.train(doc_embeddings)

        #assign cluster labels
        _, cluster_labels = kmeans.index.search(doc_embeddings, 1)
        labels_list = [x[0] for x in cluster_labels]
        if visualize_results:
            #TODO: implement visualization of BIC values or eigengap scores, as well as tSNE embeddings of clustering results 
            pass

        #fix order of labels based on input document list
        labels_list_order_fix = labels_list
        for i in range(len(labels_list)):
            _id = vectorstore.index_to_docstore_id[i]
            doc = vectorstore.docstore.search(_id)
            idx_fix = self.documents[0].index(doc)
            self.documents[0][idx_fix].metadata['cluster_id'] = labels_list[i] 
            labels_list_order_fix[idx_fix] = labels_list[i]         
      
        return labels_list
        
    def _spectral_clustering(self, embeddings:Embeddings, knn_dim:list[int], n_clusters:list[int], n_clusters_criterion:str, visualize_results:bool = False)->list[int]:
        """Compute clustering structure for documents"""
        #initially create vector store using the embeddings 
        vectorstore = FAISS.from_documents(
            self.documents[0], embeddings, distance_strategy=DistanceStrategy.COSINE
            )

        num_docs = vectorstore.index.ntotal
        embedding_dimension = vectorstore.index.d
        doc_embeddings = faiss.rev_swig_ptr(vectorstore.index.get_xb(), num_docs*embedding_dimension).reshape(num_docs, embedding_dimension)
        logger.info("Total number of chunks is %d", num_docs)
        _, indices = vectorstore.index.search(doc_embeddings, max(knn_dim))
        k_nn_graph = indices

        if (len(knn_dim)>1 or len(n_clusters)>1) and n_clusters_criterion == 'eigen_gap':
            best_gaps = []
            ncl_candidates = []
            for k_dim in knn_dim:         
                row_indeces = [[i]*k_dim for i in range(len(k_nn_graph))]
                flat_row_indeces = [item for sublist in row_indeces for item in sublist]
            
                col_indeces = [k_nn_graph[i][:k_dim] for i in range(len(k_nn_graph))]
                flat_col_indeces = [item for sublist in col_indeces for item in sublist]
                
                row = np.array(flat_row_indeces)
                col = np.array(flat_col_indeces)
                data = np.ones((len(flat_row_indeces),))

                affinity_matrix = csr_matrix((data, (row, col)), shape=(len(k_nn_graph), len(k_nn_graph)))
                affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.transpose())
                D = diags(1/np.sqrt(affinity_matrix.sum(axis=1).data).flatten())
                Laplacian = D.dot(affinity_matrix).dot(D)
                eigs = eigsh(Laplacian, max(n_clusters)+1, return_eigenvectors = False)
                eigs = np.flip(eigs)
                the_gaps = [eigs[n_clusters[i]-1] - eigs[n_clusters[i]] for i in range(len(n_clusters))]
                best_gaps.append(max(the_gaps))
                ncl_candidates.append(n_clusters[the_gaps.index(max(the_gaps))])
                logger.info("Optimal number of clusters for k=%d based on eigengap heuristic is %d", k_dim, ncl_candidates[-1])
                
            best_k = knn_dim[best_gaps.index(max(best_gaps))]
            best_ncl = ncl_candidates[best_gaps.index(max(best_gaps))]
            logger.info("Optimal clustering parameters are k=%d and number of clusters=%d", best_k, best_ncl)
        else:
            if len(knn_dim)>1 or len(n_clusters)>1:
                logger.warning("Invalid criterion for selecting optimal cluster parameters, using %d clusters and %d for k-nn graph", n_clusters[0], knn_dim[0])
            
            best_k = knn_dim[0]
            best_ncl = n_clusters[0]
                
        row_indeces = [[i]*best_k for i in range(len(k_nn_graph))]
        flat_row_indeces = [item for sublist in row_indeces for item in sublist]
        
        col_indeces = [k_nn_graph[i][:best_k] for i in range(len(k_nn_graph))]
        flat_col_indeces = [item for sublist in col_indeces for item in sublist]
        
        row = np.array(flat_row_indeces)
        col = np.array(flat_col_indeces)
        data = np.ones((len(flat_col_indeces,)))
        affinity_matrix = csr_matrix((data, (row, col)), shape=(len(k_nn_graph), len(k_nn_graph)))
        
        labels_list = SpectralClustering(n_clusters=best_ncl, affinity='precomputed_nearest_neighbors', n_neighbors=best_k, assign_labels='discretize', random_state=self.random_state).fit(affinity_matrix).labels_
        
        #fix order of labels based on input document list
        labels_list_order_fix = labels_list
        for i in range(len(labels_list)):
            _id = vectorstore.index_to_docstore_id[i]
            doc = vectorstore.docstore.search(_id)
            idx_fix = self.documents[0].index(doc)
            self.documents[0][idx_fix].metadata['cluster_id'] = labels_list[i] 
            labels_list_order_fix[idx_fix] = labels_list[i]         

        if visualize_results:
            #TODO: implement visualization of BIC values or eigengap scores, as well as tSNE embeddings of clustering results 
            pass

        return labels_list_order_fix
    

    def _shuffle_and_summarize_documents(self, cluster_labels:list[int], documents_focus:list[Document], documents_exclude:list[Document], llm:AzureChatOpenAI, limit_summarization:int)->list[Document]:
        """Shuffle documents"""
        if llm is not None:
            chain = load_summarize_chain(llm, chain_type="stuff")

        if cluster_labels is not None:
            #shuffle documents based on cluster labels
            clustered_docs = []
            for i in range(max(cluster_labels)+1):
                indexes = [idx for idx, x in enumerate(cluster_labels) if x == i]
                cluster = [self.documents[0][i] for i in indexes]
                random.shuffle(cluster)
                to_append_cluster = True
                if documents_focus is not None:
                    #only consider document clusters that contain documents in the focus list
                    if not any(x in cluster for x in documents_focus):
                        to_append_cluster = False
                        logger.info("Cluster %d excluded since it does not contain focus documents ", i)
                if documents_exclude is not None:
                    #exclude document clusters that contain documents  in the exclude list
                    if any(x in cluster for x in documents_exclude):
                        to_append_cluster = False
                        logger.info("Cluster %d excluded since it contains excluded documents ", i)
                    
                if to_append_cluster:
                    #summarize clusters 
                    cluster_sample = cluster[:min(len(cluster),self.sample_size//(max(cluster_labels)+1))]

                    summaryCluster = chain.invoke(cluster_sample[:min(limit_summarization, len(cluster))])
                    #append summarization to document metadata
                    for chunk in cluster_sample:
                        chunk.metadata['cluster_summary'] = summaryCluster['output_text'] 

                    clustered_docs.append(cluster_sample)      
            self.documents = clustered_docs
        else:
            random.shuffle(self.documents[0])
            self.documents = [self.documents[0][:self.sample_size]]

    def __iter__(self):
        for i in range(self.sample_size//len(self.documents)):
            for j in range(len(self.documents)):
                if i < len(self.documents[j]):
                    yield self.documents[j][i]
            