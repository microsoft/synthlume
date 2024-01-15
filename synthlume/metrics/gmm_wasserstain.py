import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance

from langchain_core.embeddings import Embeddings

from synthlume.metrics.metric import Metric
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)

class GMMWasserstein(Metric):
    embeddings: Embeddings

    def __init__(
            self,
            embeddings: Embeddings,
            max_components: int = 10,
        ):
        self.embeddings = embeddings
        self.max_components = max_components
    
    def _calculate_embeddings(self, sentences: list[str]) -> np.array:
        logger.debug(f"Calculating embeddings for {len(sentences)} sentences")
        return np.asarray(self.embeddings.embed_documents(sentences))
    
    
    def optimal_gmm(self, embeddings):
        lowest_bic = np.infty
        best_n_components = None
        best_model = None

        for n_components in range(1, self.max_components+1):
            logger.debug(f"Calculating BIC for {n_components} components")
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            logger.debug(f"BIC: {bic}")
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = n_components
                best_model = gmm
            elif bic > lowest_bic * 2:
                break

        return best_n_components, best_model

    def evaluate(self, sentences_true: list[str], sentences_pred: list[str]) -> float:
        logger.debug(f"Evaluating {len(sentences_pred)} sentences against {len(sentences_true)} sentences")
        
        y_true = self._calculate_embeddings(sentences_true)
        y_pred = self._calculate_embeddings(sentences_pred)

        n_components, true_gmm = self.optimal_gmm(y_true)
        pred_gmm = GaussianMixture(n_components=n_components)
        pred_gmm.fit(y_pred)

        distance = 0
        for i in range(true_gmm.n_components):
            for j in range(pred_gmm.n_components):
                distance += wasserstein_distance(true_gmm.means_[i], pred_gmm.means_[j])
        
        return distance / (true_gmm.n_components * pred_gmm.n_components)

