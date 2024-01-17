import numpy as np

from langchain_core.embeddings import Embeddings

from synthlume.metrics.metric import Metric
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)

class CosineSimilarity(Metric):
    embeddings: Embeddings

    def __init__(
            self,
            embeddings: Embeddings,
        ):
        self.embeddings = embeddings
    
    def _calculate_embeddings(self, sentences: list[str]) -> np.array:
        logger.debug(f"Calculating embeddings for {len(sentences)} sentences")
        return np.asarray(self.embeddings.embed_documents(sentences))

    def _compare_against_y(self, x: str, y: np.array) -> float:
        embedding = self.embeddings.embed_query(x)
        embedding = np.asarray(embedding).reshape(1, -1)

        scores = np.dot(embedding, y.T)
        scores = scores.flatten()

        argmax = np.argmax(scores)

        return max(scores), argmax

    def evaluate_scores(self, sentences_true: list[str], sentences_pred: list[str]):
        logger.debug(f"Evaluating {len(sentences_pred)} sentences against {len(sentences_true)} sentences")
        scores = []
        y_true = self._calculate_embeddings(sentences_true)

        for sentence in sentences_pred:
            score, _ = self._compare_against_y(sentence, y_true)
            scores.append(score)

        return scores

    def evaluate(self, sentences_true: list[str], sentences_pred: list[str]) -> float:
        scores = self.evaluate_scores(sentences_true, sentences_pred)
        return np.mean(scores)
