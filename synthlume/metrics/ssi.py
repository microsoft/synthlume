import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from langchain_core.embeddings import Embeddings

from synthlume.metrics.metric import Metric
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)

class SentenceSeparabilityIndex(Metric):
    embeddings: Embeddings

    def __init__(
            self,
            embeddings: Embeddings,
            regression_kwargs: dict = None,
            n_splits: int = 5,
            random_state: int = 42,
        ):
        self.embeddings = embeddings
        self.n_splits = n_splits
        self.random_state = random_state
        self.regression_kwargs = regression_kwargs or {}

    def _calculate_embeddings(self, sentences: list[str]) -> list[list[float]]:
        logger.debug(f"Calculating embeddings for {len(sentences)} sentences")
        return self.embeddings.embed_documents(sentences)
    
    def _create_dataset(self, sentences_true: list[str], sentences_pred: list[str]) -> tuple[list[list[float]], list[float]]:
        logger.debug(f"Creating dataset for {len(sentences_true)} true sentences and {len(sentences_pred)} predicted sentences")
        x_pos = self._calculate_embeddings(sentences_true)
        x_neg = self._calculate_embeddings(sentences_pred)

        y_pos = [1.0] * len(x_pos)
        y_neg = [0.0] * len(x_neg)

        x = x_pos + x_neg
        y = y_pos + y_neg
        
        return x, y

    def _train_linear_regression(self, x: list[list[float]], y: list[float]) -> None:
        logger.debug(f"Training linear regression model on {len(x)} samples")

        X = np.asarray(x)
        y = np.asarray(y)

        logger.debug(f"Using {self.n_splits} folds for cross-validation")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Store the results
        results = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            logger.debug(f"Training fold {i + 1} of {self.n_splits}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.regression_kwargs["random_state"] = self.regression_kwargs.get("random_state", self.random_state)
            self.regression_kwargs["solver"] = self.regression_kwargs.get("solver", "liblinear")

            logger.debug(f"Training logistic regression model with {self.regression_kwargs}")
            model = LogisticRegression(**self.regression_kwargs)

            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_pred)
            logger.debug(f"Fold {i + 1} of {self.n_splits} has ROC AUC score of {roc_auc}")

            results.append(roc_auc)

        logger.debug(f"Mean ROC AUC score is {np.mean(results)}")

        metric_value = np.mean(results)
        if metric_value < 0.5:
            metric_value = 1 - metric_value
        
        return (metric_value - 0.5) * 2

    def evaluate(self, sentences_true: list[str], sentences_pred: list[str]) -> float:
        x, y = self._create_dataset(sentences_true, sentences_pred)
        return self._train_linear_regression(x, y)
