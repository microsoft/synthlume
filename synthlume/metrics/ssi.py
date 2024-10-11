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
        self.regression_kwargs = regression_kwargs or {"C": 0.1}

    def _calculate_embeddings(self, sentences: list[str]) -> list[list[float]]:
        logger.debug(f"Calculating embeddings for {len(sentences)} sentences")
        return self.embeddings.embed_documents(sentences)

    def _create_dataset(
        self, sentences_true: list[str], sentences_pred: list[str]
    ) -> tuple[list[list[float]], list[float]]:
        logger.debug(
            f"Creating dataset for {len(sentences_true)} true sentences and {len(sentences_pred)} predicted sentences"
        )
        x_true = self._calculate_embeddings(sentences_true)
        x_gen = self._calculate_embeddings(sentences_pred)

        return x_true, x_gen

    def _train_linear_regression(
        self, x_true: list[list[float]], x_gen: list[list[float]]
    ) -> None:
        logger.debug(
            f"Training linear regression model on {len(x_gen) + len(x_true)} samples"
        )

        X = np.asarray(x_true + x_gen)
        y = np.asarray([1] * len(x_true) + [0] * len(x_gen))

        logger.debug(f"Using {self.n_splits} folds for cross-validation")
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # Store the results
        results = []

        x_gen_scores = None

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            logger.debug(f"Training fold {i + 1} of {self.n_splits}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.regression_kwargs["random_state"] = self.regression_kwargs.get(
                "random_state", self.random_state
            )
            self.regression_kwargs["solver"] = self.regression_kwargs.get(
                "solver", "liblinear"
            )

            logger.debug(
                f"Training logistic regression model with {self.regression_kwargs}"
            )
            model = LogisticRegression(**self.regression_kwargs)

            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_pred)
            logger.debug(
                f"Fold {i + 1} of {self.n_splits} has ROC AUC score of {roc_auc}"
            )

            x_true_preds = model.predict_proba(x_true)[:, 1]
            x_gen_preds = model.predict_proba(x_gen)[:, 1]

            comparison_matrix = x_gen_preds[:, np.newaxis] >= x_true_preds
            count_lower = 1.0 * comparison_matrix.sum(axis=1) / len(x_true_preds)

            if x_gen_scores is None:
                x_gen_scores = count_lower
            else:
                x_gen_scores = x_gen_scores + count_lower

            results.append(roc_auc)

        logger.debug(f"Mean ROC AUC score is {np.mean(results)}")

        x_gen_scores = x_gen_scores / self.n_splits
        x_gen_scores = 1 - 2 * abs(x_gen_scores - 0.5)

        metric_value = np.mean(results)
        if metric_value < 0.5:
            metric_value = 1 - metric_value

        return 1 - (metric_value - 0.5) * 2, x_gen_scores

    def evaluate_scores(self, sentences_true: list[str], sentences_pred: list[str]):
        score, scores = self._train_linear_regression(
            *self._create_dataset(sentences_true, sentences_pred)
        )
        return scores, score

    def evaluate(self, sentences_true: list[str], sentences_pred: list[str]) -> float:
        score, _ = self._train_linear_regression(
            *self._create_dataset(sentences_true, sentences_pred)
        )
        return score
