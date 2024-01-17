from abc import ABC, abstractmethod

class Metric(ABC):

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

    def evaluate_scores(self, y_true, y_pred):
        raise NotImplementedError