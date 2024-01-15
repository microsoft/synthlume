from abc import ABC, abstractmethod

class Metric(ABC):

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass