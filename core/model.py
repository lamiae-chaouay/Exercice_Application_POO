from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Interface commune à tous les modèles IA."""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass