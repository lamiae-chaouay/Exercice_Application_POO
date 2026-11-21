# core/model.py
from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Interface commune à tous les modèles IA.

    Tout modèle doit implémenter :
        - fit(X, y)  : entraînement sur des données
        - predict(X) : prédiction sur un échantillon
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Entraîne le modèle sur les données (X, y)."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> float:
        """Retourne une probabilité ou une prédiction pour un échantillon."""
        pass