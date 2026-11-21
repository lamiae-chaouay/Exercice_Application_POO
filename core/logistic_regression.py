# core/logistic_regression.py
import math

from core.model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01, epochs=100):
        self.weights = None
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs

    def _sigmoid(self, z):
        """Fonction sigmoïde : retourne une probabilité entre 0 et 1."""
        # Protection contre l'overflow
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1 + ez)

    def fit(self, X, y):
        """Entraînement du modèle sur les données."""
        # Conversion DataFrame → liste de listes
        if hasattr(X, 'values'):
            X = X.values.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()
        elif hasattr(y, 'values'):
            y = y.values.tolist()

        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("❌ Aucune donnée d'entraînement fournie.")

        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Descente de gradient (SGD)
        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_pred = sum(
                    self.weights[j] * X[i][j] for j in range(n_features)
                ) + self.bias
                y_pred = self._sigmoid(linear_pred)
                error = y_pred - y[i]

                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

        print(f"✅ Modèle entraîné sur {n_samples} échantillons avec {n_features} features.")

    def predict(self, patient_data):
        """Retourne une probabilité d'infection (entre 0 et 1)."""
        if self.weights is None:
            raise RuntimeError("❌ Modèle non entraîné. Appelle fit() d'abord.")

        if isinstance(patient_data, dict):
            features = [
                patient_data.get('age', 0),
                patient_data.get('temperature', 0),
                patient_data.get('cough', 0),
                patient_data.get('fatigue', 0),
            ]
        else:
            features = patient_data

        linear_pred = sum(
            self.weights[i] * features[i] for i in range(len(features))
        ) + self.bias
        return self._sigmoid(linear_pred)