
# core/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from core.model import BaseModel


class RandomForestModel(BaseModel):
    """Modèle Random Forest pour la prédiction du risque d'infection."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        random_state=42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_trained = False

    def fit(self, X, y):
        """Entraîne le modèle."""
        self.model.fit(X, y)
        self.is_trained = True

        print("Random Forest entraîné avec succès.")

    def predict(self, patient_data):
        """Retourne la probabilité de la classe positive."""

        if not self.is_trained:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")

        if isinstance(patient_data, dict):
            features = [[
                patient_data.get("age", 0),
                patient_data.get("temperature", 0),
                patient_data.get("cough", 0),
                patient_data.get("fatigue", 0)
            ]]
        else:
            features = patient_data

            # Accepter un seul patient sous forme de liste
            if len(features) > 0 and not isinstance(features[0], (list, tuple)):
                features = [features]

        probabilities = self.model.predict_proba(features)

        # Recherche de la classe positive (1)
        positive_index = list(self.model.classes_).index(1)

        return float(probabilities[0][positive_index])
