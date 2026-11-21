
# app/api.py

import os

from app.clinical_predictor import ClinicalPredictor
from core.logistic_regression import LogisticRegression
from core.random_forest import RandomForestModel
from pipeline.trainer import Trainer


# Chemin absolu vers le dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "patient_data.csv")


class VirusDiagAPI:
    """
    API haut niveau pour le diagnostic du virus.

    Deux modèles disponibles :
        - logistic       : Logistic Regression
        - random_forest  : Random Forest

    Exemple :
        api = VirusDiagAPI(model_type="random_forest")
        resultat = api.diagnose_detail({
            "age": 65,
            "temperature": 39.2,
            "cough": 1,
            "fatigue": 1
        })
    """

    # Cache séparé pour chaque modèle
    _model_instances = {}

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 500,
        model_type: str = "logistic"
    ):
        """
        Initialise l'API et entraîne le modèle choisi.

        Args:
            learning_rate: Taux d'apprentissage de Logistic Regression.
            epochs: Nombre d'époques de Logistic Regression.
            model_type: 'logistic' ou 'random_forest'.
        """

        if model_type not in ("logistic", "random_forest"):
            raise ValueError(
                "Type de modèle invalide. "
                "Choisir 'logistic' ou 'random_forest'."
            )

        self.model_type = model_type

        # Vérifier si le modèle est déjà entraîné
        if model_type not in VirusDiagAPI._model_instances:

            print(f"Premier lancement : entraînement de {model_type}...")

            # Sélection du modèle
            if model_type == "logistic":
                model = LogisticRegression(
                    learning_rate=learning_rate,
                    epochs=epochs
                )

            else:
                model = RandomForestModel(
                    n_estimators=100,
                    max_depth=None,
                    random_state=42
                )

            # Vérifier l'existence du dataset
            if not os.path.exists(CSV_PATH):
                raise FileNotFoundError(
                    f"Dataset introuvable : {CSV_PATH}\n"
                    "Vérifie que data/patient_data.csv existe."
                )

            # Entraînement
            trainer = Trainer(model)
            trainer.train(CSV_PATH)

            # Enregistrer dans le cache
            VirusDiagAPI._model_instances[model_type] = model

            print(f"Modèle {model_type} entraîné et mis en cache.")

        else:
            print(f"Modèle {model_type} récupéré depuis le cache.")

        # Récupération du modèle
        self.model = VirusDiagAPI._model_instances[model_type]

        # Initialisation du prédicteur clinique
        self.predictor = ClinicalPredictor(self.model)

    def diagnose(self, patient_data: dict) -> str:
        """
        Retourne le diagnostic du patient.

        Returns:
            'Infecté' ou 'Sain'.
        """

        self._valider_patient(patient_data)

        return self.predictor.diagnose(patient_data)

    def predict_proba(self, patient_data: dict) -> float:
        """
        Retourne la probabilité d'infection entre 0 et 1.
        """

        self._valider_patient(patient_data)

        return self.model.predict(patient_data)

    def diagnose_detail(self, patient_data: dict) -> dict:
        """
        Retourne un diagnostic détaillé.

        Returns:
            {
                'diagnostic': str,
                'probabilite': float,
                'pourcentage': str,
                'patient': dict,
                'modele': str
            }
        """

        self._valider_patient(patient_data)

        proba = self.model.predict(patient_data)
        diagnostic = (
            "Infecté" if proba >= self.predictor.seuil else "Sain"
        )

        return {
            "diagnostic": diagnostic,
            "probabilite": proba,
            "pourcentage": f"{proba * 100:.1f}%",
            "patient": patient_data.copy(),
            "modele": self.model_type
        }

    @staticmethod
    def _valider_patient(patient_data: dict) -> None:
        """
        Vérifie que les données du patient sont complètes et valides.
        """

        if not isinstance(patient_data, dict):
            raise TypeError("Les données du patient doivent être un dictionnaire.")

        champs_requis = ["age", "temperature", "cough", "fatigue"]

        for champ in champs_requis:
            if champ not in patient_data:
                raise ValueError(f"Champ manquant : '{champ}'")

        if not (0 < patient_data["age"] < 130):
            raise ValueError(f"Âge invalide : {patient_data['age']}")

        if not (30.0 <= patient_data["temperature"] <= 45.0):
            raise ValueError(
                f"Température invalide : {patient_data['temperature']}"
            )

        if patient_data["cough"] not in (0, 1):
            raise ValueError("'cough' doit être 0 ou 1.")

        if patient_data["fatigue"] not in (0, 1):
            raise ValueError("'fatigue' doit être 0 ou 1.")

    @classmethod
    def reset_cache(cls) -> None:
        """
        Réinitialise le cache des deux modèles.
        """

        cls._model_instances = {}

        print("Cache des modèles réinitialisé.")
