# app/api.py
import os

from app.clinical_predictor import ClinicalPredictor
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer


# 🔑 Chemin absolu robuste vers le dataset
# (fonctionne quel que soit le dossier depuis lequel on lance le script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "patient_data.csv")


class VirusDiagAPI:
    """
    API haut niveau pour le diagnostic du virus.

    Exemple d'utilisation :
        >>> api = VirusDiagAPI()
        >>> api.diagnose({"age": 65, "temperature": 39.2, "cough": 1, "fatigue": 1})
        'Infecté'
    """

    # 🧠 Cache du modèle entraîné (Singleton — évite de réentraîner à chaque instanciation)
    _model_instance = None

    def __init__(self, learning_rate: float = 0.01, epochs: int = 500):
        """
        Initialise l'API.

        Le modèle n'est entraîné qu'une seule fois, puis réutilisé
        pour toutes les instances suivantes (grâce au cache de classe).

        Args:
            learning_rate: Taux d'apprentissage (défaut : 0.01)
            epochs: Nombre d'itérations d'entraînement (défaut : 500)
        """
        if VirusDiagAPI._model_instance is None:
            print("🔄 Premier lancement : entraînement du modèle...")
            model = LogisticRegression(learning_rate=learning_rate, epochs=epochs)
            trainer = Trainer(model)

            if not os.path.exists(CSV_PATH):
                raise FileNotFoundError(
                    f"❌ Dataset introuvable : {CSV_PATH}\n"
                    f"   Vérifie que le fichier 'data/patient_data.csv' existe."
                )

            trainer.train(CSV_PATH)
            VirusDiagAPI._model_instance = model
            print("✅ Modèle entraîné et mis en cache.")
        else:
            print("♻️ Modèle récupéré depuis le cache (pas de réentraînement).")

        self.model = VirusDiagAPI._model_instance
        self.predictor = ClinicalPredictor(self.model)

    def diagnose(self, patient_data: dict) -> str:
        """
        Diagnostique un patient.

        Args:
            patient_data: Dictionnaire avec les clés :
                - age (int)
                - temperature (float)
                - cough (int : 0 ou 1)
                - fatigue (int : 0 ou 1)

        Returns:
            "Infecté" ou "Sain"

        Raises:
            ValueError: Si une clé est manquante ou invalide.
        """
        self._valider_patient(patient_data)
        return self.predictor.diagnose(patient_data)

    def predict_proba(self, patient_data: dict) -> float:
        """
        Retourne la probabilité d'infection (entre 0.0 et 1.0).

        Args:
            patient_data: Même format que diagnose().

        Returns:
            Probabilité d'infection en float.
        """
        self._valider_patient(patient_data)
        return self.model.predict(patient_data)

    def diagnose_detail(self, patient_data: dict) -> dict:
        """
        Retourne un diagnostic détaillé (pratique pour une interface).

        Returns:
            {
                "diagnostic": "Infecté" | "Sain",
                "probabilite": float,
                "pourcentage": str,
                "patient": dict,
            }
        """
        self._valider_patient(patient_data)
        proba = self.model.predict(patient_data)
        diagnostic = self.predictor.diagnose(patient_data)

        return {
            "diagnostic": diagnostic,
            "probabilite": proba,
            "pourcentage": f"{proba * 100:.1f}%",
            "patient": patient_data.copy(),
        }

    @staticmethod
    def _valider_patient(patient_data: dict) -> None:
        """Vérifie que les données du patient sont complètes et valides."""
        champs_requis = ["age", "temperature", "cough", "fatigue"]
        for champ in champs_requis:
            if champ not in patient_data:
                raise ValueError(f"❌ Champ manquant : '{champ}'")

        if not (0 < patient_data["age"] < 130):
            raise ValueError(f"❌ Âge invalide : {patient_data['age']}")
        if not (30.0 <= patient_data["temperature"] <= 45.0):
            raise ValueError(f"❌ Température invalide : {patient_data['temperature']}")
        if patient_data["cough"] not in (0, 1):
            raise ValueError(f"❌ 'cough' doit être 0 ou 1 (reçu : {patient_data['cough']})")
        if patient_data["fatigue"] not in (0, 1):
            raise ValueError(f"❌ 'fatigue' doit être 0 ou 1 (reçu : {patient_data['fatigue']})")

    @classmethod
    def reset_cache(cls) -> None:
        """Réinitialise le cache du modèle (utile pour les tests)."""
        cls._model_instance = None
        print("♻️ Cache du modèle réinitialisé.")