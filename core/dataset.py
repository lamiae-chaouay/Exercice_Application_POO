# core/dataset.py
import os

from utils.preprocessing import DataPreprocessor


class Dataset:
    """Charge et prépare le dataset pour l'entraînement."""

    FEATURES = ['age', 'temperature', 'cough', 'fatigue']
    TARGET = 'infected'

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Dataset introuvable : {filepath}")

        preprocessor = DataPreprocessor()
        self.df = preprocessor.load_data(filepath)
        self.df = preprocessor.clean(self.df)

        if len(self.df) == 0:
            raise ValueError("❌ Le dataset est vide après nettoyage.")

    def get_features_and_target(self):
        """Retourne (X, y) : les features et la cible."""
        colonnes_manquantes = [
            c for c in self.FEATURES + [self.TARGET] if c not in self.df.columns
        ]
        if colonnes_manquantes:
            raise ValueError(
                f"❌ Colonnes manquantes dans le dataset : {colonnes_manquantes}"
            )
        return self.df[self.FEATURES], self.df[self.TARGET]