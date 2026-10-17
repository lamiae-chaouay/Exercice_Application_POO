import pandas as pd

class DataPreprocessor:
    def load_data(self, filepath):
        """Charge le CSV et retourne un DataFrame."""
        return pd.read_csv(filepath)

    def clean(self, df):
        """Supprime les valeurs manquantes."""
        return df.dropna()

    def split(self, df, features, target, test_size=0.2):
        """Sépare en features (X) et target (y)."""
        X = df[features]
        y = df[target]
        split_idx = int(len(df) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]