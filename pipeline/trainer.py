from core.dataset import Dataset


class Trainer:
    def __init__(self, model):
        self.model = model  # N'importe quel modèle avec fit() et predict()

    def train(self, filepath):
        dataset = Dataset(filepath)
        X, y = dataset.get_features_and_target()
        self.model.fit(X, y)
        print("✅ Modèle entraîné avec succès.")

    def get_model(self):
        return self.model