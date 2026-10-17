class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, model, loss):
        # Mise à jour des poids (simplifié)
        pass