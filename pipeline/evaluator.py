from utils.metrics import Metrics

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        y_pred = [1 if self.model.predict(x) >= 0.5 else 0 for x in X_test]
        acc = Metrics.accuracy(y_test, y_pred)
        print(f"📊 Précision : {acc:.2%}")
        return acc