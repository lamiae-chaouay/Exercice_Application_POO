class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        tp = fp = tn = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1: tp += 1
            elif t == 0 and p == 1: fp += 1
            elif t == 0 and p == 0: tn += 1
            else: fn += 1
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}