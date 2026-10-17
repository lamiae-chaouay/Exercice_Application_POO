from app.clinical_predictor import ClinicalPredictor
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer

class VirusDiagAPI:
    def __init__(self):
        # Entraînement une seule fois
        model = LogisticRegression()
        trainer = Trainer(model)
        trainer.train("data/patient_data.csv")

        self.predictor = ClinicalPredictor(trainer.get_model())

    def diagnose(self, patient_data):
        return self.predictor.diagnose(patient_data)