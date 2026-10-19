# app/interface_clinique.py
from app.clinical_predictor import ClinicalPredictor
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer


class InterfaceClinique:
    def __init__(self):
        self.model = LogisticRegression()
        Trainer(self.model).train("data/patient_data.csv")
        self.predictor = ClinicalPredictor(self.model)

    def run(self):
        print("\n" + "=" * 40)
        print("🦠 MODE CONSOLE - Virus Diag")
        print("=" * 40)
        try:
            age = int(input("Âge du patient : "))
            temperature = float(input("Température (°C) : "))
            toux = int(input("Toux ? (0 = Non, 1 = Oui) : "))
            fatigue = int(input("Fatigue ? (0 = Non, 1 = Oui) : "))

            patient = {
                "age": age,
                "temperature": temperature,
                "cough": toux,
                "fatigue": fatigue,
            }

            proba = self.model.predict(patient)
            resultat = self.predictor.diagnose(patient)

            print("\n" + "-" * 40)
            print(f"📊 Diagnostic : {resultat}")
            print(f"📈 Probabilité d'infection : {proba * 100:.1f}%")
            print("-" * 40)

        except ValueError:
            print("❌ Erreur : entrée invalide. Veuillez entrer des nombres.")