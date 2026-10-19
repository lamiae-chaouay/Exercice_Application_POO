# main.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.interface_clinique import InterfaceClinique
from app.interface_tkinter import lancer_tkinter
from pipeline.trainer import Trainer
from core.logistic_regression import LogisticRegression
from app.clinical_predictor import ClinicalPredictor


def menu_principal():
    print("=" * 50)
    print("🦠 PROJET VIRUS_DIAG - Menu Principal")
    print("=" * 50)
    print("1. 🖥️  Lancer l'interface graphique")
    print("2. 💻 Mode console (terminal)")
    print("3. 🧪 Test rapide")
    print("4. 🚪 Quitter")
    print("=" * 50)

    choix = input("\nVotre choix (1/2/3/4) : ").strip()

    if choix == "1":
        print("\n🚀 Lancement de l'interface graphique...")
        lancer_tkinter()

    elif choix == "2":
        print("\n💻 Mode console activé...")
        interface = InterfaceClinique()
        interface.run()

    elif choix == "3":
        print("\n🧪 Test rapide...")
        model = LogisticRegression()
        trainer = Trainer(model)
        trainer.train("data/patient_data.csv")

        predictor = ClinicalPredictor(model)

        patients = [
            {"age": 65, "temperature": 39.2, "cough": 1, "fatigue": 1},
            {"age": 25, "temperature": 36.8, "cough": 0, "fatigue": 0},
            {"age": 45, "temperature": 38.5, "cough": 1, "fatigue": 0},
        ]

        for i, patient in enumerate(patients, 1):
            resultat = predictor.diagnose(patient)
            proba = model.predict(patient)
            print(f"  Patient {i} ({patient['age']} ans, {patient['temperature']}°C) → {resultat} ({proba*100:.1f}%)")

    elif choix == "4":
        print("\n👋 Au revoir !")
        return

    else:
        print("\n❌ Choix invalide !")

    print()
    menu_principal()


if __name__ == "__main__":
    menu_principal()