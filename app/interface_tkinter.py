# app/interface_tkinter.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# 🔑 Correction des chemins
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer
from app.clinical_predictor import ClinicalPredictor


class InterfaceTkinter:
    def __init__(self, root):
        self.root = root
        self.root.title("🦠 Virus Diag - Interface Clinique")
        self.root.geometry("700x600")
        self.root.resizable(False, False)
        
        # Couleurs
        self.COLOR_PRIMARY = "#2563eb"
        self.COLOR_DANGER = "#dc2626"
        self.COLOR_SUCCESS = "#16a34a"
        
        self.root.configure(bg="#f0f4f8")
        
        # Initialisation du modèle (fait une seule fois au démarrage)
        self.model = LogisticRegression()
        trainer = Trainer(self.model)
        trainer.train("data/patient_data.csv")
        self.predictor = ClinicalPredictor(self.model)
        
        self._creer_interface()
    
    def _creer_interface(self):
        # TITRE
        titre_frame = tk.Frame(self.root, bg=self.COLOR_PRIMARY, height=80)
        titre_frame.pack(fill=tk.X)
        titre_frame.pack_propagate(False)
        
        tk.Label(titre_frame, text="🦠 VIRUS DIAG", font=("Arial", 24, "bold"), bg=self.COLOR_PRIMARY, fg="white").pack(pady=15)
        
        # CONTENEUR PRINCIPAL
        main_frame = tk.Frame(self.root, bg="#f0f4f8")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # FORMULAIRE
        form_frame = tk.LabelFrame(main_frame, text="📋 Informations du Patient", font=("Arial", 12, "bold"), bg="white", padx=20, pady=20)
        form_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(form_frame, text="Âge :", font=("Arial", 11), bg="white").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.age_var = tk.StringVar(value="45")
        tk.Entry(form_frame, textvariable=self.age_var, font=("Arial", 11), width=30).grid(row=0, column=1, pady=8, padx=10)
        
        tk.Label(form_frame, text="Température (°C) :", font=("Arial", 11), bg="white").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.temp_var = tk.StringVar(value="37.0")
        tk.Entry(form_frame, textvariable=self.temp_var, font=("Arial", 11), width=30).grid(row=1, column=1, pady=8, padx=10)
        
        tk.Label(form_frame, text="Toux :", font=("Arial", 11), bg="white").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.toux_var = tk.IntVar(value=0)
        tk.Checkbutton(form_frame, variable=self.toux_var, bg="white").grid(row=2, column=1, sticky=tk.W, pady=8, padx=10)
        
        tk.Label(form_frame, text="Fatigue :", font=("Arial", 11), bg="white").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.fatigue_var = tk.IntVar(value=0)
        tk.Checkbutton(form_frame, variable=self.fatigue_var, bg="white").grid(row=3, column=1, sticky=tk.W, pady=8, padx=10)
        
        # BOUTONS
        btn_frame = tk.Frame(main_frame, bg="#f0f4f8")
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="🩺 DIAGNOSTIQUER", font=("Arial", 12, "bold"), bg=self.COLOR_PRIMARY, fg="white", height=2, command=self._diagnostiquer, cursor="hand2").pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="🔄 Réinitialiser", font=("Arial", 10), bg="#64748b", fg="white", command=self._reinitialiser, cursor="hand2").pack(fill=tk.X, pady=5)
        
        # RÉSULTAT
        self.resultat_frame = tk.LabelFrame(main_frame, text="📊 Résultat du Diagnostic", font=("Arial", 12, "bold"), bg="white", padx=20, pady=20)
        self.resultat_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.resultat_label = tk.Label(self.resultat_frame, text="En attente de diagnostic...", font=("Arial", 16, "bold"), bg="white", fg="#94a3b8")
        self.resultat_label.pack(pady=10)
        
        self.probabilite_label = tk.Label(self.resultat_frame, text="", font=("Arial", 11), bg="white", fg="#1e293b")
        self.probabilite_label.pack()
        
        # HISTORIQUE
        hist_frame = tk.LabelFrame(main_frame, text="📜 Historique", font=("Arial", 11, "bold"), bg="white", padx=10, pady=10)
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        columns = ("Patient", "Temp", "Toux", "Fatigue", "Diagnostic")
        self.tree = ttk.Treeview(hist_frame, columns=columns, show="headings", height=5)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _diagnostiquer(self):
        try:
            age = int(self.age_var.get())
            temperature = float(self.temp_var.get())
            toux = self.toux_var.get()
            fatigue = self.fatigue_var.get()
            
            patient_data = {"age": age, "temperature": temperature, "cough": toux, "fatigue": fatigue}
            proba = self.predictor.model.predict(patient_data)
            resultat = self.predictor.diagnose(patient_data)
            
            if resultat == "Infecté":
                self.resultat_label.config(text=f"⚠️ {resultat.upper()}", fg=self.COLOR_DANGER)
            else:
                self.resultat_label.config(text=f"✅ {resultat.upper()}", fg=self.COLOR_SUCCESS)
            
            self.probabilite_label.config(text=f"Probabilité d'infection : {proba*100:.1f}%")
            
            self.tree.insert("", 0, values=(
                f"{age} ans",
                f"{temperature}°C",
                "Oui" if toux else "Non",
                "Oui" if fatigue else "Non",
                resultat
            ))
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides !")
    
    def _reinitialiser(self):
        self.age_var.set("45")
        self.temp_var.set("37.0")
        self.toux_var.set(0)
        self.fatigue_var.set(0)
        self.resultat_label.config(text="En attente de diagnostic...", fg="#94a3b8")
        self.probabilite_label.config(text="")


def lancer_tkinter():
    root = tk.Tk()
    app = InterfaceTkinter(root)
    root.mainloop()


if __name__ == "__main__":
    lancer_tkinter()