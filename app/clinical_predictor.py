# app/clinical_predictor.py


class ClinicalPredictor:
    """
    Reçoit un modèle déjà entraîné (polymorphisme : tout modèle avec predict()).

    Retourne 'Infecté' si la probabilité >= seuil (0.5 par défaut), 'Sain' sinon.
    """

    def __init__(self, model, seuil: float = 0.5):
        """
        Args:
            model: modèle entraîné avec une méthode predict(patient_data) -> float
            seuil: seuil de décision entre 0.0 et 1.0 (défaut : 0.5)
        """
        if not 0.0 <= seuil <= 1.0:
            raise ValueError(f"❌ Le seuil doit être entre 0.0 et 1.0 (reçu : {seuil})")
        self.model = model
        self.seuil = seuil

    def diagnose(self, patient_data: dict) -> str:
        """Retourne 'Infecté' ou 'Sain' selon la probabilité prédite."""
        prediction = self.model.predict(patient_data)
        return "Infecté" if prediction >= self.seuil else "Sain"

    def diagnose_detail(self, patient_data: dict) -> dict:
        """Retourne un diagnostic détaillé avec la probabilité."""
        proba = self.model.predict(patient_data)
        return {
            "diagnostic": "Infecté" if proba >= self.seuil else "Sain",
            "probabilite": proba,
            "pourcentage": f"{proba * 100:.1f}%",
        }