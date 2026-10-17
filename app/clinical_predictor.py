class ClinicalPredictor:
    """
    Reçoit un modèle déjà entraîné.
    Retourne 'Infecté' si prédiction >= 0.5, 'Sain' sinon.
    """

    def __init__(self, model):
        self.model = model  # Polymorphisme : accepte tout modèle avec predict()

    def diagnose(self, patient_data):
        prediction = self.model.predict(patient_data)
        return "Infecté" if prediction >= 0.5 else "Sain"