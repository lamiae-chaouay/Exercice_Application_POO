from utils.preprocessing import DataPreprocessor

class Dataset:
    def __init__(self, filepath):
        preprocessor = DataPreprocessor()
        self.df = preprocessor.load_data(filepath)
        self.df = preprocessor.clean(self.df)

    def get_features_and_target(self):
        features = ['age', 'temperature', 'cough', 'fatigue']
        target = 'infected'
        return self.df[features], self.df[target]