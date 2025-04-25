from metaflow import FlowSpec, step, conda

# define once to avoid repetition
common_conda = {'python': '3.13.3', 'packages': {'joblib': '1.4.2', 'numpy': '2.2.5', 'scikit-learn': '1.6.1', 'pandas': '2.2.3'}}

class RedditClassifier(FlowSpec):
    @conda(**common_conda)
    @step
    def start(self):
        print("Flow is starting")
        self.next(self.load_data)

    @conda(**common_conda)
    @step
    def load_data(self):
        import pandas as pd
        import numpy

        print("Data is loading")
        self.x_sample = pd.read_csv('sample_reddit.csv', header=None).to_numpy().reshape((-1,))
        print("Data is loaded")
        self.next(self.load_model)

    @conda(**common_conda)
    @step
    def load_model(self):
        import joblib
        print("Pipeline loading")
        self.loaded_pipeline = joblib.load("reddit_model_pipeline.joblib")
        print("Pipeline loaded")
        self.next(self.predict_class)

    @conda(**common_conda)
    @step
    def predict_class(self):
        print("Making predictions")
        self.predictions = self.loaded_pipeline.predict_proba(self.x_sample)
        print("Predictions made")
        self.next(self.save_results)

    @conda(**common_conda)
    @step
    def save_results(self):
        import pandas as pd
        print("Saving results")
        pd.DataFrame(self.predictions).to_csv("sample_preds.csv", index=None, header=None)
        print("Results saved")
        self.next(self.end)

    @step
    def end(self):
        print("Flow is ending")

if __name__ == '__main__':
    RedditClassifier()
