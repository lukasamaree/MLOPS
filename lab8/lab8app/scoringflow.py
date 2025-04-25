from metaflow import FlowSpec, step
import pandas as pd
import mlflow.sklearn

import mlflow

mlflow.set_tracking_uri("http://localhost:5001")  # or your cloud MLflow URL


class ScoringFlow(FlowSpec):

    @step
    def start(self):
        # Import your hold-out test set from your data-cleaning module
        from clean_data import x_enc_test, y_test
        self.x_test = x_enc_test
        self.y_test = y_test
        print("Test data loaded successfully.")
        self.next(self.load_model)

    @step
    def load_model(self):
        from mlflow import MlflowClient
        model_name = "BestModel_registered"
        client = MlflowClient()

        # Grab latest version based on order
        latest = client.search_model_versions(f"name='{model_name}'")[-1]
        uri = f"models:/{model_name}/{latest.version}"

        self.model = mlflow.sklearn.load_model(uri)
        print(f"Loaded latest version: {latest.version}")
    
        self.next(self.predict)

    @step
    def predict(self):
        import numpy as np
        # Use the model to predict and evaluate
        preds = self.model.predict(self.x_test)

        from sklearn.metrics import accuracy_score, mean_squared_error
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)

        print(f"MSE: {mse:.4f}")
        print(f'rmse: {rmse}')
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow completed.")

if __name__ == "__main__":
    ScoringFlow()
