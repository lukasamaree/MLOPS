from metaflow import FlowSpec, step
import numpy as np
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5001")
mlflow.set_experiment("lab_6_ml_orchestration")

class TrainFlow(FlowSpec):

    @step
    def start(self):
        from clean_data import x_enc, y
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_enc, y, test_size=0.2, random_state=42
        )

        self.hyperparams = [
            {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 10},
            {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 5},
            {"model_type": "DecisionTree", "max_depth": 3, "min_samples_split": 2},
            {"model_type": "DecisionTree", "max_depth": 6, "min_samples_split": 4},
        ]
        self.next(self.train_model, foreach="hyperparams")

    @step
    def train_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error
        import mlflow.sklearn

        params = self.input
        model_type = params["model_type"]

        if model_type == "RandomForest":
            self.model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
        elif model_type == "DecisionTree":
            self.model = DecisionTreeRegressor(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.fit(self.x_train, self.y_train)
        preds = self.model.predict(self.x_val)
        self.rmse = np.sqrt(mean_squared_error(self.y_val, preds))
        self.model_name = model_type
        self.params = params

        with mlflow.start_run(run_name=f"{model_type}_run") as run:
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.sklearn.log_model(self.model, artifact_path="model")
            self.run_id = run.info.run_id

        print(f"Trained {self.model_name} | RMSE: {self.rmse:.2f}")
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        best = min(inputs, key=lambda x: x.rmse)

        self.model = best.model
        self.model_name = best.model_name
        self.rmse = best.rmse
        self.best_run_id = best.run_id
        self.params = best.params

        print(f"🏆 Best model: {self.model_name} with RMSE: {self.rmse:.2f}, Params: {self.params}")
        experiment_name = mlflow.get_experiment_by_name("lab_6_ml_orchestration").name

    # Register model from best run
        model_uri = f"runs:/{self.best_run_id}/model"
        registered_model_name = f"BestModel_registered"
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

        mlflow.set_tag("best_model", self.model_name)
        mlflow.set_tag("best_run_id", self.best_run_id)
        for k, v in self.params.items():
            mlflow.set_tag(f"best_param_{k}", v)
        self.registered_model_name = registered_model_name
        


        print(f"📦 Registering model {self.registered_model_name} from run {self.best_run_id}")

        self.next(self.end)

    @step
    def end(self):
        print(f"Final selected model: {self.registered_model_name}, RMSE: {self.rmse:.2f}")


if __name__ == "__main__":
    TrainFlow()

