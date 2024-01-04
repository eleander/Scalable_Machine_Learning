import modal

LOCAL=False
N_SAMPLES=4

if LOCAL == False:
   stub = modal.Stub("heart_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "ydata-synthetic==1.3.2", "pandas", "scikit-learn==1.3.2", "joblib", "numpy"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_random_heart(project):
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer
    import joblib
    import pandas as pd
    import numpy as np

    class InversePipeline:
        def __init__(self, pipeline, df_columns):
            _, self.numerical_pipeline, self.numerical = pipeline.transformers_[0]
            _, self.categorical_pipeline, self.categorical = pipeline.transformers_[1]

            # Assumes that the last step of the pipeline is the encoder
            self.numerical_pipeline = self.numerical_pipeline[-1]
            self.categorical_pipeline = self.categorical_pipeline[-1]

            self.pipeline = pipeline
            self.columns = df_columns

        def transform(self, X, y=None):
            num = self.numerical_pipeline.inverse_transform(X[:, :len(self.numerical)])
            cat = self.categorical_pipeline.inverse_transform(X[:, len(self.numerical):])

            df = pd.DataFrame(np.concatenate([num, cat], axis=1), columns=self.numerical.tolist() + self.categorical.tolist())
            # Change dtypes
            df[self.numerical] = df[self.numerical].astype(np.float64)
            df[self.categorical] = df[self.categorical].astype("category")
            
            if y is not None:
                df['heart_disease'] = y

            # Tries to re-order columns
            columns = [col for col in self.columns if col in df.columns]
            df = df.reindex(columns=columns)
            return df
        
        def save(self, path):
            joblib.dump({"pipeline": self.pipeline, "columns": self.columns}, path)

        @staticmethod
        def load(path):
            data = joblib.load(path)
            return InversePipeline(data["pipeline"], data["columns"])

    mr = project.get_model_registry()
    model = mr.get_model("heart_generator", version=1)
    model_dir = model.download()
    print("Successfully downloaded model")

    model = RegularSynthesizer.load(model_dir + '/heart_generator.pkl')
    print("Successfully loaded model")

    inverse_pipeline = InversePipeline.load(model_dir + '/inverse_pipeline.pkl')
    print("Successfully loaded inverse pipeline")

    samples = model.sample(N_SAMPLES)
    y, X = samples['heart_disease'], samples.drop(columns=['heart_disease'])
    
    return inverse_pipeline.transform(X.to_numpy(), y.to_numpy())

def g():
    import hopsworks
    from datetime import datetime
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    heart_samples = generate_random_heart(project)
    heart_samples['timestamp'] = pd.to_datetime(datetime.now())

    print(heart_samples)

    heart_fg = fs.get_feature_group(name="heart", version=1)

    # Get all user data
    user_fg = fs.get_feature_group(name="heart_user_dataset", version=1)
    df = user_fg.read()
    df["timestamp"] = pd.to_datetime(datetime.now())
    print(df)

    # Append synthetic data and user data to feature group
    all_samples = pd.concat([df, heart_samples])

    print(all_samples)

    heart_fg.insert(all_samples, write_options={"wait_for_job": False})

if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("heart_daily")
        with stub.run():
            f()