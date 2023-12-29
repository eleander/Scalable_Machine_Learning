import numpy as np
import pandas as pd
import joblib

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