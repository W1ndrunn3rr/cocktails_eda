import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Claster:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.X = self._transform_data()

    def get_X(self):
        return self.X

    def _transform_data(self) -> np.array:
        categorical_features = ["glass"]

        binary_features = ["Cocktail", "Ordinary Drink", "Punch / Party Drink"]
        numerical_features = [
            "ingredients",
            "instruction_length",
            "mean_alcohol_percentage",
            "complexity_score",
        ]
        pca_features = [
            "ingredient_pc1",
            "ingredient_pc2",
            "ingredient_pc3",
            "ingredient_pc4",
            "ingredient_pc5",
        ]

        transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                (
                    "cat",
                    OneHotEncoder(drop="first"),
                    categorical_features,
                ),
                ("pca", "passthrough", pca_features),
            ]
        )

        pipeline = Pipeline([("transformer", transformer)])

        X = pipeline.fit_transform(self.data)

        X_final = np.column_stack((X, self.data[binary_features]))

        return X_final

    def _claster_data():
        pass
