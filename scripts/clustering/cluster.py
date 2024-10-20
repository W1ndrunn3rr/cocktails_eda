import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.cluster import AffinityPropagation


class Cluster:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.X = self._transform_data()
        self.best_params = self._grid_search()
        self._cluster_data()

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

    def _grid_search(self):

        param_grid = {
            "preference": np.linspace(-50, 0, 10),
            "damping": np.linspace(0.5, 0.95, 10),
        }

        best_score = -1
        best_params = {}

        for preference in param_grid["preference"]:
            for damping in param_grid["damping"]:
                score = affinity_propagation_silhouette(self.X, preference, damping)
                if score > best_score:
                    best_score = score
                    best_params = {"preference": preference, "damping": damping}

        return best_params

    def _cluster_data(self):
        af = AffinityPropagation(
            preference=self.best_params["preference"],
            damping=self.best_params["damping"],
            max_iter=500,
            random_state=42,
        ).fit(self.X)

        labels = af.labels_
        cluster_centers_indices = af.cluster_centers_indices_
        n_clusters_ = len(cluster_centers_indices)

        silhouette_score = metrics.silhouette_score(
            self.X, labels, metric="sqeuclidean"
        )

        davies_bouldin_score = metrics.davies_bouldin_score(self.X, labels)

        print("---------------------\nWłaściwości modelu:")
        print(f"Preference: {self.best_params['preference']}")
        print(f"Damping: {self.best_params['damping']}")
        print(f"Liczba klastrów: {n_clusters_}")
        print(f"Indeksy centrów klastrów.: {cluster_centers_indices}")
        print(f"Wskaźnik Silhouette: {silhouette_score:.4f}")
        print(f"Wynik Davisa-Bouldina:  {davies_bouldin_score}\n---------------------")

        self.data["cluster"] = af.labels_

        self.data.to_csv("cluster.csv")


def affinity_propagation_silhouette(X, preference, damping):
    af = AffinityPropagation(preference=preference, damping=damping)
    labels = af.fit_predict(X)
    unique_labels = len(set(labels))
    if 2 <= unique_labels <= len(X) - 1:
        score = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    else:
        score = -1
    return score
