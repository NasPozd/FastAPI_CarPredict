from sklearn.base import BaseEstimator, TransformerMixin


class CombinedTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, threshold):
        self.threshold = threshold
        self.value_counts_ = None
        self.unique_brands_ = set()

    def fit(self, X, y=None):
        # заполнение пропусков медианой
        self.medians_ = X.select_dtypes(include=["float64", "int64"]).median()
        # подсчет брендов
        self.value_counts_ = X["name"].apply(
            lambda x: x.split(" ")[0]).value_counts()
        self.unique_brands_ = set(self.value_counts_.index)
        return self

    def transform(self, X):
        # преобразование категориальных данных
        for col in ["mileage", "engine", "max_power"]:
            X[col] = X[col].fillna("").str.extract("(\d+\.?\d*)").astype(float)

        X[["torque", "max_torque_rpm"]] = (X["torque"].str.replace(
            ",",
            "").str.extract(r"^([\d.,]+).*?([\d.,]+)(?=\D*$)").astype(float))

        # заполнение пропусков
        X[X.select_dtypes(
            include=["float64", "int64"]).columns] = X.select_dtypes(
                include=["float64", "int64"]).apply(
                    lambda x: x.fillna(x.median()))

        # преобразование типов
        X["engine"] = X["engine"].astype(int)
        X["seats"] = X["seats"].astype(int)

        # извлечение марки
        X["brand"] = X["name"].apply(lambda x: x.split(" ")[0])
        X["brand"] = X["brand"].apply(
            lambda x: (x if (x in self.unique_brands_ and self.value_counts_.
                             get(x, 0) >= self.threshold) else "Other"))
        X = X.drop(columns=["name"])

        # создание новых признаков
        X["power_per_litre"] = X["max_power"] / X["engine"]
        X["km_per_year"] = X["km_driven"] / (2024 - X["year"])
        X["is_first_owner"] = X["owner"].apply(lambda x: 1
                                               if x == "First Owner" else 0)
        X["is_test_drive_car"] = X["owner"].apply(
            lambda x: 1 if x == "Test Drive Car" else 0)
        return X
