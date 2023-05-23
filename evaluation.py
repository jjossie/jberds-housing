from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import seaborn as sns

from enum import Enum


class SplitType(Enum):
    EVAL = "Evaluation"
    TEST = "Test"


class EvaluationSet:

    def __init__(self):
        self.models = pd.DataFrame({
            "id": [],
            "split": [],
            # "learning_rate": [],
            # "n_estimators": [],
            # "max_depth": [],
            "r2": [],
            "rmse": [],
        })

    def evaluate(self, model_id, split: SplitType, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print(f"evaluating {model_id}: \n\tR2: {r2}\n\tRMSE: {rmse}")

        self.models.loc[len(self.models)] = [model_id, split, r2, rmse]

    def report(self):
        sns.barplot(
            data=self.models,
            x="id",
            y="r2",
        )
        # sns.lmplot(
        #     data=self.models,
        #     x="learning_rate",
        #     y="r2",
        #     hue="split"
        # )
        # sns.lmplot(
        #     data=self.models,
        #     x="n_estimators",
        #     y="r2",
        #     hue="split"
        # )
        # sns.lmplot(
        #     data=self.models,
        #     x="max_depth",
        #     y="r2",
        #     hue="split"
        # )

    def create_subspaced_models(self, X: pd.DataFrame, y: pd.Series):
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.evaluate(i, SplitType.TEST, y_test, y_pred)
