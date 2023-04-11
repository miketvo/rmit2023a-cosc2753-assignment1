import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from utils.processing import DataCSV


class LogisticRegressionModel:
    """
    Automated training and predictions - Logistic Regression
    """

    def __init__(self, train_data: str, predict_data: str, random_state: int = None, imputer_n_neighbors: int = 30):
        """
        :param train_data: The path to the training dataset CSV file.
        :param predict_data: The path to the dataset CSV file to be predicted.
        :param imputer_n_neighbors: The number of neighbors to use in the KNN imputer to deal with outliers. Defaults to 30.
        """
        self.data = DataCSV(train_data, predict_data, imputer_n_neighbors)
        self.lambda_para = np.logspace(-5, 2, num=100)[78]

        ##############
        #  Training  #
        ##############

        # Separate features (X) and target (y)
        df_X = self.data.train.drop(["Sepsis"], axis=1)
        df_y = self.data.train[["Sepsis"]]

        # Splitting training data into a train set and a validation set
        train_X, val_X, train_y, val_y = train_test_split(
            df_X, df_y,
            shuffle=True,
            random_state=random_state,
            test_size=0.2
        )

        # Degree 4 Polynomial Features:
        poly = PolynomialFeatures(4)
        train_X_poly4 = poly.fit_transform(train_X)
        val_X_poly4 = poly.transform(val_X)
        scaler = StandardScaler()
        train_X_poly4 = scaler.fit_transform(train_X_poly4)
        val_X_poly4 = scaler.transform(val_X_poly4)

        # Model definition
        clf = LogisticRegression(
            solver="liblinear",  # Good for our small dataset
            penalty="l2",
            C=1 / self.lambda_para,
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state  # Reproducible result
        )

        # Cross validation to find the best model
        cv_results = cross_validate(
            clf,
            train_X_poly4,
            train_y.values.ravel(),
            cv=4,
            scoring="f1",
            return_estimator=True,
            return_train_score=True
        )
        self.estimator = cv_results["estimator"][np.argmax(cv_results["test_score"])]

        ##############################
        #  Performance Measurements  #
        ##############################

        self.performance = {
            "train_f1": cv_results["test_score"].mean(),
            "val_f1": f1_score(val_y, self.estimator.predict(val_X_poly4)),
        }

        ################
        #  Prediction  #
        ################
        pred_X = poly.fit_transform(self.data.predict)
        pred_X = scaler.fit_transform(pred_X)
        pred_y = self.estimator.predict(pred_X)
        pred_y = np.where(pred_y == 0.0, "Negative", "Positive")
        self.predictions = self.estimator.predict(pred_y)
