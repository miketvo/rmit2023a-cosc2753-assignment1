import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from utils.visualization import show_aggregate_distribution, show_boxplots


class DataCSV:
    """
    A class for processing CSV data for machine leaning algorithms.
    """

    def __init__(self, train_data_path: str, predict_data_path: str, imputer_n_neighbors: int = 30):
        """
        A class to automate cleaning and processing data.
        :param train_data_path: The path to the training dataset file.
        :param predict_data_path: The path to the test dataset file.
        :param imputer_n_neighbors: The number of neighbors to use in the KNN imputer to deal with outliers. Defaults to 30.
        """
        self.source_train = pd.read_csv(train_data_path)
        self.source_predict = pd.read_csv(predict_data_path)

        # Cleaning
        self.train = self.source_train.drop_duplicates(inplace=False)
        self.train = self.train.drop(columns=['ID', 'Insurance'])
        self.predict = self.source_predict.drop(columns=['ID', 'Insurance'])
        self.train.drop_duplicates(inplace=True)

        self.train['PRG'] = self.train['PRG'].replace(0, self.train['PRG'].mean())
        self.train['PL'] = self.train['PL'].replace(0, self.train['PL'].mean())
        self.train['PR'] = self.train['PR'].replace(0, self.train['PR'].mean())
        self.train['SK'] = self.train['SK'].replace(0, self.train['SK'].mean())
        self.train['TS'] = self.train['TS'].replace(0, self.train['TS'].mean())
        self.train['M11'] = self.train['M11'].replace(0, self.train['M11'].mean())
        self.train['BD2'] = self.train['BD2'].replace(0, self.train['BD2'].mean())

        self.train = self.train.rename(columns={"Sepssis": "Sepsis"})
        self.train["Sepsis"] = self.train["Sepsis"].map({"Negative": 0.0, "Positive": 1.0})

        # Processing
        for col in self.train.columns:
            if col == "Sepsis":  # Target column does not have outliers, as it is categorical
                continue
            impute_outliers_iqr(col, self.train, KNNImputer(n_neighbors=imputer_n_neighbors))
            cap_outliers_iqr(col, self.train)

        scaler = StandardScaler()
        added_const = 0.001
        for col in self.train.columns:
            if col == "Sepsis":
                continue
            self.train[col] = np.where(self.train[col] != 0, np.log(self.train[col] + added_const), 0)
            self.predict[col] = np.where(self.predict[col] != 0, np.log(self.predict[col] + added_const), 0)
            self.train[[col]] = scaler.fit_transform(self.train[[col]])
            self.predict[[col]] = scaler.fit_transform(self.predict[[col]])

        negatives = self.train[self.train["Sepsis"] == 0]
        positives = self.train[self.train["Sepsis"] == 1]
        positives_up_sampled = resample(
            positives,
            replace=True,  # sample with replacement
            n_samples=len(negatives),  # 1:1 balanced
            random_state=0  # reproducible results
        )

        self.train = pd.concat([negatives, positives_up_sampled])

    def show_distribution(self) -> None:
        """
        Displays an overview histogram map (y-axis is probability density) of all features in train and test DataFrame,
        overlaid on top of each other (except for the target column).
        :return: None
        """
        show_aggregate_distribution(self.train, self.predict)

    def show_boxplots(self) -> None:
        """
        Shows comparison histograms and accompanying box plots for col of both train and test DataFrame.
        :return: None
        """
        show_boxplots(self.train, self.predict)


def impute_outliers_iqr(col: str, df: pd.DataFrame, imputer, whisker_width: float = 1.5) -> None:
    """
    Imputes outliers in the given column (in-place) using the interquartile range (IQR) method.
    :param col: Name of the column to impute outliers in.
    :param df: Pandas dataframe containing the column.
    :param imputer: An imputer object that will be used to impute the missing values.
    :param whisker_width: The multiplier used to calculate the lower and upper bounds for outlier detection. Defaults to 1.5.
    :return: None
    """
    # Calculate IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    # Replace outliers with NaN
    df[col] = df[col].where((df[col] >= lower_bound) & (df[col] <= upper_bound), np.nan)

    # Use imputer
    df[[col]] = imputer.fit_transform(df[[col]])


def cap_outliers_iqr(col: str, df: pd.DataFrame, whisker_width: float = 1.5) -> None:
    """
    Caps the outliers in a given column of a pandas DataFrame (in-place) using the interquartile range (IQR) method.
    :param col: A string representing the name of the column to cap outliers.
    :param df: A pandas DataFrame containing the data.
    :param whisker_width: A float representing the width of the whisker used in the IQR method. Default is 1.5.
    :return: None
    """
    # Calculate IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    # Cap outliers
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
