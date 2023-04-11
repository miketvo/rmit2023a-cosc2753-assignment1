import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from utils.visualization import show_aggregate_distribution, show_boxplots


class Data:
    def __init__(self, train_data_path: str, test_data_path: str, imputer_n_neighbors: int = 30):
        self.__source_train = pd.read_csv(train_data_path)
        self.__source_test = pd.read_csv(test_data_path)

        # Preprocessing
        self.__train = self.__source_train.drop_duplicates()
        self.__train = self.__train.drop(columns=['ID', 'Insurance'])
        self.__test = self.__source_test.drop(columns=['ID', 'Insurance'])
        self.__train = self.__source_train.drop_duplicates(inplace=True)

        self.__train['PRG'] = self.__train['PRG'].replace(0, self.__train['PRG'].mean())
        self.__train['PL'] = self.__train['PL'].replace(0, self.__train['PL'].mean())
        self.__train['PR'] = self.__train['PR'].replace(0, self.__train['PR'].mean())
        self.__train['SK'] = self.__train['SK'].replace(0, self.__train['SK'].mean())
        self.__train['TS'] = self.__train['TS'].replace(0, self.__train['TS'].mean())
        self.__train['M11'] = self.__train['M11'].replace(0, self.__train['M11'].mean())
        self.__train['BD2'] = self.__train['BD2'].replace(0, self.__train['BD2'].mean())

        self.__train = self.__train.rename(columns={"Sepssis": "Sepsis"})
        self.__train["Sepsis"] = self.__train["Sepsis"].map({"Negative": 0.0, "Positive": 1.0})

        # Processing
        for col in self.__train.columns:
            if col == "Sepsis":  # Target column does not have outliers, as it is categorical
                continue
            impute_outliers_iqr(col, self.__train, KNNImputer(n_neighbors=imputer_n_neighbors))
            cap_outliers_iqr(col, self.__train)

        scaler = StandardScaler()
        added_const = 0.001
        for col in self.__train.columns:
            if col == "Sepsis":
                continue
            self.__train[col] = np.where(self.__train[col] != 0, np.log(self.__train[col] + added_const), 0)
            self.__train[col] = np.where(self.__train[col] != 0, np.log(self.__train[col] + added_const), 0)
            self.__train[[col]] = scaler.fit_transform(self.__train[[col]])
            self.__train[[col]] = scaler.fit_transform(self.__train[[col]])

        negatives = self.__train[self.__train["Sepsis"] == 0]
        positives = self.__train[self.__train["Sepsis"] == 1]
        positives_up_sampled = resample(
            positives,
            replace=True,  # sample with replacement
            n_samples=len(negatives),  # 1:1 balanced
            random_state=0  # reproducible results
        )

        self.__train = pd.concat([negatives, positives_up_sampled])

    def show_distribution(self) -> None:
        show_aggregate_distribution(self.__train, self.__test)

    def show_boxplots(self) -> None:
        show_boxplots(self.__train, self.__test)


def impute_outliers_iqr(col: str, df: pd.DataFrame, imputer, whisker_width: float = 1.5) -> None:
    """
    Imputes outliers in the given column (in-place) using the interquartile range (IQR) method.
    :param col: Name of the column to impute outliers in.
    :param df: Pandas dataframe containing the column.
    :param imputer: An imputer object that will be used to impute the missing values.
    :param whisker_width: The multiplier used to calculate the lower and upper bounds for outlier detection.
    Defaults to 1.5.
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
