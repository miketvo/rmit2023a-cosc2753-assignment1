import pandas as pd
import numpy as np


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
