from sklearn.model_selection import train_test_split

from utils.processing import DataCSV


class LogisticRegression:
    """
    :param train_data_path: The path to the training dataset file.
    :param test_data_path: The path to the test dataset file.
    :param imputer_n_neighbors: The number of neighbors to use in the KNN imputer to deal with outliers. Defaults to 30.
    """

    def __init__(self, train_data_path: str, test_data_path: str, imputer_n_neighbors: int = 30):
        self.data = DataCSV(train_data_path, test_data_path, imputer_n_neighbors)

        # Separate our features (X) and target (y)
        df_X = self.data.train.drop(["Sepsis"], axis=1)
        df_y = self.data.train[["Sepsis"]]

        # Splitting our training data into a train set and a validation set
        train_X, val_X, train_y, val_y = train_test_split(
            df_X, df_y,
            shuffle=True,
            random_state=0,  # Ensure reproducible results
            test_size=0.2
        )


