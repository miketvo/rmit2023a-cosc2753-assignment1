import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker


def header(width: int, title: str) -> str:
    """
    Just a function to generate a pretty header for our output data
    :param width: The width (in characters) of the header
    :param title: The content (title) of the header
    :return: The generated header, as a string
    """
    calculated_width = width if len(title) + 4 <= width else len(title) + 4  # Auto padding
    border_top = '╔' + '═' * (calculated_width - 2) + '╗'
    border_bottom = '╚' + '═' * (calculated_width - 2) + '╝'
    border_vertical = '║'
    return f"{border_top}\n{border_vertical}{title : ^{calculated_width - 2}}{border_vertical}\n{border_bottom}"


def show_aggregate_distribution(train: pd.DataFrame, test: pd.DataFrame, save_path: str = None, save_dpi: int = 300):
    """
    Displays an overview histogram map (y-axis is probability density) of all features in train and test DataFrame,
    overlaid on top of each other (except for the target column).
    :param train: train DataFrame.
    :param test: test DataFrame.
    :param save_path: The location for the plot produced by this function to be saved at. Defaults to None (no saving).
    :param save_dpi: The quality of the saved plot. Only takes effect if save_path is not None.
    :return: None
    """
    plt.style.use('ggplot')

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), constrained_layout=True)
    axes = axes.ravel()

    for i, col in enumerate(train.columns):
        if col != "Sepsis":
            axes[i].hist(
                train[col],
                bins=min(train.shape[0] // 10, 18),
                color="royalblue",
                density=True,
                alpha=0.5,
                label="Train Data"
            )
            sns.kdeplot(train[col], ax=axes[i], color="blue", alpha=0.8, label="Train Data KDE")

            axes[i].hist(
                test[col],
                bins=min(test.shape[0] // 10, 18),
                color="crimson",
                density=True,
                alpha=0.5,
                label="Test Data"
            )
            sns.kdeplot(test[col], ax=axes[i], color="red", alpha=0.8, label="Test Data KDE")

            axes[i].set_ylabel("Probability density", fontsize=10)
            axes[i].set_xlabel("Value", fontsize=10)
            axes[i].set_xlim(left=min(test[col].min(), test[col].min()),
                             right=max(train[col].max(), train[col].max()))
        else:
            axes[i].hist(
                train[col],
                bins=2,
                color="sandybrown",
                alpha=0.8,
                label="Train Data Target"
            )
            axes[i].set_ylabel("Count", fontsize=10)
            axes[i].set_xticks([0.25, 0.75], labels=["Negative", "Positive"])
        axes[i].tick_params(axis="both", which="major", labelsize=10)
        axes[i].set_title(col, fontsize=14, fontweight="bold")
        axes[i].legend()

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi)


def show_distribution(col: str, train: pd.DataFrame, test: pd.DataFrame, save_path: str = None,
                      save_dpi: int = 300) -> None:
    """
    Displays comparison histograms and accompanying box plots for col of both train and test DataFrame. Then save that
    plot into the specified path, if it exists.
    :param col: Name of the column to be analyzed.
    :param train: train DataFrame.
    :param test: test DataFrame.
    :param save_path: The location for the plot produced by this function to be saved at. Defaults to None (no saving).
    :param save_dpi: The quality of the saved plot. Only takes effect if save_path is not None.
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), constrained_layout=True, sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})

    # Graph 1: Histogram
    axes[0].set_title(f"{col} Distribution")

    axes[0].set_ylabel("Count")
    axes[0].hist(
        train[col],
        bins=min(train.shape[0] // 10, 24),
        color="royalblue",
        alpha=0.5,
        label="Train Data"
    )

    axes[0].set_ylabel("Count")
    axes[0].hist(
        test[col],
        bins=min(test.shape[0] // 10, 24),
        color="crimson",
        alpha=0.5,
        label="Test Data"
    )

    axes[0].legend()

    # Graph 2: Box plot
    axes[1].boxplot([test[col], train[col]], vert=False, labels=["Test Data", "Train Data"], widths=0.5)
    plt.show()

    # Print out numerical distribution statistics
    df_prg_summary = pd.concat([train[[col]].describe(), test[[col]].describe()], axis=1)
    df_prg_summary.columns = [f"{col}_TRAIN", f"{col}_TEST"]
    print(df_prg_summary)

    # Count values outside of 1.5*IQR range (outliers)
    q1 = train[[col]].quantile(0.25)
    q2 = train[[col]].quantile(0.75)
    iqr = q2 - q1
    outliers_count_train = ((train[[col]] < (q1 - 1.5 * iqr)) | (train[[col]] > (q2 + 1.5 * iqr))).sum().sum()
    q1 = test[[col]].quantile(0.25)
    q2 = test[[col]].quantile(0.75)
    iqr = q2 - q1
    outliers_count_test = ((test[[col]] < (q1 - 1.5 * iqr)) | (test[[col]] > (q2 + 1.5 * iqr))).sum().sum()

    print(f"\nNumber of outliers in train data: {outliers_count_train}")
    print(f"Number of outliers in test data: {outliers_count_test}")

    if save_path is not None:
        fig.savefig(f"../images/EDA_Distribution_{col}.png", dpi=save_dpi)


def show_distribution_corr(feature_col: str, target_col: str, df: pd.DataFrame, title: str = None,
                           ticker_multiple_locator: int = 5, save_path: str = None, save_dpi: int = 300) -> None:
    """
    Show a comparison plot a target categorical binary distribution on a specified feature.
    classes.
    :param feature_col: Name of the feature column.
    :param target_col: Name of the target column.
    :param df: DataFrame containing feature and target column.
    :param title: Title for the plot, defaults to no title.
    :param ticker_multiple_locator: Set a tick on each integer multiple of the base within the view interval. Defaults
    to 5.
    :param save_path: The location for the plot produced by this function to be saved at. Defaults to None (no saving).
    :param save_dpi: The quality of the saved plot. Only takes effect if save_path is not None.
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 5), constrained_layout=True, sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})

    sns.kdeplot(df.loc[df[target_col] == 0, feature_col], fill=True, color="dodgerblue", label="Negative", ax=axes[0])
    sns.kdeplot(df.loc[df[target_col] == 1, feature_col], fill=True, color="red", label="Positive", ax=axes[0])
    axes[0].legend(title="Sepsis", loc="upper right", labels=["Negative", "Positive"])
    axes[0].set_ylabel("Probability density", fontsize=12)
    axes[0].xaxis.set_major_locator(
        ticker.MultipleLocator(
            ticker_multiple_locator
        )
    )
    if title is not None:
        axes[0].set_title(title, fontsize=16, fontweight="bold")

    axes[1].boxplot(
        [
            df.loc[df[target_col] == 0, feature_col],
            df.loc[df[target_col] == 1, feature_col]
        ],
        vert=False, labels=["Positive", "Negative"], widths=0.5
    )
    axes[1].set_xlabel(feature_col, fontsize=12)

    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi)


def show_boxplots(train: pd.DataFrame, test: pd.DataFrame, save_path: str = None, save_dpi: int = 300) -> None:
    """
    A wrapper for Seaborn boxplot to show distribution and outliers for both train and test data.
    :param train: train DataFrame.
    :param test: test DataFrame.
    :param save_path: The location for the plot produced by this function to be saved at. Defaults to None (no saving).
    :param save_dpi: The quality of the saved plot. Only takes effect if save_path is not None.
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    sns.boxplot(data=train, orient="h", ax=axes[0])
    sns.boxplot(data=test, orient="h", ax=axes[1])

    axes[0].set_title("Train Data")
    axes[1].set_title("Test Data")

    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi)


def count_zero_vals(df: pd.DataFrame, cols: list[str]) -> str:
    """
    Count the number of numerical values equal to zero in a dataset on specified columns
    :param df: The dataset to be processed
    :param cols: Selected column to be processed
    :return: A string report
    """
    head = f"{'':10}{'Count':>10}{'Percentage':>15}\n{'-' * 35}\n"
    rows = ""
    for col in cols:
        zeroes_count = 0
        if 0 in df[col].value_counts().index:
            zeroes_count = df[col].value_counts()[0]
        missing_percentage = zeroes_count / len(df) * 100
        rows += f"{col:10}{zeroes_count:>10}{missing_percentage:>14.2f}%\n"
    return head + rows
