import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    :param train: train DataFrame or Numpy Array.
    :param test: test DataFrame or Numpy Array.
    :param save_path: The location for the plot produced by this function to be saved at. Defaults to None (no saving).
    :param save_dpi: The quality of the saved plot. Only takes effect if save_path is not None.
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5), constrained_layout=True, sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})

    # Graph 1: Histogram
    axes[0, 0].set_title(f"{col} Distribution - Train Data")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].hist(
        train[col],
        bins=min(train.shape[0] // 10, 24),
        color="royalblue",
        alpha=0.9,
    )

    axes[0, 1].set_title(f"{col} Distribution - Test Data")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].hist(
        test[col],
        bins=min(test.shape[0] // 10, 24),
        color="crimson",
        alpha=0.8,
    )

    # Graph 2: Box plot
    axes[1, 0].boxplot(train[col], vert=False)
    axes[1, 0].set_yticks([])
    axes[1, 1].boxplot(test[col], vert=False)
    axes[1, 1].set_yticks([])
    plt.show()

    # Print out numerical distribution statistics
    df_prg_summary = pd.concat([train[[col]].describe(), test[[col]].describe()], axis=1)
    df_prg_summary.columns = [f"{col}_TRAIN", f"{col}_TEST"]
    print(df_prg_summary)

    if save_path is not None:
        fig.savefig(f"../images/EDA_Distribution_{col}.png", dpi=save_dpi)
