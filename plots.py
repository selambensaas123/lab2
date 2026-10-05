"""Plots for laboratory work 2."""

import pandas as pd
import matplotlib.pyplot as plt


def plot_brain_data():
    # Read the brain size dataset
    table = pd.read_csv(
        "brainsize.txt",
        sep="\t",
        na_values="NA"
    )

    # Separate women and men
    women = table[table["Gender"] == "Female"]
    men = table[table["Gender"] == "Male"]

    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height"
    ]

    # Create scatter plots for women
    for feature in features:
        plt.figure()

        plt.scatter(
            women[feature],
            women["MRI_Count"]
        )

        plt.xlabel(feature)
        plt.ylabel("MRI_Count")
        plt.title("Women: " + feature + " and MRI_Count")

        plt.show()

    # Create scatter plots for men
    for feature in features:
        plt.figure()

        plt.scatter(
            men[feature],
            men["MRI_Count"]
        )

        plt.xlabel(feature)
        plt.ylabel("MRI_Count")
        plt.title("Men: " + feature + " and MRI_Count")

        plt.show()

    # Calculate the correlation matrix
    correlation_table = table[
        [
            "FSIQ",
            "VIQ",
            "PIQ",
            "Weight",
            "Height",
            "MRI_Count"
        ]
    ].corr()

    # Create a simple correlation heatmap
    plt.figure()

    plt.imshow(
        correlation_table,
        cmap="coolwarm",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation_table.columns)),
        correlation_table.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlation_table.columns)),
        correlation_table.columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_brain_data()