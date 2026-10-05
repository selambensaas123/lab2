"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    # Read the table using tab as a separator
    table = pd.read_csv(
        data.csv_path,
        sep="\t",
        na_values="NA"
    )

    # Separate women and men
    women = table[table["Gender"] == "Female"]
    men = table[table["Gender"] == "Male"]

    # Count women and men
    women_count = len(women)
    men_count = len(men)

    # Features that will be compared with MRI_Count
    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height"
    ]

    # Calculate correlations for women
    women_mri_correlation = {}

    for feature in features:
        correlation = women[feature].corr(
            women["MRI_Count"],
            method="pearson"
        )

        women_mri_correlation[feature] = float(correlation)

    # Calculate correlations for men
    men_mri_correlation = {}

    for feature in features:
        correlation = men[feature].corr(
            men["MRI_Count"],
            method="pearson"
        )

        men_mri_correlation[feature] = float(correlation)

    # Find the strongest correlation
    strongest_mri_feature = ""
    strongest_value = -1

    for feature in features:
        women_value = abs(women_mri_correlation[feature])

        if women_value > strongest_value:
            strongest_value = women_value
            strongest_mri_feature = feature

        men_value = abs(men_mri_correlation[feature])

        if men_value > strongest_value:
            strongest_value = men_value
            strongest_mri_feature = feature

    # Return all results
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )