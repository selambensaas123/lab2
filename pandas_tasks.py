"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    # Read the CSV file
    table = pd.read_csv(data.csv_path)

    # Count the total number of rows
    row_count = len(table)

    # Count missing values in each column
    missing_by_column = {}

    for column in table.columns:
        missing_count = table[column].isna().sum()
        missing_by_column[column] = int(missing_count)

    # Count passengers older than 30
    adults_over_30_count = int((table["Age"] > 30).sum())

    # Calculate the mean age for each passenger class
    mean_age_by_pclass = {}

    age_means = table.groupby("Pclass")["Age"].mean()

    for pclass, mean_age in age_means.items():
        mean_age_by_pclass[int(pclass)] = float(mean_age)

    # Calculate the survival rate for each passenger class
    survival_rate_by_pclass = {}

    survival_rates = table.groupby("Pclass")["Survived"].mean()

    for pclass, rate in survival_rates.items():
        survival_rate_by_pclass[int(pclass)] = float(rate)

    # Find the five highest fares
    highest_fares = table["Fare"].nlargest(5).tolist()

    highest_fares = [float(fare) for fare in highest_fares]

    # Return all results
    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )