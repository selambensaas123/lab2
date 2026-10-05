import pandas as pd

from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput


# -------------------------
# Titanic tests
# -------------------------

def test_titanic_row_count():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    assert result.row_count == len(table)


def test_titanic_missing_values():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    for column in table.columns:
        expected = int(table[column].isna().sum())

        assert result.missing_by_column[column] == expected


def test_titanic_over_30():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    expected = int((table["Age"] > 30).sum())

    assert result.adults_over_30_count == expected


def test_titanic_mean_age():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    expected = table.groupby("Pclass")["Age"].mean()

    for pclass, mean_age in expected.items():
        assert abs(
            result.mean_age_by_pclass[int(pclass)] - float(mean_age)
        ) < 0.000001


def test_titanic_survival_rate():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    expected = table.groupby("Pclass")["Survived"].mean()

    for pclass, rate in expected.items():
        assert abs(
            result.survival_rate_by_pclass[int(pclass)] - float(rate)
        ) < 0.000001


def test_titanic_highest_fares():
    data = TitanicInput(csv_path="titanic.csv")

    result = analyze_titanic(data)

    table = pd.read_csv("titanic.csv")

    expected = table["Fare"].nlargest(5).tolist()

    assert result.highest_fares == expected


# -------------------------
# Brain correlation tests
# -------------------------

def test_brain_gender_count():
    data = BrainDataInput(csv_path="brainsize.txt")

    result = analyze_brain_correlations(data)

    table = pd.read_csv(
        "brainsize.txt",
        sep="\t",
        na_values="NA"
    )

    expected_women = len(
        table[table["Gender"] == "Female"]
    )

    expected_men = len(
        table[table["Gender"] == "Male"]
    )

    assert result.women_count == expected_women
    assert result.men_count == expected_men


def test_women_correlations():
    data = BrainDataInput(csv_path="brainsize.txt")

    result = analyze_brain_correlations(data)

    table = pd.read_csv(
        "brainsize.txt",
        sep="\t",
        na_values="NA"
    )

    women = table[
        table["Gender"] == "Female"
    ]

    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height"
    ]

    for feature in features:
        expected = women[feature].corr(
            women["MRI_Count"],
            method="pearson"
        )

        actual = result.women_mri_correlation[feature]

        assert abs(actual - expected) < 0.000001


def test_men_correlations():
    data = BrainDataInput(csv_path="brainsize.txt")

    result = analyze_brain_correlations(data)

    table = pd.read_csv(
        "brainsize.txt",
        sep="\t",
        na_values="NA"
    )

    men = table[
        table["Gender"] == "Male"
    ]

    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height"
    ]

    for feature in features:
        expected = men[feature].corr(
            men["MRI_Count"],
            method="pearson"
        )

        actual = result.men_mri_correlation[feature]

        assert abs(actual - expected) < 0.000001


def test_strongest_correlation():
    data = BrainDataInput(csv_path="brainsize.txt")

    result = analyze_brain_correlations(data)

    all_correlations = {}

    for feature in result.women_mri_correlation:
        value = abs(
            result.women_mri_correlation[feature]
        )

        all_correlations[feature + "_women"] = value

    for feature in result.men_mri_correlation:
        value = abs(
            result.men_mri_correlation[feature]
        )

        all_correlations[feature + "_men"] = value

    strongest = max(
        all_correlations,
        key=all_correlations.get
    )

    expected_feature = strongest.split("_")[0]

    assert result.strongest_mri_feature == expected_feature