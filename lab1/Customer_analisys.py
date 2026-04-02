from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import CHURN, DATA_FILE, MONTHLY_CHARGES, SENIOR_CITIZEN, TENURE, TOTAL_CHARGES
from pandas import DataFrame


def load_file(path: Path) -> DataFrame:
    return pd.read_csv(path)


def convert_to_numeric(df: DataFrame) -> DataFrame:
    numeric_columns = [TENURE, MONTHLY_CHARGES, TOTAL_CHARGES, SENIOR_CITIZEN]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def numeric_statistics(df: DataFrame) -> DataFrame:
    numeric_columns = [TENURE, MONTHLY_CHARGES, TOTAL_CHARGES, SENIOR_CITIZEN]
    numeric_data = df[numeric_columns]

    statistics = pd.DataFrame(
        {
            "srednia": numeric_data.mean(),
            "mediana": numeric_data.median(),
            "min": numeric_data.min(),
            "max": numeric_data.max(),
            "odchylenie_standardowe": numeric_data.std(),
        }
    )

    return statistics.round(2)


def tenure_histogram(df: DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=TENURE, bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram: czas trwania umowy")
    plt.xlabel("czas trwania umowy")
    plt.ylabel("Liczba klientow")
    plt.tight_layout()
    plt.show()


def tenure_churn_boxplot(df: DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=CHURN, y=TENURE)
    plt.title("Boxplot: czas trwania umowy wzgledem rezygnacji")
    plt.xlabel("Rezygnacja")
    plt.ylabel("czas trwania umowy")
    plt.tight_layout()
    plt.show()


def monthly_charges_histogram(df: DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=MONTHLY_CHARGES, bins=20, color="orange", edgecolor="black")
    plt.title("Histogram: miesieczna oplata")
    plt.xlabel("miesieczna oplata")
    plt.ylabel("Liczba klientow")
    plt.tight_layout()
    plt.show()


def monthly_charges_churn_boxplot(df: DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=CHURN, y=MONTHLY_CHARGES)
    plt.title("Boxplot: miesieczna oplata wzgledem rezygnacji")
    plt.xlabel("Rezygnacja")
    plt.ylabel("miesieczna oplata")
    plt.tight_layout()
    plt.show()


def tenure_monthly_charges_scatter(df: DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=TENURE, y=MONTHLY_CHARGES, hue=CHURN, alpha=0.7)
    plt.title("Wykres rozrzutu: czas trwania umowy i miesieczna oplata")
    plt.xlabel("czas trwania umowy")
    plt.ylabel("miesieczna oplata")
    plt.legend(title="Rezygnacja")
    plt.tight_layout()
    plt.show()


def main() -> None:
    data_frame = load_file(DATA_FILE)
    data_frame = convert_to_numeric(data_frame)

    print("---- PODGLAD -------")
    print(data_frame.head())
    print()

    print("---- INFO O KOLUMNACH ------")
    data_frame.info()
    print()

    print("--- PODSTAWOWE STATYSTYKI ----")
    print(numeric_statistics(data_frame))
    print()

    tenure_histogram(data_frame)
    tenure_churn_boxplot(data_frame)
    monthly_charges_histogram(data_frame)
    monthly_charges_churn_boxplot(data_frame)
    tenure_monthly_charges_scatter(data_frame)


if __name__ == "__main__":
    main()
