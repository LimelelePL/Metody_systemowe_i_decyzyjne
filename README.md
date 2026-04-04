# Metody systemowe i decyzyjne

Projekt do pierwszej części zajęć. Zawiera analizę danych, prosty system regułowy oraz wersję regresyjną.

## Zakres projektu

W repozytorium są dwa główne przykłady:

1. `Telco Customer Churn` - problem klasyfikacji.
2. `Diamonds` - problem regresji.

Oba zbiory mają cechy liczbowe i kategorialne, więc nadają się do EDA, kodowania danych i budowania prostych reguł decyzyjnych.

## Co jest w projekcie

- wczytanie danych z plików CSV,
- analiza numeryczna i kategorialna,
- wykresy pokazujące najciekawsze zależności,
- statystyki opisowe,
- podział na zbiór treningowy i testowy,
- proste systemy decyzyjne oparte na regułach `if/else`,
- ocena jakości modeli na zbiorze testowym.

## Struktura

- `lab1/zadania.ipynb` - główny notebook z analizą i modelami,
- `lab1/Customer_analisys.py` - pomocniczy skrypt do analizy danych Telco,
- `lab1/constants.py` - stałe z nazwami kolumn i ścieżkami do danych,
- `lab1/resources/` - pliki CSV z danymi,
- `src/app/main.py` - prosty punkt startowy aplikacji,
- `pyproject.toml` - konfiguracja projektu i zależności dla `uv`.

## Dane

W projekcie używam:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv` dla klasyfikacji,
- `diamonds.csv` dla regresji.

Pliki powinny znajdować się w `lab1/resources/`.

## Jak uruchomić

Najprościej przez `uv`:

```powershell
uv sync
```

```powershell
uv run jupyter notebook
```

Jeśli chcesz użyć zwykłego `pip`:

```powershell
pip install -r requirements.txt
```

Potem możesz uruchomić notebook lub skrypt Python bezpośrednio.

## Wymagania

Zależności do uruchomienia projektu są zapisane w `requirements.txt` i `pyproject.toml`.

