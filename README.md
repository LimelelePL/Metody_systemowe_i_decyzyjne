# Metody systemowe i decyzyjne lista 1

Projekt do pierwszej części zajęć. Pokazuje analizę danych, statystykę opisową oraz proste systemy decyzyjne oparte na regułach.

## Zbiory danych

W projekcie wykorzystano dwa zbiory danych, ponieważ służą one do dwóch różnych zadań:

1. `Telco Customer Churn` - klasyfikacja.
2. `Diamonds` - regresja.

### `Telco Customer Churn`

To zbiór opisujący klientów firmy telekomunikacyjnej i informujący, czy klient zrezygnował z usług (`Churn`).

### `Diamonds`

To zbiór opisujący diamenty i ich cenę (`price`).

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
- `lab1/constants.py` - stałe z nazwami kolumn i ścieżkami do danych,
- `lab1/resources/` - pliki CSV z danymi,
- `src/app/main.py` - prosty punkt startowy aplikacji,
- `pyproject.toml` - konfiguracja projektu i zależności dla `uv`.

Najprosciej przez `uv`:

```powershell
uv sync
```

```powershell
uv run jupyter notebook
```

Jesli chcesz uzyc zwyklego `pip`:

```powershell
pip install -r requirements.txt
```

Potem mozesz uruchomic notebook lub skrypt Python bezposrednio.

## Wymagania

Zaleznosci do uruchomienia projektu sa zapisane w `requirements.txt` i `pyproject.toml`.
