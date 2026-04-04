# Metody systemowe i decyzyjne

Repozytorium do realizacji projektu z analizy danych i prostych systemow decyzyjnych opartych na regulach `if/else`.

## Cel czesci I

W tej czesci projektu przygotowujemy:

1. wczytanie danych,
2. eksploracyjna analize danych (EDA),
3. podzial na zbior treningowy i testowy,
4. prosty klasyfikator oparty na regulach,
5. rozbudowany system regulowy,
6. opcjonalnie wersje regresyjna dla oceny 5.0.

## Proponowany wybor zbiorow danych

- Klasyfikacja: `Heart Disease Dataset` albo `Telco Customer Churn`
- Regresja: `Medical Cost Personal Datasets` albo `Diamonds`

Ten podzial jest wygodny, bo zbiory zawieraja jednoczesnie cechy liczbowe i kategorialne.

## Struktura projektu

- `lab1/` - materialy do pierwszej czesci projektu
- `lab1/resources/` - miejsce na pliki z danymi
- `lab1/project_template.py` - prosty kod startowy
- `src/app/main.py` - minimalny punkt startowy projektu

## Instalacja

Jesli korzystasz z `uv`:

```powershell
uv sync
```

Jesli chcesz uruchomic notebook:

```powershell
uv run jupyter notebook
```

## Co dalej

Najrozsadniejsza kolejnosc prac:

1. wybrac zbior danych do klasyfikacji,
2. wgrac go do `lab1/resources/`,
3. uzupelnic analize EDA,
4. zbudowac 1 prosta regule,
5. rozbudowac system do co najmniej 5 regul,
6. na koncu przygotowac raport.

## Uwaga

Od momentu podzialu na zbior treningowy i testowy wszystkie decyzje analityczne powinny byc oparte tylko na zbiorze treningowym.
