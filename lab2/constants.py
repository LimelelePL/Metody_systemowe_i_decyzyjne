from pathlib import Path

DATA_DIR = Path("resources")
DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

DIAMONDS_DATA_FILE = DATA_DIR / "diamonds.csv"

# Kolumny numeryczne
TENURE = "tenure"
MONTHLY_CHARGES = "MonthlyCharges"
TOTAL_CHARGES = "TotalCharges"
SENIOR_CITIZEN = "SeniorCitizen"
NUMERIC_COLUMNS = [
    TENURE,
    MONTHLY_CHARGES,
    TOTAL_CHARGES,
    SENIOR_CITIZEN,
]

# Kolumny kategorialne
GENDER = "gender"
PARTNER = "Partner"
DEPENDENTS = "Dependents"
PHONE_SERVICE = "PhoneService"
MULTIPLE_LINES = "MultipleLines"
INTERNET_SERVICE = "InternetService"
ONLINE_SECURITY = "OnlineSecurity"
ONLINE_BACKUP = "OnlineBackup"
DEVICE_PROTECTION = "DeviceProtection"
TECH_SUPPORT = "TechSupport"
STREAMING_TV = "StreamingTV"
STREAMING_MOVIES = "StreamingMovies"
CONTRACT = "Contract"
PAPERLESS_BILLING = "PaperlessBilling"
PAYMENT_METHOD = "PaymentMethod"

CATEGORICAL_COLUMNS = [
    GENDER,
    PARTNER,
    DEPENDENTS,
    PHONE_SERVICE,
    MULTIPLE_LINES,
    INTERNET_SERVICE,
    ONLINE_SECURITY,
    ONLINE_BACKUP,
    DEVICE_PROTECTION,
    TECH_SUPPORT,
    STREAMING_TV,
    STREAMING_MOVIES,
    CONTRACT,
    PAPERLESS_BILLING,
    PAYMENT_METHOD,
]

# Zmienna docelowa
CHURN = "Churn"

# identyfikator
CUSTOMER_ID = "customerID"

CARAT = "carat"
CUT = "cut"
COLOR = "color"
CLARITY = "clarity"
DEPTH = "depth"
TABLE = "table"
PRICE = "price"
X = "x"
Y = "y"
Z = "z"

D_NUMERIC = [CARAT, DEPTH, TABLE, PRICE, X, Y, Z]
D_CATEGORIAL = [CUT, COLOR, CLARITY]
