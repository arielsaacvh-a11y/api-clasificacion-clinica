import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =========================
# CARGA DATASET
# =========================
df = pd.read_csv(
    "Fibrosis-pulmonar-predicción.csv",
    encoding="latin-1"
)

df.columns = df.columns.str.strip()

# =========================
# TARGET
# =========================
target_col = "Clase_de_enfermedad"

# =========================
# FEATURES
# =========================
feature_cols = [
    "Edad",
    "GÃ©nero",
    "IMC",
    "Peso_corporal_kg",
    "Estado_de_obesidad",
    "Antecedentes_familiares",
    "Marcadores_genÃ©ticos",
    "Ãndice_microbiano",
    "Enfermedades_autoinmunes",
    "Smoking_Status",
    "Alcohol_Use",
    "Stress_Level",
    "Physical_Activity",
    "Abdominal_Pain",
    "Bloating",
    "Diarrhea",
    "Constipation",
    "Sangrado_rectal",
    "PÃ©rdida_de_apetito",
    "PÃ©rdida_de_peso",
    "Frecuencia_de_deposiciones",
    "Uso_AINEs",
    "Uso_antibiÃ³ticos",
    "Uso_IBP"
]

X = df[feature_cols].copy()
y = df[target_col]

# =========================
# ENCODING
# =========================
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = le.fit_transform(X[col].astype(str))

y = le.fit_transform(y.astype(str))

# =========================
# TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# BALANCEO
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# ESCALADO
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# =========================
# MODELO
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# GUARDAR
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
