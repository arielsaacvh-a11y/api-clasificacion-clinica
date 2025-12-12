from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API Clasificación Clínica")

# =========================
# CORS (OBLIGATORIO PARA WORDPRESS)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego puedes restringir al dominio WP
    allow_credentials=True,
    allow_methods=["*"],  # IMPORTANTE: permite OPTIONS
    allow_headers=["*"],
)

# =========================
# MODELO
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# SCHEMA
# =========================
class Paciente(BaseModel):
    Edad: int
    Género: int
    IMC: float
    Peso_corporal_kg: float
    Estado_de_obesidad: int
    Antecedentes_familiares: int
    Marcadores_genéticos: int
    Índice_microbiano: float
    Enfermedades_autoinmunes: int
    Smoking_Status: int
    Alcohol_Use: int
    Stress_Level: int
    Physical_Activity: int
    Abdominal_Pain: int
    Bloating: int
    Diarrhea: int
    Constipation: int
    Sangrado_rectal: int
    Pérdida_de_apetito: int
    Pérdida_de_peso: int
    Frecuencia_de_deposiciones: int
    Uso_AINEs: int
    Uso_antibióticos: int
    Uso_IBP: int

# =========================
# ENDPOINT
# =========================
@app.post("/predecir")
def predecir(p: Paciente):
    X = np.array([[  
        p.Edad,
        p.Género,
        p.IMC,
        p.Peso_corporal_kg,
        p.Estado_de_obesidad,
        p.Antecedentes_familiares,
        p.Marcadores_genéticos,
        p.Índice_microbiano,
        p.Enfermedades_autoinmunes,
        p.Smoking_Status,
        p.Alcohol_Use,
        p.Stress_Level,
        p.Physical_Activity,
        p.Abdominal_Pain,
        p.Bloating,
        p.Diarrhea,
        p.Constipation,
        p.Sangrado_rectal,
        p.Pérdida_de_apetito,
        p.Pérdida_de_peso,
        p.Frecuencia_de_deposiciones,
        p.Uso_AINEs,
        p.Uso_antibióticos,
        p.Uso_IBP
    ]])

    X = scaler.transform(X)
    pred = model.predict(X)[0]

    return {
        "clasificacion": int(pred)
    }
