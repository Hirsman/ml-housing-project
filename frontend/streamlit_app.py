import os

import requests
import streamlit as st

# =========================
# CONFIG BACKEND
# =========================
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# =========================
# UI
# =========================
st.title("🏠 Prédiction du prix immobilier")
st.write("Saisissez les caractéristiques du quartier pour obtenir une estimation.")

# =========================
# USER ID (AJOUT IMPORTANT A/B TESTING)
# =========================
user_id = st.text_input("User ID", value="anonymous")

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Revenu médian (en 10k$)", value=3.5, step=0.1)
    house_age = st.number_input("Âge médian des maisons", value=20.0, step=1.0)
    ave_rooms = st.number_input("Nombre moyen de pièces", value=5.0, step=0.1)
    ave_bedrms = st.number_input("Nombre moyen de chambres", value=1.0, step=0.1)

with col2:
    population = st.number_input("Population du quartier", value=1000.0, step=50.0)
    ave_occup = st.number_input("Occupation moyenne", value=3.0, step=0.1)
    latitude = st.number_input("Latitude", value=34.0, format="%.2f")
    longitude = st.number_input("Longitude", value=-118.0, format="%.2f")

# =========================
# PREDICTION
# =========================
if st.button("🔮 Calculer l'estimation", type="primary"):
    payload = {
        "user_id": user_id,  # 🔥 AJOUT A/B TESTING
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    try:
        with st.spinner("Calcul en cours..."):
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)

            response.raise_for_status()
            result = response.json()

        prediction = result.get("prediction")
        variant = result.get("variant")  # 🔥 AJOUT

        st.success(f"### 💰 Prix estimé : {prediction:.2f} (x100k$)")
        st.metric("Estimation finale", f"{prediction * 100_000:,.0f} $")

        # 🔥 affichage A/B test
        st.info(f"🧪 Modèle utilisé : {variant}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de contacter le backend FastAPI.")
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
