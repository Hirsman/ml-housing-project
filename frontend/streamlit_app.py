import logging
import os
import random

import requests
import streamlit as st

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
PREDICT_URL = f"{BACKEND_URL}/predict"

logger.info("Demarrage Streamlit - BACKEND_URL=%s", BACKEND_URL)


# =========================
# HELPERS
# =========================
def build_payload(
    user_id: str,
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float,
) -> dict:
    """Construit le payload attendu par l'API FastAPI."""

    return {
        "user_id": user_id,
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }


# =========================
# UI HEADER
# =========================
st.title("🏠 A/B Testing - Prédiction Immobilière")

st.write("""
Cette interface envoie les données utilisateur au backend FastAPI.

Le backend :
- choisit automatiquement la variante A ou B
- sélectionne le modèle ML correspondant
- loggue les expérimentations A/B
""")

# =========================
# USER ID
# =========================
st.header("1️⃣ Identifiant utilisateur")

user_id = st.text_input(
    "User ID",
    value="alice",
    help="Même user_id = même variante A/B",
)

# =========================
# FEATURES
# =========================
st.header("2️⃣ Caractéristiques du logement")

col1, col2 = st.columns(2)

with col1:
    median_income = st.number_input(
        "Revenu médian",
        value=3.5,
        step=0.1,
    )

    housing_median_age = st.number_input(
        "Âge médian des logements",
        value=20.0,
        step=1.0,
    )

    average_rooms = st.number_input(
        "Nombre moyen de pièces",
        value=5.0,
        step=0.1,
    )

    average_bedrooms = st.number_input(
        "Nombre moyen de chambres",
        value=1.0,
        step=0.1,
    )

with col2:
    population = st.number_input(
        "Population",
        value=1000.0,
        step=50.0,
    )

    average_occupancy = st.number_input(
        "Occupation moyenne",
        value=3.0,
        step=0.1,
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0,
        format="%.2f",
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0,
        format="%.2f",
    )

# =========================
# BUILD PAYLOAD
# =========================
payload = build_payload(
    user_id=user_id,
    MedInc=median_income,
    HouseAge=housing_median_age,
    AveRooms=average_rooms,
    AveBedrms=average_bedrooms,
    Population=population,
    AveOccup=average_occupancy,
    Latitude=latitude,
    Longitude=longitude,
)

# =========================
# SINGLE PREDICTION
# =========================
if st.button("🔮 Lancer une prédiction", type="primary"):
    logger.info(
        "Nouvelle requete de prediction - user_id=%s",
        user_id,
    )

    try:
        with st.spinner("Calcul en cours..."):
            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=20,
            )

            response.raise_for_status()

            result = response.json()

        logger.info(
            "Prediction recue - request_id=%s variant=%s model=%s",
            result.get("request_id"),
            result.get("variant"),
            result.get("model_version"),
        )

        # =========================
        # DISPLAY RESULTS
        # =========================
        st.success("✅ Prédiction effectuée avec succès")

        prediction = result.get("prediction", 0)

        st.metric(
            "Prix estimé",
            f"{prediction * 100_000:,.0f} $",
        )

        st.subheader("🧪 Informations A/B Testing")

        st.write(
            "Variante utilisée :",
            result.get("variant", "unknown"),
        )

        st.write(
            "Version du modèle :",
            result.get("model_version", "unknown"),
        )

        st.write(
            "Mode d'exécution :",
            result.get("execution_mode", "unknown"),
        )

        st.write(
            "Latence :",
            f"{result.get('latency_ms', 'unknown')} ms",
        )

        st.write(
            "Request ID :",
            result.get("request_id", "unknown"),
        )

    except requests.exceptions.ConnectionError:
        logger.exception("Connexion backend impossible")

        st.error("❌ Impossible de contacter le backend FastAPI.")

    except Exception as exc:
        logger.exception("Erreur inattendue")

        st.error(f"❌ Une erreur est survenue : {exc}")

# =========================
# TRAFFIC SIMULATION
# =========================
st.header("3️⃣ Simulation de trafic A/B")

st.write(
    "Permet de générer du trafic artificiel afin de "
    "tester la distribution des variantes."
)

if st.button("🚀 Simuler 100 utilisateurs"):
    success = 0

    variant_counts = {
        "A": 0,
        "B": 0,
    }

    rng = random.Random()  # noqa: S311

    progress_bar = st.progress(0)

    for i in range(100):
        simulated_payload = build_payload(
            user_id=f"user_{i}",
            MedInc=round(
                rng.uniform(1.0, 8.0),
                2,
            ),
            HouseAge=float(rng.randint(1, 50)),
            AveRooms=round(
                rng.uniform(2.0, 8.0),
                2,
            ),
            AveBedrms=round(
                rng.uniform(1.0, 3.0),
                2,
            ),
            Population=float(rng.randint(200, 5000)),
            AveOccup=round(
                rng.uniform(1.0, 6.0),
                2,
            ),
            Latitude=round(
                rng.uniform(32.0, 42.0),
                2,
            ),
            Longitude=round(
                rng.uniform(-124.0, -114.0),
                2,
            ),
        )

        try:
            response = requests.post(
                PREDICT_URL,
                json=simulated_payload,
                timeout=20,
            )

            response.raise_for_status()

            result = response.json()

            variant = result.get("variant", "unknown")

            if variant in variant_counts:
                variant_counts[variant] += 1

            success += 1

        except Exception as exc:
            logger.warning(
                "Echec requete simulation : %s",
                exc,
            )

        progress_bar.progress((i + 1) / 100)

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.success(f"✅ {success} requêtes envoyées avec succès")

    st.subheader("📊 Répartition observée")

    st.write(variant_counts)

    st.bar_chart(variant_counts)
