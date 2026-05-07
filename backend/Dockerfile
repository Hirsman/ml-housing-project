# =========================
# Étape 1 : Builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /build

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .

# Installer les packages dans /install
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================
# Étape 2 : Runtime
# =========================
FROM python:3.10-slim

WORKDIR /app

# Copier uniquement les dépendances installées
COPY --from=builder /install /usr/local

# Copier le code utile à l'inférence
COPY src/ ./src/
COPY backend/ ./backend/

# Copier le meilleur modèle entraîné
COPY artifacts/model_latest.joblib ./artifacts/

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Port FastAPI
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]