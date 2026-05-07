FROM python:3.11-slim

WORKDIR /app

# Copier uniquement requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement ce qui est nécessaire
COPY streamlit_app.py ./streamlit_app.py

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]