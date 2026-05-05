from fastapi import FastAPI

app = FastAPI(title="ML API")

@app.get("/health")
def health():
    return {"status": "ok"}