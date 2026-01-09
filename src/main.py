import uvicorn
from fastapi import FastAPI
from src.api.routes import router as api_router

app = FastAPI(title="TechDocAI API")

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "TechDocAI is Online"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
