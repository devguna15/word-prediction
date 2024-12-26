from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# Include the routes
app.include_router(router)

@app.get("/")
def welcome():
    return {"message": "Welcome to the Word Prediction API"}
