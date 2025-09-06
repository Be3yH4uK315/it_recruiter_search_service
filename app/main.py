from fastapi import FastAPI
from .api.v1.search import router

app = FastAPI(title="Search Service")

app.include_router(router, prefix="/v1")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Search Service"}
