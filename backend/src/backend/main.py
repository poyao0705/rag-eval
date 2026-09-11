from fastapi import FastAPI

from backend.core.config import Settings

app = FastAPI()
settings = Settings()  # pyright: ignore[reportCallIssue]


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
