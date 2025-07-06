from fastapi import FastAPI
from app.routers import routers

app = FastAPI()
app.include_router(routers.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)