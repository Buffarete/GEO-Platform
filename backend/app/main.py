from fastapi import FastAPI
from .api import routes

app = FastAPI(title="GEO Platform")
app.include_router(routes.router)
