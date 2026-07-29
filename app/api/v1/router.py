from fastapi import APIRouter

from app.api.v1.endpoints import simulation, scheduler, forecasting, datasets

api_router = APIRouter()

api_router.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
api_router.include_router(scheduler.router, prefix="/scheduler", tags=["Scheduler"])
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["Forecasting"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
