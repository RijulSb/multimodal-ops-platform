#check the overall status and health of operations
#just a system performance dynamics check

from fastapi import APIRouter
from api.schemas import HealthResponse
import os

router = APIRouter()

#Make sure that the system is performing 'ok'
#also display the development version and status when in app


@router.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='ok',
        version=os.getenv("APP_VERSION", "0.0.1"),
        environment=os.getenv("APP_ENV", 'development')
    )