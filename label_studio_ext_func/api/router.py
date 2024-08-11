from fastapi import APIRouter

from api.resources.label_studio import router as lb


router = APIRouter()

router.include_router(lb)
