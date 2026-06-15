import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ocr as ocr_router
from app.routes import downtime as downtime_router
from app.routes import job as job_router
from app.services.ocr_service import OCRService
from app.services.downtime_service import DowntimeService
from app.services.job_service import JobService
from app.db.database import init_db

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Machine Count Read Backend Starting ===")
    init_db()  # SQLite by default, MS SQL via MCR_DATABASE_URL
    app.state.ocr_service = OCRService()  # EasyOCR model loads here
    app.state.downtime_service = DowntimeService(app.state.ocr_service)
    app.state.job_service = JobService(app.state.ocr_service)
    logger.info("=== Startup complete. Ready to accept requests. ===")

    yield  # App runs here

    logger.info("=== Machine Count Read Backend Shutting Down ===")
    app.state.job_service.shutdown()
    app.state.downtime_service.shutdown()
    app.state.ocr_service.shutdown()


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Machine Count Read API",
    description=(
        "OCR-based machine counter reading\n\n"
        "- `/ocr/...` — Read fast-changing machine counter values from a "
        "camera / video / screen using EasyOCR + OpenCV"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(ocr_router.router)
app.include_router(downtime_router.router)
app.include_router(job_router.router)


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "system": "Machine Count Read"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
