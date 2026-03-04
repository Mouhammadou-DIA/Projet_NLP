"""
FastAPI Application - Professional Reddit RAG Chatbot
Production-ready REST API with OpenAPI documentation
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config.logging_config import get_logger, log_request, log_shutdown, log_startup
from src.config.settings import settings
from src.models.schemas import ErrorResponse


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events

    Handles startup and shutdown logic
    """
    # Startup
    log_startup()
    logger.info("Initializing services...")

    # Initialize services here if needed
    # await initialize_services()

    logger.info("Application ready")

    yield

    # Shutdown
    log_shutdown()
    logger.info("Cleaning up resources...")

    # Cleanup here if needed
    # await cleanup_services()

    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ==================== MIDDLEWARE ====================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (time.time() - start_time) * 1000  # ms

    # Log request
    log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
    )

    # Add custom headers
    response.headers["X-Process-Time"] = f"{duration:.2f}ms"

    return response


# Error handling middleware
@app.middleware("http")
async def error_handler(request: Request, call_next):
    """Global error handler"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {e!s}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(e) if settings.DEBUG else "An unexpected error occurred",
            ).dict(),
        )


# ==================== EXCEPTION HANDLERS ====================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}",
        ).dict(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            code="VALIDATION_ERROR",
        ).dict(),
    )


# ==================== ROUTES ====================

# Import routes
from api.routes import chat, health


# Register routes
app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"],
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health"],
)


# Root endpoint (API info)
@app.get("/api", include_in_schema=False)
async def api_root():
    """API root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled in production",
    }


# Serve frontend (static files)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ==================== STARTUP MESSAGE ====================


@app.on_event("startup")
async def startup_message():
    """Print startup message"""
    print("\n" + "=" * 70)
    print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 70)
    print(f"API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"ReDoc: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug: {settings.DEBUG}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS if not settings.DEBUG else 1,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
