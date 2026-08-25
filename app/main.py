import logging
from contextlib import asynccontextmanager

import structlog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import AgentScalarConfig, get_scalar_api_reference
from app.config import get_settings
from app.routers import estimations, estimations_text, sessions


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development.

    Every phase of an LLM call (cache check, dispatch, retry, fallback,
    success/failure) logs through this configuration, so a single request can
    be traced end-to-end when reviewing logs.
    """
    settings = get_settings()
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.APP_ENV == "production"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        # Filters by LOG_LEVEL before a log line is even built.
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.LOG_LEVEL)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown. Settings() validation (at least one API
    key configured) runs here so the app fails fast instead of erroring on the
    first request."""
    configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    log.info(
        "application_started",
        environment=settings.APP_ENV,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        prompt_version=settings.PROMPT_VERSION,
    )
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="CAG Estimator",
    description="API for software project estimation using CAG architecture",
    version="0.5.0",
    docs_url=None, # In order to deactivate /docs swagger default
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(estimations.router)
app.include_router(estimations_text.router)
app.include_router(sessions.router)

@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        telemetry=False,
        agent=AgentScalarConfig(disabled=True),
        overrides={"mcp": {"disabled": True}},
    )

@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "cag-estimator"}
    )