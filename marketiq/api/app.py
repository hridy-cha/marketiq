"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from marketiq import __version__
from marketiq.api.routers import ops, stocks
from marketiq.config.settings import get_settings
from marketiq.data.db import init_db

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MarketIQ API", version=__version__,
                  description="Explainable stock research platform. Research tool only, not investment advice.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.api.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(stocks.router)
    app.include_router(ops.router)

    @app.on_event("startup")
    def _startup():
        init_db()
        logger.info("MarketIQ API %s using %s", __version__, settings.database.resolved_url())

    return app


app = create_app()
