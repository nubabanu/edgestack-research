"""Authenticated, read-only FastAPI surface for the Android companion."""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from edgestack.disclaimer import DISCLAIMER
from edgestack.mobile.models import MobileSnapshot
from edgestack.mobile.service import (
    MobileSnapshotService,
    SnapshotUnavailableError,
    stable_etag,
)


def create_mobile_app(
    *,
    artifact_root: str | Path = "artifacts",
    campaign_id: str | None = None,
    bearer_token: str | None = None,
    demo: bool = False,
) -> FastAPI:
    """Create a no-mutation API with optional constant-time bearer auth."""

    if not demo and (bearer_token is None or len(bearer_token) < 24):
        raise ValueError("non-demo mobile API requires a 24+ character bearer token")
    service = MobileSnapshotService(artifact_root, campaign_id=campaign_id, demo=demo)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if bearer_token is None:
            return
        expected = f"Bearer {bearer_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(
        title="EdgeStack Mobile API",
        version="1.0.0",
        description="Read-only paper research evidence. No broker/order endpoints.",
        docs_url="/docs" if demo else None,
        redoc_url=None,
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "demo" if demo else "sealed"}

    @app.get(
        "/api/v1/mobile/snapshot",
        response_model=MobileSnapshot,
        dependencies=[Depends(authorize)],
    )
    def snapshot(response: Response) -> MobileSnapshot:
        try:
            payload = service.load()
        except SnapshotUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        response.headers["Cache-Control"] = "private, no-cache"
        response.headers["ETag"] = f'"{stable_etag(payload)}"'
        response.headers["X-EdgeStack-Disclaimer"] = DISCLAIMER[:160]
        return payload

    return app
