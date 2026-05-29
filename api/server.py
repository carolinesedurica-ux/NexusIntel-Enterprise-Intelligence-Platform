"""
NexusIntel — FastAPI REST Server
===================================
Endpoints:

  POST /intelligence          — Run a full multi-track intelligence pipeline
  POST /intelligence/gtm      — Run GTM track only
  POST /intelligence/finance  — Run Finance track only
  POST /intelligence/security — Run Security track only
  GET  /reports/{request_id}  — Retrieve a cached report (in-memory for demo)
  GET  /health                — Liveness probe

All endpoints accept an IntelligenceRequest body.
Results are returned synchronously; for large jobs consider a background
task queue (Celery / ARQ) in production.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory report cache (replace with Redis in production)
_report_cache: dict[str, Any] = {}

app = FastAPI(
    title="NexusIntel — Enterprise Intelligence Platform",
    description=(
        "Multi-track intelligence powered by Bright Data and Claude AI.\n\n"
        "**Track 1 — GTM Intelligence**: Competitor monitoring, buying signals, account enrichment.\n"
        "**Track 2 — Finance Intelligence**: Pricing surveillance, regulatory alerts, alternative data.\n"
        "**Track 3 — Security & Compliance**: Threat detection, compliance monitoring, vendor risk.\n"
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os as _os
_static_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_and_cache(request_dict: dict) -> dict:
    """Import lazily to avoid circular imports at module load time."""
    from orchestrator import run_intelligence_pipeline
    from models.schemas import IntelligenceRequest

    req = IntelligenceRequest(**request_dict)
    report = run_intelligence_pipeline(req)
    report_dict = report.model_dump(mode="json")
    _report_cache[report.request_id] = report_dict
    return report_dict


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_os.path.join(_static_dir, "index.html"))

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "NexusIntel"}


@app.post("/intelligence", status_code=status.HTTP_200_OK)
async def run_all_tracks(body: dict) -> JSONResponse:
    """
    Run all configured intelligence tracks for the given target.

    Example request body:
    ```json
    {
      "target": "Salesforce",
      "tracks": ["gtm", "finance", "security"],
      "context": "enterprise CRM market",
      "webhook_url": "https://your-crm.com/webhook/intelligence"
    }
    ```
    """
    try:
        report = _run_and_cache(body)
        return JSONResponse(content=report)
    except Exception as exc:
        logger.error("/intelligence error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/intelligence/gtm", status_code=status.HTTP_200_OK)
async def run_gtm_only(body: dict) -> JSONResponse:
    """Run only Track 1 — GTM Intelligence."""
    body["tracks"] = ["gtm"]
    return await run_all_tracks(body)


@app.post("/intelligence/finance", status_code=status.HTTP_200_OK)
async def run_finance_only(body: dict) -> JSONResponse:
    """Run only Track 2 — Finance Intelligence."""
    body["tracks"] = ["finance"]
    return await run_all_tracks(body)


@app.post("/intelligence/security", status_code=status.HTTP_200_OK)
async def run_security_only(body: dict) -> JSONResponse:
    """Run only Track 3 — Security & Compliance."""
    body["tracks"] = ["security"]
    return await run_all_tracks(body)


@app.get("/reports/{request_id}", status_code=status.HTTP_200_OK)
async def get_report(request_id: str) -> JSONResponse:
    """Retrieve a previously generated intelligence report by ID."""
    if request_id not in _report_cache:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")
    return JSONResponse(content=_report_cache[request_id])


@app.get("/reports", status_code=status.HTTP_200_OK)
async def list_reports() -> JSONResponse:
    """List all cached report IDs."""
    return JSONResponse(content={
        "count": len(_report_cache),
        "report_ids": list(_report_cache.keys()),
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import get_settings
    cfg = get_settings()
    uvicorn.run(
        "api.server:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.env != "production",
        log_level=cfg.log_level.lower(),
    )
