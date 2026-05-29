"""
NexusIntel — Cognee Agent Memory Layer
========================================
Persists every intelligence run as a knowledge graph, enabling:

  • Cross-run context recall: "What threats has Stripe faced historically?"
  • Trend detection: "Fastspring risk score rose from 5.2 → 7.8 across runs"
  • Entity relationships: targets ↔ competitors ↔ vendors ↔ threats
  • Accumulated compliance history per target

Uses Cognee 1.1.0 remember/recall API backed by:
  LLM:        Anthropic Claude (ANTHROPIC_API_KEY)
  Vector DB:  LanceDB (local, zero-infra)
  Graph DB:   Kuzu (local, bundled with Cognee)

Graceful degradation
--------------------
  All public methods return empty strings / silently log on any failure so
  the main intelligence pipeline is never blocked by memory errors.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Disable Cognee's multi-tenant auth for local/demo use
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")

# ── Cognee availability guard ─────────────────────────────────────────────────

_cognee_available = False
try:
    import cognee  # type: ignore
    _cognee_available = True
    logger.info("[CogneeMemory] cognee %s ready", cognee.__version__)
except ImportError:
    logger.warning(
        "[CogneeMemory] cognee not installed — memory disabled. "
        "Run: pip install cognee"
    )


def _run_async(coro: Any, timeout: int = 60) -> Any:
    """Run an async coroutine from synchronous code, safely."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


# ── CogneeMemory ──────────────────────────────────────────────────────────────

class CogneeMemory:
    """
    Persistent knowledge-graph memory for the NexusIntel agent pipeline.

    Each intelligence run is serialised to structured text, stored via
    cognee.remember(), and later retrieved with cognee.recall() to enrich
    subsequent analysis with historical context and trend awareness.
    """

    def __init__(self) -> None:
        self._ready = False
        if not _cognee_available:
            return
        self._configure()

    def _configure(self) -> None:
        try:
            from config import get_settings
            settings = get_settings()

            # Configure LLM — Anthropic Claude
            cognee.config.set_llm_provider("anthropic")
            cognee.config.set_llm_model(settings.claude_model)
            cognee.config.set_llm_api_key(settings.anthropic_api_key)

            # Configure vector store — LanceDB (local, no extra infra)
            cognee.config.set_vector_db_provider("lancedb")

            self._ready = True
            logger.info("[CogneeMemory] Configured — LLM=anthropic/%s VectorDB=lancedb",
                        settings.claude_model)
        except Exception as exc:
            logger.warning("[CogneeMemory] Configuration failed: %s — memory disabled", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def store_report(self, report: Any) -> None:
        """
        Persist an IntelligenceReport to the Cognee knowledge graph.

        Uses cognee.remember() which handles ingestion + graph building.
        Called fire-and-forget from a background thread so the API response
        is never delayed. Safe from both sync and async contexts.
        """
        if not self._ready:
            return
        try:
            text = self._report_to_text(report)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(self._store_blocking, text, report.target)
            executor.shutdown(wait=False)
        except Exception as exc:
            logger.warning("[CogneeMemory] store_report failed: %s", exc)

    def recall(self, target: str, query: str = "") -> str:
        """
        Query the knowledge graph for past intelligence about a target.

        Returns a plain-text passage (≤1 500 chars) summarising relevant
        historical context, or an empty string when nothing is found.
        """
        if not self._ready:
            return ""
        try:
            full_query = f"{target}: {query}" if query else target
            results = _run_async(self._recall_async(full_query), timeout=20)
            return self._format_recall(results)
        except Exception as exc:
            logger.debug("[CogneeMemory] recall failed: %s", exc)
            return ""

    def list_targets(self) -> list[str]:
        """Return targets that have been stored in Cognee memory."""
        if not self._ready:
            return []
        try:
            results = _run_async(
                cognee.recall("list all intelligence targets analysed"), timeout=15
            )
            parts = self._format_recall(results)
            return [line.strip("- ").strip() for line in parts.splitlines() if line.strip()][:20]
        except Exception as exc:
            logger.debug("[CogneeMemory] list_targets failed: %s", exc)
            return []

    # ── Async internals ───────────────────────────────────────────────────────

    def _store_blocking(self, text: str, target: str) -> None:
        """Runs in a background thread with its own event loop."""
        try:
            asyncio.run(self._store_async(text))
            logger.info("[CogneeMemory] Stored intelligence for target '%s'", target)
        except Exception as exc:
            logger.warning("[CogneeMemory] Background store failed for '%s': %s", target, exc)

    async def _store_async(self, text: str) -> None:
        await cognee.remember(text)

    async def _recall_async(self, query: str) -> Any:
        return await cognee.recall(query)

    # ── Result formatting ─────────────────────────────────────────────────────

    def _format_recall(self, results: Any) -> str:
        """
        Turn Cognee recall results into a compact context passage for Claude.
        Handles RememberResult objects, lists of dicts, and plain strings.
        """
        if not results:
            return ""

        parts: list[str] = []

        # Cognee 1.1.0 recall() returns a list of MemoryEntry / dict objects
        items = results if isinstance(results, list) else [results]
        for r in items[:8]:
            if hasattr(r, "text"):
                parts.append(str(r.text))
            elif hasattr(r, "content"):
                parts.append(str(r.content))
            elif hasattr(r, "node"):
                node = r.node
                text = getattr(node, "text", "") or getattr(node, "description", "")
                if text:
                    parts.append(str(text))
            elif isinstance(r, dict):
                text = (r.get("text") or r.get("content")
                        or r.get("description") or str(r))
                parts.append(str(text))
            else:
                parts.append(str(r))

        combined = "\n".join(p.strip() for p in parts if p.strip())
        return combined[:1500]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _report_to_text(self, report: Any) -> str:
        """
        Convert an IntelligenceReport to structured plain text for Cognee.
        Sections are explicitly labelled so the knowledge graph can extract
        named entities (targets, competitors, vendors, CVEs, regulations).
        """
        run_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "NexusIntel Intelligence Report",
            f"Target: {report.target}",
            f"Date: {run_date}",
            f"Request ID: {report.request_id}",
            "",
        ]

        # GTM
        if report.gtm:
            lines += ["=== GTM Intelligence ==="]
            if report.gtm.competitors:
                comps = ", ".join(c.company for c in report.gtm.competitors)
                lines.append(f"Competitors identified for {report.target}: {comps}")
                for c in report.gtm.competitors:
                    lines.append(f"Competitor: {c.company} ({c.domain})")
                    for sig in c.pricing_signals[:2]:
                        lines.append(f"  Pricing signal: {sig[:150]}")
                    for msg in c.messaging_themes[:2]:
                        lines.append(f"  Messaging: {msg[:150]}")
                    if c.summary:
                        lines.append(f"  Strategy: {c.summary[:200]}")
            if report.gtm.buying_signals:
                lines.append(f"Buying signals: {len(report.gtm.buying_signals)} detected")
                for s in report.gtm.buying_signals[:3]:
                    lines.append(f"  [{s.signal_type}] {s.description[:150]}")
            if report.gtm.market_positioning:
                lines.append(f"Market positioning: {report.gtm.market_positioning[:300]}")
            if report.gtm.action_items:
                lines.append("GTM action items: " + " | ".join(report.gtm.action_items[:3]))
            lines.append("")

        # Finance
        if report.finance:
            lines += ["=== Finance Intelligence ==="]
            lines.append(f"Financial risk score: {report.finance.risk_score:.1f}/10")
            for a in report.finance.regulatory_alerts[:3]:
                lines.append(f"  Regulatory alert: {a.title[:120]} — {a.summary[:150]}")
            for sig in report.finance.alternative_signals[:3]:
                lines.append(f"  Alt signal [{sig.signal_type}]: {sig.interpretation[:150]}")
            if report.finance.investment_summary:
                lines.append(f"Investment summary: {report.finance.investment_summary[:300]}")
            lines.append("")

        # Security
        if report.security:
            lines += ["=== Security & Compliance Intelligence ==="]
            sev = report.security.overall_severity.value.upper()
            lines.append(f"Security severity: {sev}")
            lines.append(f"Threat indicators: {len(report.security.threat_indicators)}")
            for t in report.security.threat_indicators[:6]:
                lines.append(
                    f"  Threat [{t.severity.value.upper()}] {t.indicator_type}: {t.description[:150]}"
                )
            for c in report.security.compliance_changes[:4]:
                lines.append(
                    f"  Compliance [{c.severity.value.upper()}] {c.regulation}: {c.action_required[:120]}"
                )
            for v in report.security.vendor_risks:
                lines.append(
                    f"  Vendor {v.vendor}: risk {v.risk_score:.1f}/10 — {v.recommendation[:120]}"
                )
            for e in report.security.brand_exposure[:3]:
                lines.append(f"  Exposure: {e[:150]}")
            if report.security.summary:
                lines.append(f"Security brief: {report.security.summary[:400]}")
            lines.append("")

        # Synthesis
        if report.executive_summary:
            lines += ["=== Executive Summary ===", report.executive_summary[:600], ""]
        if report.top_priorities:
            lines += ["=== Top Priorities ==="]
            for p in report.top_priorities:
                lines.append(f"  Priority: {p}")

        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_memory: Optional[CogneeMemory] = None


def get_memory() -> CogneeMemory:
    """Return the shared CogneeMemory singleton (lazy-initialised)."""
    global _memory
    if _memory is None:
        _memory = CogneeMemory()
    return _memory
