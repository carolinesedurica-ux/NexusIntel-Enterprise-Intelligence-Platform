"""
Cross-track Synthesis Agent
==============================
Combines GTM, Finance, and Security intelligence into a single
executive-grade IntelligenceReport.

Responsibilities:
  1. Receive outputs from all three track agents
  2. Find cross-track connections (e.g. a competitor's hiring spike is
     also a buying signal AND a supply-chain risk indicator)
  3. Produce a unified executive_summary and top_priorities list
  4. Dispatch the completed report via configured alert webhooks
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from config import get_settings
from models.schemas import (
    FinanceIntelligence,
    GTMIntelligence,
    IntelligenceReport,
    SecurityIntelligence,
    Track,
)
from tools.ai_tools import AIAnalyzer
from tools.cognee_memory import get_memory

logger = logging.getLogger(__name__)
settings = get_settings()


class SynthesisAgent:
    """
    Combines all track outputs into a unified IntelligenceReport
    and delivers it to configured webhook targets.
    """

    def __init__(self) -> None:
        self._ai = AIAnalyzer()

    def run(
        self,
        request_id: str,
        target: str,
        tracks_run: list[Track],
        gtm: Optional[GTMIntelligence] = None,
        finance: Optional[FinanceIntelligence] = None,
        security: Optional[SecurityIntelligence] = None,
        errors: Optional[list[str]] = None,
    ) -> IntelligenceReport:
        """
        Synthesize all available track data into a final report.

        Args:
            request_id: Unique ID for this intelligence run.
            target:     The target organization/market analyzed.
            tracks_run: Which tracks were executed.
            gtm:        GTMIntelligence result (if Track.GTM was run).
            finance:    FinanceIntelligence result (if Track.FINANCE was run).
            security:   SecurityIntelligence result (if Track.SECURITY was run).
            errors:     Accumulated errors from all tracks.

        Returns:
            IntelligenceReport ready to deliver.
        """
        logger.info("[SynthesisAgent] Synthesizing report for '%s'", target)
        all_errors = list(errors or [])
        memory = get_memory()

        # ── Step 0: Recall historical context from Cognee ────────────────────
        memory_context = ""
        try:
            memory_context = memory.recall(
                target,
                "threat indicators, competitors, risk scores, compliance changes",
            )
            if memory_context:
                logger.info(
                    "[SynthesisAgent] Memory recalled %d chars of historical context for '%s'",
                    len(memory_context), target,
                )
        except Exception as exc:
            logger.debug("[SynthesisAgent] Memory recall skipped: %s", exc)

        # Build per-track summary strings for Claude
        gtm_summary = self._gtm_summary_text(gtm) if gtm else "Not run."
        finance_summary = self._finance_summary_text(finance) if finance else "Not run."
        security_summary = self._security_summary_text(security) if security else "Not run."

        # Claude cross-track synthesis — enriched with Cognee memory context
        executive_summary = ""
        top_priorities: list[str] = []
        memory_trends = ""
        try:
            synthesis = self._ai.synthesize_report(
                target, gtm_summary, finance_summary, security_summary,
                memory_context=memory_context,
            )
            executive_summary = synthesis.get("executive_summary", "")
            top_priorities = synthesis.get("top_priorities", [])
            memory_trends = synthesis.get("memory_trends", "")
        except Exception as exc:
            logger.warning("[SynthesisAgent] Synthesis failed: %s", exc)
            all_errors.append(f"synthesis:{exc}")
            parts = [s for s in [gtm_summary, finance_summary, security_summary] if s != "Not run."]
            executive_summary = " | ".join(parts[:2])

        # Prepend trend note to executive summary when memory is active
        if memory_trends and memory_context:
            executive_summary = f"[Memory trend: {memory_trends}] {executive_summary}"

        report = IntelligenceReport(
            request_id=request_id,
            target=target,
            tracks_run=tracks_run,
            gtm=gtm,
            finance=finance,
            security=security,
            executive_summary=executive_summary,
            top_priorities=top_priorities,
            memory_context=memory_context,
            errors=all_errors,
        )

        # Deliver via configured webhooks
        self._dispatch(report)

        # ── Store report in Cognee for future runs ───────────────────────────
        # Fire-and-forget: cognify() is slow and must not block the API response
        try:
            memory.store_report(report)
            logger.info("[SynthesisAgent] Report queued for Cognee memory storage")
        except Exception as exc:
            logger.debug("[SynthesisAgent] Memory store skipped: %s", exc)

        logger.info(
            "[SynthesisAgent] Report complete | priorities=%d memory=%s errors=%d",
            len(top_priorities),
            "active" if memory_context else "cold-start",
            len(all_errors),
        )
        return report

    # ── Summary text builders ─────────────────────────────────────────────────

    def _gtm_summary_text(self, gtm: GTMIntelligence) -> str:
        lines = [f"Target: {gtm.target}"]
        if gtm.competitors:
            names = ", ".join(c.company for c in gtm.competitors[:3])
            lines.append(f"Competitors profiled: {names}")
        if gtm.buying_signals:
            top_signal = max(gtm.buying_signals, key=lambda s: s.relevance_score)
            lines.append(f"Top buying signal: {top_signal.description[:120]}")
        if gtm.market_positioning:
            lines.append(f"Market position: {gtm.market_positioning[:200]}")
        if gtm.action_items:
            lines.append("Actions: " + "; ".join(gtm.action_items[:3]))
        return "\n".join(lines)

    def _finance_summary_text(self, fin: FinanceIntelligence) -> str:
        lines = [f"Target: {fin.target}", f"Risk score: {fin.risk_score:.1f}/10"]
        if fin.pricing_data:
            lines.append(f"Pricing data points collected: {len(fin.pricing_data)}")
        if fin.regulatory_alerts:
            lines.append(f"Regulatory alerts: {len(fin.regulatory_alerts)}")
            lines.append(f"Top alert: {fin.regulatory_alerts[0].title[:100]}")
        if fin.investment_summary:
            lines.append(f"Investment note: {fin.investment_summary[:200]}")
        return "\n".join(lines)

    def _security_summary_text(self, sec: SecurityIntelligence) -> str:
        lines = [
            f"Target: {sec.target}",
            f"Overall severity: {sec.overall_severity.value.upper()}",
            f"Threat indicators: {len(sec.threat_indicators)}",
            f"Compliance changes: {len(sec.compliance_changes)}",
            f"Brand exposures: {len(sec.brand_exposure)}",
        ]
        if sec.threat_indicators:
            worst = max(
                sec.threat_indicators,
                key=lambda t: ["low", "medium", "high", "critical"].index(t.severity.value),
            )
            lines.append(f"Highest threat: {worst.description[:120]}")
        if sec.summary:
            lines.append(f"Security brief: {sec.summary[:200]}")
        return "\n".join(lines)

    # ── Webhook delivery ──────────────────────────────────────────────────────

    def _dispatch(self, report: IntelligenceReport) -> None:
        """POST the report to all configured alert webhook URLs."""
        webhooks = settings.alert_webhooks
        if not webhooks:
            return

        payload = report.model_dump(mode="json")
        for url in webhooks:
            try:
                resp = httpx.post(url, json=payload, timeout=10.0)
                resp.raise_for_status()
                logger.info("[SynthesisAgent] Webhook delivered → %s (%d)", url, resp.status_code)
            except Exception as exc:
                logger.warning("[SynthesisAgent] Webhook failed %s: %s", url, exc)
