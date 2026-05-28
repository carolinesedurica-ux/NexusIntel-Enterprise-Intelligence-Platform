"""
Track 2 — Finance & Market Intelligence Agent
================================================
Responsibilities:
  1. Scrape competitor/market pricing pages via Web Unlocker
  2. Monitor regulatory sources for compliance alerts
  3. Aggregate alternative data signals (job velocity, news sentiment)
  4. Analyze with Claude to produce structured financial intelligence
  5. Return a FinanceIntelligence object

Bright Data tools used:
  - Web Unlocker (pricing pages, regulatory sites)
  - SERP API (regulatory news, alternative signals)
  - Web Scraper API (LinkedIn company & jobs datasets)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from models.schemas import (
    AlternativeDataSignal,
    FinanceIntelligence,
    PriceDataPoint,
    RegulatoryAlert,
)
from tools.brightdata_tools import SERPApi, WebScraperApi, WebUnlocker
from tools.ai_tools import AIAnalyzer

logger = logging.getLogger(__name__)

# Bright Data dataset IDs
_LINKEDIN_COMPANY_DATASET = "gd_l1vikfch9l3vvkz96"
_INDEED_DATASET_ID = "gd_m794d2gkl1kl8oo3j"

# Regulatory sources to monitor
_REGULATORY_SOURCES = [
    ("SEC Edgar", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20"),
    ("FTC News", "https://www.ftc.gov/news-events/news/press-releases"),
    ("GDPR", "https://gdpr-info.eu/"),
]


class FinanceAgent:
    """
    Produces Finance & Market intelligence for a target company.

    run(target, markets, context) → FinanceIntelligence
    """

    def __init__(self) -> None:
        self._serp = SERPApi()
        self._unlocker = WebUnlocker()
        self._scraper_api = WebScraperApi()
        self._ai = AIAnalyzer()

    def run(
        self,
        target: str,
        markets: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> FinanceIntelligence:
        """
        Full Finance intelligence pipeline.

        Args:
            target:  Company name or market sector to analyze.
            markets: Additional markets/regions to monitor pricing in.
            context: Optional focus ("supply chain risk", "pricing", etc.).

        Returns:
            FinanceIntelligence with pricing, regulatory alerts, and alt signals.
        """
        logger.info("[FinanceAgent] Starting finance intelligence for '%s'", target)
        errors: list[str] = []

        # ── Step 1: Scrape pricing ────────────────────────────────────────────
        pricing_data: list[PriceDataPoint] = []
        try:
            pricing_data = self._collect_pricing(target, markets or [])
        except Exception as exc:
            logger.warning("[FinanceAgent] Pricing collection failed: %s", exc)
            errors.append(f"pricing:{exc}")

        # ── Step 2: Regulatory monitoring ────────────────────────────────────
        regulatory_alerts: list[RegulatoryAlert] = []
        try:
            regulatory_alerts = self._monitor_regulatory(target)
        except Exception as exc:
            logger.warning("[FinanceAgent] Regulatory monitoring failed: %s", exc)
            errors.append(f"regulatory:{exc}")

        # ── Step 3: Alternative data signals ─────────────────────────────────
        alt_signals: list[AlternativeDataSignal] = []
        try:
            alt_signals = self._collect_alternative_signals(target)
        except Exception as exc:
            logger.warning("[FinanceAgent] Alt data collection failed: %s", exc)
            errors.append(f"alt_signals:{exc}")

        # ── Step 4: Risk scoring & investment summary ─────────────────────────
        risk_score = 0.0
        investment_summary = ""
        try:
            risk_score, investment_summary = self._score_and_summarize(
                target, pricing_data, regulatory_alerts, alt_signals
            )
        except Exception as exc:
            logger.warning("[FinanceAgent] Risk scoring failed: %s", exc)
            errors.append(f"risk_score:{exc}")

        result = FinanceIntelligence(
            target=target,
            pricing_data=pricing_data,
            regulatory_alerts=regulatory_alerts,
            alternative_signals=alt_signals,
            risk_score=risk_score,
            investment_summary=investment_summary,
        )
        logger.info(
            "[FinanceAgent] Done | pricing=%d regulatory=%d alt=%d errors=%d",
            len(pricing_data),
            len(regulatory_alerts),
            len(alt_signals),
            len(errors),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_pricing(
        self, target: str, markets: list[str]
    ) -> list[PriceDataPoint]:
        """Scrape pricing pages for the target and key competitors."""
        points: list[PriceDataPoint] = []
        domain = target.lower().replace(" ", "") + ".com"
        urls = [
            f"https://{domain}/pricing",
            f"https://{domain}/plans",
        ]
        for market in markets[:2]:
            urls.append(f"https://{domain}/pricing/{market.lower()}")

        for url in urls:
            try:
                html = self._unlocker.fetch_with_retry(url)
                analysis = self._ai.analyze_pricing_page(target, html)
                for product in analysis.get("products", []):
                    points.append(
                        PriceDataPoint(
                            source=url,
                            product=product.get("name", "unknown"),
                            price=product.get("price"),
                            currency=product.get("currency", "USD"),
                            metadata={
                                "tier": product.get("tier", ""),
                                "billing_period": product.get("billing_period", ""),
                            },
                        )
                    )
                if points:
                    break
            except Exception as exc:
                logger.debug("[FinanceAgent] Pricing URL failed %s: %s", url, exc)

        # Supplement with SERP pricing signals
        try:
            serp_results = self._serp.google_search(
                f"{target} pricing plans cost 2024 2025", num_results=5
            )
            for r in serp_results[:3]:
                if r.get("snippet"):
                    points.append(
                        PriceDataPoint(
                            source=r.get("url", "serp"),
                            product=target,
                            metadata={"snippet": r.get("snippet", ""), "title": r.get("title", "")},
                        )
                    )
        except Exception:
            pass

        return points

    def _monitor_regulatory(self, target: str) -> list[RegulatoryAlert]:
        """Check regulatory sources for relevant alerts."""
        alerts: list[RegulatoryAlert] = []

        # SERP-based regulatory search (broad, fast)
        queries = [
            f"{target} regulatory compliance fine penalty 2024 2025",
            "data privacy GDPR CCPA enforcement action 2025",
            "SEC enforcement action fintech 2025",
        ]
        for query in queries:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results[:3]:
                    snippet = r.get("snippet", "")
                    if any(
                        kw in snippet.lower()
                        for kw in ["fine", "penalty", "violation", "enforcement", "action", "required"]
                    ):
                        alerts.append(
                            RegulatoryAlert(
                                source=r.get("url", ""),
                                title=r.get("title", ""),
                                summary=snippet,
                                url=r.get("url"),
                            )
                        )
            except Exception:
                pass

        # Direct scrape of one regulatory source
        try:
            source_name, source_url = _REGULATORY_SOURCES[0]
            html = self._unlocker.fetch_with_retry(source_url)
            analysis = self._ai.analyze_regulatory_content(source_name, html)
            for alert_data in analysis.get("alerts", [])[:5]:
                alerts.append(
                    RegulatoryAlert(
                        source=source_name,
                        title=alert_data.get("regulation", ""),
                        summary=alert_data.get("summary", ""),
                        url=source_url,
                        impact=alert_data.get("action_required", ""),
                    )
                )
        except Exception as exc:
            logger.debug("[FinanceAgent] Regulatory scrape failed: %s", exc)

        return alerts[:10]

    def _collect_alternative_signals(self, target: str) -> list[AlternativeDataSignal]:
        """Gather job velocity and news sentiment as alternative data."""
        signals: list[AlternativeDataSignal] = []

        # Job postings via Indeed dataset
        job_titles: list[str] = []
        try:
            rows = self._scraper_api.collect_and_wait(
                _INDEED_DATASET_ID,
                [{"keyword": target, "location": "United States", "country": "US"}],
                timeout_seconds=60,
            )
            job_titles = [r.get("job_title", "") for r in rows[:20] if r.get("job_title")]
        except Exception as exc:
            logger.debug("[FinanceAgent] Indeed dataset skipped: %s", exc)

        # LinkedIn company profile via Scraper API
        linkedin_data: list[dict] = []
        try:
            linkedin_data = self._scraper_api.collect_and_wait(
                _LINKEDIN_COMPANY_DATASET,
                [{"url": f"https://www.linkedin.com/company/{target.lower().replace(' ', '-')}"}],
                timeout_seconds=60,
            )
        except Exception as exc:
            logger.debug("[FinanceAgent] LinkedIn dataset skipped: %s", exc)

        # News via SERP
        news_titles: list[str] = []
        try:
            results = self._serp.google_search(
                f"{target} funding layoff expansion acquisition 2024 2025", num_results=10
            )
            news_titles = [r.get("title", "") for r in results if r.get("title")]
        except Exception:
            pass

        # Claude interpretation
        if job_titles or news_titles:
            interpretation = self._ai.interpret_alternative_signals(
                target, job_titles, news_titles
            )
            for sig in interpretation.get("signals", []):
                signals.append(
                    AlternativeDataSignal(
                        signal_type=sig.get("type", "unknown"),
                        company=target,
                        value=sig.get("value", ""),
                        interpretation=sig.get("interpretation", ""),
                    )
                )

        return signals

    def _score_and_summarize(
        self,
        target: str,
        pricing: list[PriceDataPoint],
        regulatory: list[RegulatoryAlert],
        alt_signals: list[AlternativeDataSignal],
    ) -> tuple[float, str]:
        """Derive a risk score and investment summary."""
        data = {
            "target": target,
            "pricing_count": len(pricing),
            "regulatory_alerts": len(regulatory),
            "alt_signals": [s.interpretation for s in alt_signals[:5]],
            "regulatory_summaries": [a.summary for a in regulatory[:3]],
        }
        result = self._ai._analyze_json(
            self._ai._FINANCE_SYSTEM,
            (
                f"Data: {data}\n\n"
                "Return JSON: {\"risk_score\": float 0-10, \"investment_summary\": str (150 words max)}"
            ),
            max_tokens=300,
        )
        risk_score = float(result.get("risk_score", 0.0))
        risk_score = max(0.0, min(10.0, risk_score))
        return risk_score, result.get("investment_summary", "")
