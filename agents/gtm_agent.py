"""
Track 1 — GTM Intelligence Agent
===================================
Responsibilities:
  1. Run SERP searches for competitor activity and buying signals
  2. Scrape competitor websites via Web Unlocker
  3. Collect job postings via Web Scraper API (Indeed dataset)
  4. Analyze results with Claude to extract GTM intelligence
  5. Return a fully structured GTMIntelligence object

Bright Data tools used:
  - SERP API (competitor news, buying signals)
  - Web Unlocker (competitor pricing pages)
  - Web Scraper API (job postings dataset)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from models.schemas import (
    BuyingSignal,
    CompetitorProfile,
    GTMIntelligence,
)
from tools.brightdata_tools import SERPApi, ScrapingBrowser, WebScraperApi, WebUnlocker, run_coro
from tools.ai_tools import AIAnalyzer

logger = logging.getLogger(__name__)

# Bright Data dataset ID for Indeed job postings
_INDEED_DATASET_ID = "gd_m794d2gkl1kl8oo3j"


class GTMAgent:
    """
    Produces GTM intelligence for a target company or market segment.

    run(target, competitors, context) → GTMIntelligence
    """

    def __init__(self) -> None:
        self._serp = SERPApi()
        self._unlocker = WebUnlocker()
        self._browser = ScrapingBrowser()
        self._scraper_api = WebScraperApi()
        self._ai = AIAnalyzer()

    def run(
        self,
        target: str,
        competitors: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> GTMIntelligence:
        """
        Full GTM intelligence pipeline.

        Args:
            target:       Company name or market segment to analyze.
            competitors:  Known competitor names. If empty, auto-discovered via SERP.
            context:      Optional focus area (e.g. "enterprise sales", "SMB pricing").

        Returns:
            GTMIntelligence with competitors, buying signals, and recommendations.
        """
        logger.info("[GTMAgent] Starting GTM intelligence for '%s'", target)
        errors: list[str] = []

        # ── Step 1: Discover competitors ──────────────────────────────────────
        competitor_list = list(competitors or [])
        if not competitor_list:
            competitor_list = self._discover_competitors(target)

        # ── Step 2: Profile each competitor ──────────────────────────────────
        competitor_profiles: list[CompetitorProfile] = []
        for comp in competitor_list[:4]:  # cap at 4 to respect rate limits
            try:
                profile = self._profile_competitor(comp, target)
                competitor_profiles.append(profile)
            except Exception as exc:
                logger.warning("[GTMAgent] Failed to profile '%s': %s", comp, exc)
                errors.append(f"competitor_profile:{comp}:{exc}")

        # ── Step 3: Detect buying signals ────────────────────────────────────
        buying_signals: list[BuyingSignal] = []
        try:
            buying_signals = self._detect_buying_signals(target, context)
        except Exception as exc:
            logger.warning("[GTMAgent] Buying signal detection failed: %s", exc)
            errors.append(f"buying_signals:{exc}")

        # ── Step 4: Collect market intelligence via SERP ──────────────────────
        market_positioning = ""
        action_items: list[str] = []
        try:
            market_positioning, action_items = self._market_analysis(
                target, competitor_profiles, buying_signals
            )
        except Exception as exc:
            logger.warning("[GTMAgent] Market analysis failed: %s", exc)
            errors.append(f"market_analysis:{exc}")

        # ── Step 5: Generate executive summary ───────────────────────────────
        summary = ""
        try:
            raw_data = {
                "competitors": [p.model_dump(mode="json") for p in competitor_profiles],
                "buying_signals": [s.model_dump(mode="json") for s in buying_signals],
                "market_positioning": market_positioning,
            }
            summary = self._ai.generate_gtm_summary(target, raw_data)
        except Exception as exc:
            logger.warning("[GTMAgent] Summary generation failed: %s", exc)
            errors.append(f"summary:{exc}")

        result = GTMIntelligence(
            target=target,
            competitors=competitor_profiles,
            buying_signals=buying_signals,
            market_positioning=market_positioning,
            action_items=action_items,
            summary=summary,
        )
        logger.info(
            "[GTMAgent] Done | competitors=%d signals=%d errors=%d",
            len(competitor_profiles),
            len(buying_signals),
            len(errors),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    # Domains that are infrastructure/meta sites, not actual competitors
    _SKIP_DOMAINS = {
        "google.com", "google.co", "support.google.com", "accounts.google.com",
        "linkedin.com", "youtube.com", "facebook.com", "twitter.com", "reddit.com",
        "wikipedia.org", "amazon.com", "bing.com", "yahoo.com", "quora.com",
        "trustpilot.com", "glassdoor.com", "indeed.com", "forbes.com",
        "techcrunch.com", "crunchbase.com", "capterra.com",
    }

    def _discover_competitors(self, target: str) -> list[str]:
        """Use SERP to find top competitors of `target`."""
        query = f"{target} vs competitors alternatives CRM software 2024 2025"
        results = self._serp.google_search(query, num_results=10)

        competitors: list[str] = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")

            # Extract names from "X vs Y" titles
            if " vs " in title.lower():
                parts = re.split(r"\s+vs\.?\s+", title, flags=re.IGNORECASE)
                for p in parts:
                    name = re.sub(r"[^a-zA-Z0-9 ]", "", p.strip()).split()[0] if p.strip() else ""
                    if (name and len(name) > 2
                            and name.lower() != target.lower()
                            and name.lower() not in {"the", "best", "top", "free"}):
                        competitors.append(name)

            # Extract from URL domain — skip known non-competitor domains
            if url:
                parsed = urlparse(url)
                domain = parsed.netloc.replace("www.", "")
                base = domain.split(".")[0]  # e.g. "pipedrive" from "pipedrive.com"
                skip = any(skip in domain for skip in self._SKIP_DOMAINS)
                # Also skip country-code Google domains like google.cl, google.br
                if not skip and not re.match(r"^google\.", domain):
                    if base and len(base) > 3 and base.lower() != target.lower():
                        competitors.append(base.capitalize())

        # Deduplicate preserving order, cap at 4
        seen: set[str] = set()
        unique = []
        for c in competitors:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:4]

    def _profile_competitor(self, company: str, target: str) -> CompetitorProfile:
        """Scrape and analyze a single competitor."""
        # Build candidate URLs
        domain_guess = company.lower().replace(" ", "") + ".com"
        urls_to_try = [
            f"https://{domain_guess}/pricing",
            f"https://{domain_guess}",
        ]

        html_text = ""
        final_url = urls_to_try[0]
        for url in urls_to_try:
            # Try Web Unlocker first, fall back to Scraping Browser
            try:
                html_text = self._unlocker.fetch_with_retry(url)
                final_url = url
                break
            except Exception:
                pass
            try:
                html_text = run_coro(self._browser.fetch_js_page(url))
                final_url = url
                break
            except Exception:
                continue

        # AI analysis
        analysis: dict[str, Any] = {}
        if html_text:
            analysis = self._ai.analyze_competitor_page(company, final_url, html_text)

        # Job postings via Web Scraper API (Indeed)
        job_postings: list[str] = []
        try:
            rows = self._scraper_api.collect_and_wait(
                _INDEED_DATASET_ID,
                [{"keyword": company, "location": "United States", "country": "US"}],
                timeout_seconds=60,
            )
            job_postings = [r.get("job_title", "") for r in rows[:10] if r.get("job_title")]
        except Exception as exc:
            logger.debug("[GTMAgent] Job scrape skipped for %s: %s", company, exc)

        # SERP news
        news_results = []
        try:
            news_results = self._serp.google_search(
                f"{company} news product launch announcement 2024 2025", num_results=5
            )
        except Exception:
            pass
        recent_news = [r.get("title", "") for r in news_results if r.get("title")]

        return CompetitorProfile(
            company=company,
            domain=domain_guess,
            pricing_signals=analysis.get("pricing_signals", []),
            messaging_themes=analysis.get("messaging_themes", []),
            recent_job_postings=job_postings,
            recent_news=recent_news,
            hiring_signals=analysis.get("hiring_signals", []),
            summary=analysis.get("strategic_summary", ""),
        )

    def _detect_buying_signals(
        self, target: str, context: Optional[str]
    ) -> list[BuyingSignal]:
        """Search for buying intent signals around the target market."""
        focus = context or target
        queries = [
            f"{focus} alternative vendor switch 2024 2025",
            f'"{focus}" pain points "looking for" OR "need" solution',
            f"{focus} customer reviews complaints 2024",
        ]
        all_results: list[dict] = []
        for q in queries:
            try:
                all_results.extend(self._serp.google_search(q, num_results=5))
            except Exception:
                pass

        if not all_results:
            return []

        analysis = self._ai.analyze_buying_signals(target, all_results)
        signals_raw = analysis.get("signals", [])

        return [
            BuyingSignal(
                source=s.get("source_url", "web"),
                signal_type=s.get("type", "unknown"),
                description=s.get("description", ""),
                relevance_score=float(s.get("relevance_score", 0.5)),
                url=s.get("source_url"),
            )
            for s in signals_raw
        ]

    def _market_analysis(
        self,
        target: str,
        competitors: list[CompetitorProfile],
        signals: list[BuyingSignal],
    ) -> tuple[str, list[str]]:
        """Derive market positioning statement and top action items."""
        competitor_names = [c.company for c in competitors]
        signal_descriptions = [s.description for s in signals[:5]]

        prompt_data = {
            "target": target,
            "competitors": competitor_names,
            "top_signals": signal_descriptions,
        }
        result = self._ai._analyze_json(
            self._ai._GTM_SYSTEM,
            (
                f"Data: {prompt_data}\n\n"
                "Return JSON: {\"market_positioning\": str, \"action_items\": [str (max 5)]}"
            ),
            max_tokens=400,
        )
        return result.get("market_positioning", ""), result.get("action_items", [])
