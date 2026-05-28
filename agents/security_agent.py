"""
Track 3 — Security & Compliance Agent
========================================
Responsibilities:
  1. Monitor open web for threat indicators targeting the org
  2. Scrape regulatory sites for compliance changes
  3. Assess vendor/third-party risk via web signals
  4. Detect brand exposure (credentials, impersonation, data leaks)
  5. Return a SecurityIntelligence object

Bright Data tools used:
  - Web Unlocker (threat intel sites, regulatory pages, dark web mirrors)
  - SERP API (brand monitoring, threat searches, compliance news)
  - Web Scraper API (structured news & job datasets)
  - Scraping Browser (JS-heavy sites, paste sites, leak databases)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from models.schemas import (
    ComplianceChange,
    SecurityIntelligence,
    Severity,
    ThreatIndicator,
    VendorRiskProfile,
)
from tools.brightdata_tools import SERPApi, ScrapingBrowser, WebScraperApi, WebUnlocker
from tools.ai_tools import AIAnalyzer

logger = logging.getLogger(__name__)

# Compliance monitoring sources
_COMPLIANCE_SOURCES = [
    ("NIST CSF", "https://www.nist.gov/cyberframework"),
    ("CISA Alerts", "https://www.cisa.gov/news-events/cybersecurity-advisories"),
    ("GDPR Updates", "https://gdpr-info.eu/"),
    ("PCI DSS", "https://www.pcisecuritystandards.org/"),
]

# Threat intelligence / paste-monitoring sources
_THREAT_SOURCES = [
    "https://haveibeenpwned.com",
    "https://pastebin.com",
]


class SecurityAgent:
    """
    Produces Security & Compliance intelligence for a target organization.

    run(target, vendors, context) → SecurityIntelligence
    """

    def __init__(self) -> None:
        self._serp = SERPApi()
        self._unlocker = WebUnlocker()
        self._scraper_api = WebScraperApi()
        self._browser = ScrapingBrowser()
        self._ai = AIAnalyzer()

    def run(
        self,
        target: str,
        vendors: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> SecurityIntelligence:
        """
        Full security & compliance intelligence pipeline.

        Args:
            target:   Organization name or domain to protect.
            vendors:  Known third-party vendors to assess.
            context:  Optional focus ("HIPAA", "PCI", "supply chain").

        Returns:
            SecurityIntelligence with threats, compliance changes, and vendor risks.
        """
        logger.info("[SecurityAgent] Starting security intelligence for '%s'", target)
        errors: list[str] = []

        # ── Step 1: Threat surface monitoring ────────────────────────────────
        threat_indicators: list[ThreatIndicator] = []
        try:
            threat_indicators = self._monitor_threats(target)
        except Exception as exc:
            logger.warning("[SecurityAgent] Threat monitoring failed: %s", exc)
            errors.append(f"threats:{exc}")

        # ── Step 2: Compliance changes ────────────────────────────────────────
        compliance_changes: list[ComplianceChange] = []
        try:
            compliance_changes = self._monitor_compliance(context)
        except Exception as exc:
            logger.warning("[SecurityAgent] Compliance monitoring failed: %s", exc)
            errors.append(f"compliance:{exc}")

        # ── Step 3: Vendor risk assessment ───────────────────────────────────
        vendor_risks: list[VendorRiskProfile] = []
        for vendor in (vendors or [])[:3]:
            try:
                profile = self._assess_vendor(vendor)
                vendor_risks.append(profile)
            except Exception as exc:
                logger.warning("[SecurityAgent] Vendor assessment failed for %s: %s", vendor, exc)
                errors.append(f"vendor:{vendor}:{exc}")

        # ── Step 4: Brand exposure scan ──────────────────────────────────────
        brand_exposure: list[str] = []
        try:
            brand_exposure = self._scan_brand_exposure(target)
        except Exception as exc:
            logger.warning("[SecurityAgent] Brand exposure scan failed: %s", exc)
            errors.append(f"brand_exposure:{exc}")

        # ── Step 5: Overall severity ─────────────────────────────────────────
        overall_severity = self._calculate_severity(threat_indicators, compliance_changes, brand_exposure)

        # ── Step 6: Summary ───────────────────────────────────────────────────
        summary = ""
        try:
            summary = self._generate_summary(
                target, threat_indicators, compliance_changes, vendor_risks, brand_exposure
            )
        except Exception as exc:
            errors.append(f"summary:{exc}")

        result = SecurityIntelligence(
            target=target,
            threat_indicators=threat_indicators,
            compliance_changes=compliance_changes,
            vendor_risks=vendor_risks,
            brand_exposure=brand_exposure,
            overall_severity=overall_severity,
            summary=summary,
        )
        logger.info(
            "[SecurityAgent] Done | threats=%d compliance=%d vendors=%d brand_exposure=%d errors=%d",
            len(threat_indicators),
            len(compliance_changes),
            len(vendor_risks),
            len(brand_exposure),
            len(errors),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _monitor_threats(self, target: str) -> list[ThreatIndicator]:
        """Search for threat indicators targeting the organization."""
        indicators: list[ThreatIndicator] = []
        domain = target.lower().replace(" ", "") + ".com"

        # SERP-based threat hunting
        queries = [
            f'"{target}" data breach leak 2024 2025',
            f'"{domain}" exposed vulnerability CVE',
            f'"{target}" phishing impersonation fraud',
            f'site:pastebin.com "{domain}"',
        ]
        for query in queries:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results:
                    title = r.get("title", "").lower()
                    snippet = r.get("snippet", "").lower()
                    combined = title + " " + snippet

                    severity, indicator_type = self._classify_threat(combined)
                    if severity:
                        indicators.append(
                            ThreatIndicator(
                                indicator_type=indicator_type,
                                value=r.get("url", domain),
                                source=r.get("url", "serp"),
                                severity=severity,
                                description=f"{r.get('title', '')} — {r.get('snippet', '')}",
                                url=r.get("url"),
                            )
                        )
            except Exception as exc:
                logger.debug("[SecurityAgent] Threat query failed: %s", exc)

        # Scrape CISA advisories for active threats
        try:
            html = self._unlocker.fetch_with_retry(_COMPLIANCE_SOURCES[1][1])
            analysis = self._ai.analyze_threat_surface(target, html, "CISA Advisories")
            for ind in analysis.get("indicators", [])[:5]:
                indicators.append(
                    ThreatIndicator(
                        indicator_type=ind.get("type", "advisory"),
                        value=ind.get("value", ""),
                        source="CISA",
                        severity=Severity(ind.get("severity", "low")),
                        description=ind.get("description", ""),
                    )
                )
        except Exception as exc:
            logger.debug("[SecurityAgent] CISA scrape skipped: %s", exc)

        return indicators[:15]

    def _classify_threat(self, text: str) -> tuple[Optional[Severity], str]:
        """Quick keyword-based threat classification."""
        critical_keywords = ["breach", "leak", "exposed", "compromised", "stolen", "ransomware"]
        high_keywords = ["vulnerability", "cve", "exploit", "phishing", "fraud", "impersonat"]
        medium_keywords = ["misconfiguration", "weak", "outdated", "unsecured"]

        if any(k in text for k in critical_keywords):
            return Severity.CRITICAL, "credential_leak"
        if any(k in text for k in high_keywords):
            return Severity.HIGH, "vulnerability"
        if any(k in text for k in medium_keywords):
            return Severity.MEDIUM, "misconfiguration"
        return None, ""

    def _monitor_compliance(self, context: Optional[str]) -> list[ComplianceChange]:
        """Scrape regulatory sources for compliance updates."""
        changes: list[ComplianceChange] = []

        # SERP for recent regulatory changes
        focus = context or "data privacy security"
        queries = [
            f"{focus} regulatory change requirement 2025",
            "GDPR CCPA HIPAA enforcement update 2025",
            "cybersecurity regulation compliance deadline 2025",
        ]
        for query in queries[:2]:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results[:3]:
                    if any(kw in r.get("title", "").lower() for kw in ["regulation", "law", "rule", "compliance", "requirement", "act"]):
                        changes.append(
                            ComplianceChange(
                                regulation=r.get("title", "")[:80],
                                jurisdiction="Global",
                                summary=r.get("snippet", ""),
                                source_url=r.get("url"),
                            )
                        )
            except Exception:
                pass

        # Direct scrape of CISA advisories via Scraping Browser (JS-rendered page)
        import asyncio
        try:
            cisa_name, cisa_url = _COMPLIANCE_SOURCES[1]
            try:
                html = asyncio.run(self._browser.fetch_js_page(cisa_url))
            except Exception:
                html = self._unlocker.fetch_with_retry(cisa_url)
            analysis = self._ai.parse_compliance_update(cisa_name, html)
            for change in analysis.get("changes", [])[:5]:
                changes.append(
                    ComplianceChange(
                        regulation=change.get("regulation", "CISA Advisory"),
                        jurisdiction=change.get("jurisdiction", "US"),
                        summary=change.get("summary", ""),
                        effective_date=change.get("effective_date"),
                        action_required=change.get("action_required", ""),
                        source_url=cisa_url,
                    )
                )
        except Exception as exc:
            logger.debug("[SecurityAgent] CISA compliance scrape skipped: %s", exc)

        return changes[:10]

    def _assess_vendor(self, vendor: str) -> VendorRiskProfile:
        """Build a risk profile for a third-party vendor."""
        domain = vendor.lower().replace(" ", "") + ".com"
        signals: list[str] = []

        # SERP signals
        queries = [
            f"{vendor} security incident breach 2024 2025",
            f"{vendor} financial trouble layoff bankruptcy",
            f"{vendor} regulatory fine violation",
        ]
        for q in queries:
            try:
                results = self._serp.google_search(q, num_results=3)
                signals.extend(r.get("title", "") + ": " + r.get("snippet", "") for r in results)
            except Exception:
                pass

        # Web Unlocker: vendor's own security page
        try:
            html = self._unlocker.fetch_with_retry(f"https://{domain}/security")
            signals.append("Security page content: " + html[:500])
        except Exception:
            pass

        assessment = self._ai.assess_vendor_risk(vendor, signals)
        return VendorRiskProfile(
            vendor=vendor,
            domain=domain,
            risk_score=float(assessment.get("risk_score", 0.0)),
            risk_factors=assessment.get("risk_factors", []),
            open_issues=assessment.get("open_issues", []),
        )

    def _scan_brand_exposure(self, target: str) -> list[str]:
        """Search for brand impersonation, leaked credentials, or data exposures."""
        exposures: list[str] = []
        domain = target.lower().replace(" ", "") + ".com"

        exposure_queries = [
            f'"{domain}" site:pastebin.com OR site:hastebin.com',
            f'"{target}" credentials dump password 2024 2025',
            f'"{target}" brand abuse typosquatting fake',
            f'"{domain}" exposed S3 bucket OR git repository',
        ]
        for q in exposure_queries:
            try:
                results = self._serp.google_search(q, num_results=5)
                for r in results:
                    if any(
                        kw in (r.get("title", "") + r.get("snippet", "")).lower()
                        for kw in ["exposed", "leaked", "dump", "credentials", "abuse", "fake", "phish"]
                    ):
                        exposures.append(f"{r.get('title', '')} ({r.get('url', '')})")
            except Exception:
                pass

        return list(dict.fromkeys(exposures))[:10]

    def _calculate_severity(
        self,
        threats: list[ThreatIndicator],
        compliance: list[ComplianceChange],
        exposure: list[str],
    ) -> Severity:
        """Determine overall severity from all findings."""
        if any(t.severity == Severity.CRITICAL for t in threats):
            return Severity.CRITICAL
        if any(t.severity == Severity.HIGH for t in threats) or len(exposure) >= 3:
            return Severity.HIGH
        if len(threats) >= 3 or len(compliance) >= 5:
            return Severity.MEDIUM
        return Severity.LOW

    def _generate_summary(
        self,
        target: str,
        threats: list[ThreatIndicator],
        compliance: list[ComplianceChange],
        vendors: list[VendorRiskProfile],
        exposure: list[str],
    ) -> str:
        """Generate a security summary via Claude."""
        data = {
            "threats": len(threats),
            "critical_threats": sum(1 for t in threats if t.severity == Severity.CRITICAL),
            "compliance_changes": len(compliance),
            "high_risk_vendors": sum(1 for v in vendors if v.risk_score >= 7.0),
            "brand_exposures": len(exposure),
            "top_threats": [t.description[:100] for t in threats[:3]],
            "top_compliance": [c.summary[:100] for c in compliance[:3]],
        }
        return self._ai._analyze(
            self._ai._SECURITY_SYSTEM,
            (
                f"Security intelligence for {target}:\n{data}\n\n"
                "Write a concise (150-word) security briefing for a CISO. "
                "Include: top threats, compliance priorities, vendor concerns, and immediate actions."
            ),
            max_tokens=300,
        )
