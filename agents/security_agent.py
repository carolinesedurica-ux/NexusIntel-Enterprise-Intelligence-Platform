"""
Track 3 — Security & Compliance Agent
========================================
Covers all six Security & Compliance use cases:

  1. Threat intelligence pipelines — multi-source open web monitoring for
     org-specific risk indicators (SERP + CISA + NVD/CVE + GitHub exposure)

  2. Regulatory monitoring — all four compliance sources scraped with
     structured, actionable alerts and severity ratings

  3. Third-party risk — vendor auto-discovery + multi-signal risk assessment

  4. Brand & data exposure — paste sites, code leaks, cloud misconfigs,
     typosquatting, and credential dump detection

  5. Autonomous AI investigation — when HIGH/CRITICAL indicators are found,
     the agent autonomously fetches the source URL and performs a deep-dive
     structured analysis to surface additional related indicators

  6. Compliance alert delivery — structured action_required on every change;
     webhook POST for HIGH/CRITICAL findings

Bright Data tools used:
  - SERP API (threat hunting, compliance news, vendor discovery, brand monitoring)
  - Web Unlocker (regulatory pages, vendor security pages, threat source pages)
  - Scraping Browser (JS-heavy sites: CISA, NIST, GDPR portals)
  - Web Scraper API (structured news datasets)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

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

# All four compliance sources — every one is scraped on each run
_COMPLIANCE_SOURCES = [
    ("CISA Advisories",   "https://www.cisa.gov/news-events/cybersecurity-advisories"),
    ("NIST CSF",          "https://www.nist.gov/cyberframework"),
    ("GDPR Info",         "https://gdpr-info.eu/"),
    ("PCI DSS",           "https://www.pcisecuritystandards.org/document_library/"),
]


class SecurityAgent:
    """
    Produces Security & Compliance intelligence for a target organisation.

    run(target, vendors, context, webhook_url) → SecurityIntelligence
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
        webhook_url: Optional[str] = None,
    ) -> SecurityIntelligence:
        """
        Full security & compliance intelligence pipeline.

        Pillars addressed:
          1. Multi-source open-web threat monitoring
          2. Autonomous deep-dive on HIGH/CRITICAL indicators
          3. Regulatory change monitoring (4 sources) with structured alerts
          4. Auto-discovered vendor risk assessment
          5. Brand/credential exposure scanning
          6. Webhook alert delivery for critical findings

        Args:
            target:      Organisation name or domain to protect.
            vendors:     Known third-party vendors. Auto-discovered if omitted.
            context:     Optional compliance focus ("HIPAA", "PCI", "supply chain").
            webhook_url: URL to POST alert payload when severity is HIGH or CRITICAL.

        Returns:
            SecurityIntelligence with threats, compliance, vendor risks, and exposure.
        """
        logger.info("[SecurityAgent] Starting security intelligence for '%s'", target)
        errors: list[str] = []

        # ── Step 1: Multi-source threat monitoring ────────────────────────────
        threat_indicators: list[ThreatIndicator] = []
        try:
            threat_indicators = self._monitor_threats(target)
        except Exception as exc:
            logger.warning("[SecurityAgent] Threat monitoring failed: %s", exc)
            errors.append(f"threats:{exc}")

        # ── Step 2: Autonomous deep investigation (AI agentic loop) ──────────
        # For every HIGH/CRITICAL indicator that has a source URL, the agent
        # autonomously fetches that page and performs a structured deep-dive,
        # surfacing additional related indicators without human instruction.
        deep_indicators: list[ThreatIndicator] = []
        for ind in [
            t for t in threat_indicators
            if t.severity in (Severity.HIGH, Severity.CRITICAL) and t.url
        ][:3]:
            try:
                discovered = self._deep_investigate(target, ind)
                deep_indicators.extend(discovered)
                logger.info(
                    "[SecurityAgent] Deep investigation of '%s' found %d more indicators",
                    ind.indicator_type, len(discovered),
                )
            except Exception as exc:
                logger.debug("[SecurityAgent] Deep investigation failed: %s", exc)
        threat_indicators = (threat_indicators + deep_indicators)[:20]

        # ── Step 3: Compliance monitoring (all 4 sources) ────────────────────
        compliance_changes: list[ComplianceChange] = []
        try:
            compliance_changes = self._monitor_compliance(context)
        except Exception as exc:
            logger.warning("[SecurityAgent] Compliance monitoring failed: %s", exc)
            errors.append(f"compliance:{exc}")

        # ── Step 4: Vendor risk — auto-discover if none provided ─────────────
        if not vendors:
            try:
                vendors = self._discover_vendors(target)
                logger.info("[SecurityAgent] Auto-discovered %d vendors for '%s'", len(vendors), target)
            except Exception as exc:
                logger.debug("[SecurityAgent] Vendor auto-discovery failed: %s", exc)
                vendors = []

        vendor_risks: list[VendorRiskProfile] = []
        for vendor in (vendors or [])[:3]:
            try:
                profile = self._assess_vendor(vendor)
                vendor_risks.append(profile)
            except Exception as exc:
                logger.warning("[SecurityAgent] Vendor assessment failed for %s: %s", vendor, exc)
                errors.append(f"vendor:{vendor}:{exc}")

        # ── Step 5: Brand & credential exposure scan ─────────────────────────
        brand_exposure: list[str] = []
        try:
            brand_exposure = self._scan_brand_exposure(target)
        except Exception as exc:
            logger.warning("[SecurityAgent] Brand exposure scan failed: %s", exc)
            errors.append(f"brand_exposure:{exc}")

        # ── Step 6: Overall severity + CISO summary ──────────────────────────
        overall_severity = self._calculate_severity(
            threat_indicators, compliance_changes, brand_exposure
        )

        summary = ""
        try:
            summary = self._generate_summary(
                target, threat_indicators, compliance_changes, vendor_risks, brand_exposure
            )
        except Exception as exc:
            logger.warning("[SecurityAgent] Summary generation failed: %s", exc)
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

        # ── Step 7: Webhook alert delivery ───────────────────────────────────
        # Structured alert POSTed to webhook when findings are HIGH or CRITICAL
        if webhook_url and overall_severity in (Severity.HIGH, Severity.CRITICAL):
            try:
                self._deliver_alert(webhook_url, result)
            except Exception as exc:
                logger.warning("[SecurityAgent] Webhook delivery failed: %s", exc)

        logger.info(
            "[SecurityAgent] Done | threats=%d (deep+%d) compliance=%d vendors=%d "
            "brand=%d severity=%s errors=%d",
            len(threat_indicators), len(deep_indicators),
            len(compliance_changes), len(vendor_risks),
            len(brand_exposure), overall_severity.value, len(errors),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _monitor_threats(self, target: str) -> list[ThreatIndicator]:
        """
        Pillar 1 — Multi-source open-web threat intelligence pipeline.
        Sources: SERP (6 targeted query types) + CISA advisories + NVD/CVE search.
        """
        indicators: list[ThreatIndicator] = []
        domain = target.lower().replace(" ", "") + ".com"

        # Six query types covering the full threat surface
        threat_queries = [
            (f'"{target}" data breach leak 2024 2025',                    "credential_leak"),
            (f'"{domain}" vulnerability CVE exploit',                      "vulnerability"),
            (f'"{target}" phishing impersonation fraud brand abuse',       "brand_abuse"),
            (f'site:pastebin.com OR site:hastebin.com "{domain}"',         "data_dump"),
            (f'site:github.com "{target}" password OR secret OR api_key',  "code_exposure"),
            (f'"{target}" ransomware malware attack incident 2024 2025',   "malware"),
        ]
        for query, default_type in threat_queries:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results:
                    combined = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                    severity, indicator_type = self._classify_threat(combined)
                    if severity:
                        indicators.append(ThreatIndicator(
                            indicator_type=indicator_type or default_type,
                            value=r.get("url", domain),
                            source=r.get("url", "serp"),
                            severity=severity,
                            description=f"{r.get('title', '')} — {r.get('snippet', '')}",
                            url=r.get("url"),
                        ))
            except Exception as exc:
                logger.debug("[SecurityAgent] Threat query failed: %s", exc)

        # CISA active advisories
        try:
            cisa_url = _COMPLIANCE_SOURCES[0][1]
            try:
                html = asyncio.run(self._browser.fetch_js_page(cisa_url))
            except Exception:
                html = self._unlocker.fetch_with_retry(cisa_url)
            analysis = self._ai.analyze_threat_surface(target, html, "CISA Advisories")
            for ind in analysis.get("indicators", [])[:5]:
                indicators.append(ThreatIndicator(
                    indicator_type=ind.get("type", "advisory"),
                    value=ind.get("value", ""),
                    source="CISA",
                    severity=Severity(ind.get("severity", "low")),
                    description=ind.get("description", ""),
                ))
        except Exception as exc:
            logger.debug("[SecurityAgent] CISA threat scrape failed: %s", exc)

        # NVD / CVE database search via SERP
        try:
            nvd_results = self._serp.google_search(
                f'site:nvd.nist.gov OR site:cve.org "{target}" CVE 2024 2025',
                num_results=5,
            )
            for r in nvd_results:
                indicators.append(ThreatIndicator(
                    indicator_type="cve",
                    value=r.get("title", "")[:100],
                    source=r.get("url", "NVD"),
                    severity=Severity.HIGH,
                    description=r.get("snippet", ""),
                    url=r.get("url"),
                ))
        except Exception:
            pass

        return indicators[:15]

    def _deep_investigate(
        self, target: str, indicator: ThreatIndicator
    ) -> list[ThreatIndicator]:
        """
        Pillar 5 — Autonomous AI investigation.
        Fetches the source URL of a HIGH/CRITICAL indicator and runs a structured
        deep-dive analysis without any human instruction, returning additional
        related threat indicators discovered in the source content.
        """
        source_content = ""
        try:
            source_content = self._unlocker.fetch_with_retry(indicator.url)
        except Exception:
            try:
                source_content = asyncio.run(self._browser.fetch_js_page(indicator.url))
            except Exception:
                return []

        if not source_content:
            return []

        analysis = self._ai.investigate_threat_indicator(
            target,
            indicator.indicator_type,
            indicator.value,
            source_content,
        )

        return [
            ThreatIndicator(
                indicator_type=ci.get("type", "unknown"),
                value=ci.get("value", ""),
                source=indicator.url,
                severity=Severity(ci.get("severity", "medium")),
                description=ci.get("description", ""),
                url=indicator.url,
            )
            for ci in analysis.get("confirmed_indicators", [])[:5]
        ]

    def _discover_vendors(self, target: str) -> list[str]:
        """
        Pillar 3 — Auto-discover third-party vendors used by the target org.
        Queries SERP for tech stack and partner signals, then uses Claude to
        extract vendor names for subsequent risk assessment.
        """
        signals: list[str] = []
        for query in [
            f"{target} technology stack integrations uses powered by",
            f"{target} third-party vendors suppliers partners cloud",
        ]:
            try:
                results = self._serp.google_search(query, num_results=5)
                signals.extend(
                    r.get("title", "") + ": " + r.get("snippet", "")
                    for r in results
                )
            except Exception:
                pass

        if not signals:
            return []

        result = self._ai.discover_vendors(target, signals)
        return result.get("vendors", [])[:5]

    def _monitor_compliance(self, context: Optional[str]) -> list[ComplianceChange]:
        """
        Pillars 2 & 6 — Scrape all four compliance sources for structured updates.
        Every change includes an explicit action_required and severity rating.
        """
        changes: list[ComplianceChange] = []

        # SERP sweep for recent regulatory news
        focus = context or "data privacy security cybersecurity"
        for query in [
            f"{focus} regulatory change requirement deadline 2025",
            "GDPR CCPA HIPAA SOX PCI enforcement update 2025",
        ]:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results[:3]:
                    title_lower = r.get("title", "").lower()
                    if any(kw in title_lower for kw in
                           ["regulation", "law", "rule", "compliance",
                            "requirement", "act", "directive", "enforcement"]):
                        changes.append(ComplianceChange(
                            regulation=r.get("title", "")[:80],
                            jurisdiction="Global",
                            summary=r.get("snippet", ""),
                            severity=Severity.MEDIUM,
                            action_required=(
                                "Review this regulatory update and assess applicability "
                                "to your organisation's compliance posture."
                            ),
                            source_url=r.get("url"),
                        ))
            except Exception:
                pass

        # Scrape all four compliance sources
        for source_name, source_url in _COMPLIANCE_SOURCES:
            try:
                try:
                    html = asyncio.run(self._browser.fetch_js_page(source_url))
                except Exception:
                    html = self._unlocker.fetch_with_retry(source_url)

                analysis = self._ai.parse_compliance_update(source_name, html)
                for change in analysis.get("changes", [])[:3]:
                    action = change.get("action_required", "")
                    # Assign severity based on urgency keywords in the action
                    sev = (
                        Severity.HIGH
                        if any(k in action.lower() for k in
                               ["immediate", "critical", "urgent", "mandatory", "required by"])
                        else Severity.MEDIUM
                    )
                    changes.append(ComplianceChange(
                        regulation=change.get("regulation", source_name),
                        jurisdiction=change.get("jurisdiction", "US"),
                        summary=change.get("summary", ""),
                        effective_date=change.get("effective_date"),
                        severity=sev,
                        action_required=(
                            action
                            if action
                            else f"Review {source_name} update and assess impact on compliance posture."
                        ),
                        source_url=source_url,
                    ))
            except Exception as exc:
                logger.debug("[SecurityAgent] Compliance source '%s' failed: %s", source_name, exc)

        return changes[:12]

    def _assess_vendor(self, vendor: str) -> VendorRiskProfile:
        """
        Pillar 3 — Multi-signal third-party vendor risk assessment.
        Collects SERP signals + vendor security page content, then uses Claude
        to produce a scored risk profile with specific risk factors and recommendation.
        """
        domain = vendor.lower().replace(" ", "") + ".com"
        signals: list[str] = []

        for query in [
            f"{vendor} security incident data breach 2024 2025",
            f"{vendor} financial trouble layoff bankruptcy restructure",
            f"{vendor} regulatory fine violation GDPR SOC2",
        ]:
            try:
                results = self._serp.google_search(query, num_results=3)
                signals.extend(
                    r.get("title", "") + ": " + r.get("snippet", "")
                    for r in results
                )
            except Exception:
                pass

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
            recommendation=assessment.get("recommendation", ""),
        )

    def _scan_brand_exposure(self, target: str) -> list[str]:
        """
        Pillar 4 — Brand and data exposure monitoring.
        Six query vectors: paste sites, credential dumps, GitHub code leaks,
        brand abuse, cloud misconfigs, and raw data file exposure.
        """
        raw_findings: list[str] = []
        domain = target.lower().replace(" ", "") + ".com"

        exposure_queries = [
            f'"{domain}" site:pastebin.com OR site:hastebin.com OR site:ghostbin.com',
            f'"{target}" credentials dump password leak 2024 2025',
            f'site:github.com "{target}" password OR secret OR api_key OR token',
            f'"{target}" brand abuse typosquatting fake phishing domain',
            f'"{domain}" exposed S3 bucket OR git config OR .env file',
            f'inurl:"{domain}" filetype:sql OR filetype:log OR filetype:csv',
        ]
        for query in exposure_queries:
            try:
                results = self._serp.google_search(query, num_results=5)
                for r in results:
                    combined = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                    if any(kw in combined for kw in [
                        "exposed", "leaked", "dump", "credentials", "abuse",
                        "fake", "phish", "password", "secret", "token", "key",
                        "bucket", "misconfigured", "typosquat",
                    ]):
                        raw_findings.append(
                            f"{r.get('title', '')} ({r.get('url', '')})"
                        )
            except Exception:
                pass

        return list(dict.fromkeys(raw_findings))[:10]

    def _deliver_alert(
        self, webhook_url: str, result: SecurityIntelligence
    ) -> None:
        """
        Pillar 6 — Structured alert delivery to webhook endpoint.
        Triggered automatically when overall_severity is HIGH or CRITICAL.
        """
        payload = {
            "alert_type": "nexusintel_security_alert",
            "target": result.target,
            "overall_severity": result.overall_severity.value,
            "threat_count": len(result.threat_indicators),
            "critical_threats": [
                {
                    "type": t.indicator_type,
                    "severity": t.severity.value,
                    "description": t.description[:200],
                    "url": t.url,
                }
                for t in result.threat_indicators
                if t.severity in (Severity.HIGH, Severity.CRITICAL)
            ][:5],
            "compliance_changes": [
                {
                    "regulation": c.regulation,
                    "severity": c.severity.value,
                    "action_required": c.action_required,
                }
                for c in result.compliance_changes
                if c.severity in (Severity.HIGH, Severity.CRITICAL)
            ][:5],
            "brand_exposures": len(result.brand_exposure),
            "summary": result.summary[:500],
        }
        httpx.post(webhook_url, json=payload, timeout=10)
        logger.info("[SecurityAgent] Alert delivered to %s (severity=%s)",
                    webhook_url, result.overall_severity.value)

    def _classify_threat(self, text: str) -> tuple[Optional[Severity], str]:
        """Keyword-based rapid threat classification for SERP snippets."""
        critical_kw = [
            "breach", "leak", "exposed", "compromised", "stolen",
            "ransomware", "dump", "credential",
        ]
        high_kw = [
            "vulnerability", "cve", "exploit", "phishing", "fraud",
            "impersonat", "malware", "attack",
        ]
        medium_kw = [
            "misconfiguration", "weak", "outdated", "unsecured",
            "secret", "exposed key", "api key",
        ]
        if any(k in text for k in critical_kw):
            return Severity.CRITICAL, "credential_leak"
        if any(k in text for k in high_kw):
            return Severity.HIGH, "vulnerability"
        if any(k in text for k in medium_kw):
            return Severity.MEDIUM, "misconfiguration"
        return None, ""

    def _calculate_severity(
        self,
        threats: list[ThreatIndicator],
        compliance: list[ComplianceChange],
        exposure: list[str],
    ) -> Severity:
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
        data = {
            "threats": len(threats),
            "critical": sum(1 for t in threats if t.severity == Severity.CRITICAL),
            "high": sum(1 for t in threats if t.severity == Severity.HIGH),
            "compliance_changes": len(compliance),
            "high_risk_vendors": sum(1 for v in vendors if v.risk_score >= 7.0),
            "brand_exposures": len(exposure),
            "top_threats": [t.description[:120] for t in threats[:3]],
            "top_compliance": [
                f"{c.regulation}: {c.action_required[:80]}"
                for c in compliance[:3]
            ],
            "vendor_risks": [f"{v.vendor}: {v.risk_score}/10" for v in vendors],
        }
        return self._ai._analyze(
            self._ai._SECURITY_SYSTEM,
            (
                f"Security intelligence for {target}:\n{data}\n\n"
                "Write a concise (150-word) security briefing for a CISO. "
                "Cover: top threats with severity, compliance priorities with specific "
                "actions required, vendor risk concerns, and 3 immediate actions. "
                "Be specific — name actual threats, regulations, and vendors."
            ),
            max_tokens=350,
        )
