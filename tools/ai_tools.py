"""
NexusIntel — Claude AI analysis helpers
=========================================
Stateless helpers that call Claude with cached system prompts to analyze
raw web data and return structured intelligence across all three tracks.

Prompt caching is used on the large system prompts to cut latency and cost.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import anthropic

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AIAnalyzer:
    """
    Thin wrapper around the Anthropic SDK with track-specific analysis methods.
    Each method accepts raw text/dicts and returns structured analysis dicts.
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    # ── Internal helper ──────────────────────────────────────────────────────

    def _analyze(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        """Call Claude and return the text response."""
        system_block: Any = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if use_cache
            else system
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_block,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()

    def _analyze_json(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Call Claude expecting a JSON response and parse it."""
        system_with_json = (
            system
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. No prose, no markdown fences."
        )
        raw = self._analyze(system_with_json, user_content, max_tokens)
        try:
            # Strip any accidental markdown fences
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("Claude JSON parse failed; returning raw text in dict")
            return {"raw": raw}

    # ── Track 1 — GTM ────────────────────────────────────────────────────────

    _GTM_SYSTEM = (
        "You are an elite GTM intelligence analyst. "
        "Your job is to extract competitive signals, pricing strategies, messaging themes, "
        "and hiring patterns from raw web data scraped from competitor websites, "
        "job boards, and news sources. "
        "Always be specific: quote exact prices, exact job titles, exact phrases. "
        "Identify implied strategic moves based on hiring velocity and product messaging. "
        "Output must be actionable for a VP of Sales or CMO."
    )

    def analyze_competitor_page(self, company: str, url: str, html_text: str) -> dict[str, Any]:
        """Extract competitor signals from a scraped page."""
        prompt = (
            f"Company: {company}\nSource URL: {url}\n\n"
            f"Raw web content (first 4000 chars):\n{html_text[:4000]}\n\n"
            "Extract:\n"
            "1. pricing_signals: list of pricing mentions (plans, tiers, prices)\n"
            "2. messaging_themes: list of key value propositions / slogans\n"
            "3. hiring_signals: list of any role types being hired\n"
            "4. strategic_summary: 2-sentence strategic inference\n\n"
            "Return JSON with keys: pricing_signals, messaging_themes, hiring_signals, strategic_summary"
        )
        return self._analyze_json(self._GTM_SYSTEM, prompt, max_tokens=800)

    def analyze_buying_signals(self, target: str, search_results: list[dict]) -> dict[str, Any]:
        """Score a list of SERP results for buying intent signals."""
        results_text = "\n".join(
            f"{i+1}. [{r.get('title','')}] {r.get('url','')} — {r.get('snippet','')}"
            for i, r in enumerate(search_results[:15])
        )
        prompt = (
            f"Target company/market: {target}\n\n"
            f"Recent web search results:\n{results_text}\n\n"
            "Identify buying signals — evidence that {target}'s customers or prospects "
            "are actively seeking solutions, switching vendors, or experiencing pain points. "
            "Return JSON: {\"signals\": [{\"type\": str, \"description\": str, "
            "\"relevance_score\": float 0-1, \"source_url\": str}]}"
        )
        return self._analyze_json(self._GTM_SYSTEM, prompt, max_tokens=600)

    def generate_gtm_summary(self, target: str, raw_data: dict[str, Any]) -> str:
        """Generate an executive GTM intelligence brief."""
        prompt = (
            f"Target: {target}\n\n"
            f"Collected intelligence:\n{json.dumps(raw_data, indent=2)[:3000]}\n\n"
            "Write a concise (200-word) GTM intelligence brief for a VP of Sales. "
            "Include: competitive positioning gaps, key buying signals, "
            "recommended ICP focus areas, and top 3 action items."
        )
        return self._analyze(self._GTM_SYSTEM, prompt, max_tokens=400)

    # ── Track 2 — Finance ─────────────────────────────────────────────────────

    _FINANCE_SYSTEM = (
        "You are a senior financial intelligence analyst specializing in alternative data. "
        "Extract pricing intelligence, regulatory risk, and alternative data signals from "
        "raw web content. Be quantitative where possible. Flag material changes. "
        "Your output informs portfolio managers, procurement teams, and CFOs."
    )

    def analyze_pricing_page(self, company: str, html_text: str) -> dict[str, Any]:
        """Extract structured pricing data from a pricing/product page."""
        prompt = (
            f"Company: {company}\n\nPage content:\n{html_text[:3500]}\n\n"
            "Extract all pricing information. Return JSON:\n"
            "{\"products\": [{\"name\": str, \"price\": float|null, \"currency\": str, "
            "\"billing_period\": str, \"tier\": str, \"metadata\": {}}]}"
        )
        return self._analyze_json(self._FINANCE_SYSTEM, prompt, max_tokens=600)

    def analyze_regulatory_content(self, source: str, html_text: str) -> dict[str, Any]:
        """Identify compliance/regulatory alerts from a regulatory source page."""
        prompt = (
            f"Source: {source}\n\nContent:\n{html_text[:3500]}\n\n"
            "Identify any regulatory changes, enforcement actions, or compliance requirements. "
            "Return JSON:\n"
            "{\"alerts\": [{\"regulation\": str, \"jurisdiction\": str, "
            "\"summary\": str, \"effective_date\": str|null, \"action_required\": str}]}"
        )
        return self._analyze_json(self._FINANCE_SYSTEM, prompt, max_tokens=600)

    def interpret_alternative_signals(
        self, company: str, job_postings: list[str], news_items: list[str]
    ) -> dict[str, Any]:
        """Interpret job postings and news as alternative financial signals."""
        prompt = (
            f"Company: {company}\n\n"
            f"Recent job postings ({len(job_postings)}):\n" + "\n".join(f"- {j}" for j in job_postings[:20]) + "\n\n"
            f"Recent news ({len(news_items)}):\n" + "\n".join(f"- {n}" for n in news_items[:10]) + "\n\n"
            "Interpret these as alternative financial signals. What do they imply about "
            "headcount growth, product investment, financial health, and risk factors? "
            "Return JSON: {\"signals\": [{\"type\": str, \"value\": str, \"interpretation\": str}], "
            "\"risk_score\": float 0-10, \"investment_summary\": str}"
        )
        return self._analyze_json(self._FINANCE_SYSTEM, prompt, max_tokens=700)

    # ── Track 3 — Security & Compliance ──────────────────────────────────────

    _SECURITY_SYSTEM = (
        "You are a senior threat intelligence and compliance analyst. "
        "Assess web content for cyber threats, data exposures, regulatory non-compliance, "
        "and third-party risk indicators. Be precise about severity. "
        "Your findings are acted on by security operations and compliance teams. "
        "Never speculate — only flag what the evidence supports."
    )

    def analyze_threat_surface(self, target: str, html_text: str, source: str) -> dict[str, Any]:
        """Scan web content for threat indicators related to the target org."""
        prompt = (
            f"Target organization: {target}\nSource: {source}\n\n"
            f"Content:\n{html_text[:3500]}\n\n"
            "Identify threat indicators: leaked credentials, exposed infrastructure, "
            "vulnerability disclosures, dark web mentions, or brand abuse. "
            "Return JSON:\n"
            "{\"indicators\": [{\"type\": str, \"value\": str, \"severity\": \"low|medium|high|critical\", "
            "\"description\": str}]}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=600)

    def assess_vendor_risk(self, vendor: str, web_signals: list[str]) -> dict[str, Any]:
        """Generate a vendor risk profile from web-gathered signals."""
        signals_text = "\n".join(f"- {s}" for s in web_signals[:20])
        prompt = (
            f"Vendor: {vendor}\n\n"
            f"Web intelligence signals:\n{signals_text}\n\n"
            "Assess the vendor's risk profile. Look for: financial instability signals, "
            "security incidents, leadership changes, compliance violations, "
            "negative news, or operational red flags. "
            "Return JSON:\n"
            "{\"risk_score\": float 0-10, \"risk_factors\": [str], "
            "\"open_issues\": [str], \"recommendation\": str}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=500)

    def parse_compliance_update(self, regulation: str, html_text: str) -> dict[str, Any]:
        """Extract structured compliance change data from a regulatory page."""
        prompt = (
            f"Regulation/Source: {regulation}\n\nContent:\n{html_text[:3500]}\n\n"
            "Extract compliance changes. For each change, provide a specific, actionable "
            "'action_required' (what a compliance team must DO). Return JSON:\n"
            "{\"changes\": [{\"regulation\": str, \"jurisdiction\": str, "
            "\"summary\": str, \"effective_date\": str|null, \"action_required\": str}]}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=600)

    def investigate_threat_indicator(
        self,
        target: str,
        indicator_type: str,
        indicator_value: str,
        source_content: str,
    ) -> dict[str, Any]:
        """
        Autonomous deep-dive investigation of a detected threat indicator.
        Called when the agent finds a HIGH/CRITICAL signal and fetches the source.
        """
        prompt = (
            f"Target organization: {target}\n"
            f"Initial threat — type: {indicator_type}, value: {indicator_value}\n\n"
            f"Source content (first 3000 chars):\n{source_content[:3000]}\n\n"
            "Perform a deep investigation of this threat. Extract ALL related indicators "
            "confirmed by the evidence, assess actual vs potential exposure, identify affected "
            "systems or data categories, and specify immediate mitigation steps.\n"
            "Return JSON:\n"
            "{\"confirmed_indicators\": [{\"type\": str, \"value\": str, "
            "\"severity\": \"low|medium|high|critical\", \"description\": str}], "
            "\"exposure_summary\": str, \"immediate_actions\": [str]}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=700)

    def discover_vendors(self, target: str, signals: list[str]) -> dict[str, Any]:
        """
        Extract third-party vendor and supplier names from web signals about the target.
        Used for auto-populating vendor risk assessments.
        """
        signals_text = "\n".join(f"- {s}" for s in signals[:20])
        prompt = (
            f"Organization: {target}\n\n"
            f"Web intelligence signals about their technology and partners:\n{signals_text}\n\n"
            "Extract all third-party vendors, suppliers, and technology providers. "
            "Focus on: cloud providers, SaaS tools, payment processors, data vendors, "
            "infrastructure providers, and key contractors. Exclude the target itself.\n"
            "Return JSON: {\"vendors\": [str], \"critical_vendors\": [str]}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=300)

    def analyze_brand_exposure_structured(
        self, target: str, raw_findings: list[str]
    ) -> dict[str, Any]:
        """Classify raw brand/data exposure findings into a structured risk assessment."""
        findings_text = "\n".join(f"- {f}" for f in raw_findings[:20])
        prompt = (
            f"Target: {target}\n\n"
            f"Raw exposure findings from web monitoring:\n{findings_text}\n\n"
            "Classify each finding. Identify: credential leaks, brand impersonation, "
            "typosquatting domains, GitHub/code leaks, cloud misconfiguration, or data dumps. "
            "Return JSON:\n"
            "{\"exposures\": [{\"type\": str, \"description\": str, "
            "\"severity\": \"low|medium|high|critical\", \"source\": str}], "
            "\"total_risk\": \"low|medium|high|critical\"}"
        )
        return self._analyze_json(self._SECURITY_SYSTEM, prompt, max_tokens=500)

    # ── Cross-track synthesis ─────────────────────────────────────────────────

    _SYNTHESIS_SYSTEM = (
        "You are the Chief Intelligence Officer at an enterprise analytics firm. "
        "You synthesize GTM, financial, and security intelligence into a unified "
        "executive brief. Your output is read by C-suite executives who need to "
        "act quickly. Be concise, prioritize ruthlessly, and always connect dots "
        "across the three intelligence domains."
    )

    def synthesize_report(
        self,
        target: str,
        gtm_summary: str,
        finance_summary: str,
        security_summary: str,
    ) -> dict[str, Any]:
        """Generate the final cross-track executive intelligence brief."""
        # Truncate individual summaries to avoid overrunning token budget
        prompt = (
            f"Target: {target}\n\n"
            f"GTM:\n{gtm_summary[:800]}\n\n"
            f"Finance:\n{finance_summary[:600]}\n\n"
            f"Security:\n{security_summary[:600]}\n\n"
            "Return ONLY this JSON (no prose, no fences):\n"
            "{\n"
            '  "executive_summary": "2-3 sentence brief for C-suite",\n'
            '  "top_priorities": ["priority 1", "priority 2", "priority 3"],\n'
            '  "cross_track_connections": ["connection 1"],\n'
            '  "recommended_actions": ["action 1", "action 2"]\n'
            "}"
        )
        result = self._analyze_json(self._SYNTHESIS_SYSTEM, prompt, max_tokens=600)
        # Ensure required keys always exist
        result.setdefault("executive_summary", result.get("raw", "Intelligence synthesis complete."))
        result.setdefault("top_priorities", [])
        result.setdefault("cross_track_connections", [])
        result.setdefault("recommended_actions", [])
        return result
