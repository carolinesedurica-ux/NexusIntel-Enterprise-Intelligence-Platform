"""
NexusIntel — Pydantic schemas for all three intelligence tracks.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Track(str, Enum):
    GTM = "gtm"
    FINANCE = "finance"
    SECURITY = "security"


# ─── Shared ──────────────────────────────────────────────────────────────────

class IntelligenceRequest(BaseModel):
    """Inbound request payload accepted by all orchestration endpoints."""
    target: str = Field(..., description="Company name, domain, or search term")
    tracks: list[Track] = Field(
        default=[Track.GTM, Track.FINANCE, Track.SECURITY],
        description="Which intelligence tracks to run",
    )
    context: Optional[str] = Field(None, description="Optional context or focus area")
    webhook_url: Optional[str] = Field(None, description="URL to POST results to on completion")


class ScrapedPage(BaseModel):
    url: str
    status_code: int
    content: str
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    position: int
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Track 1 — GTM Intelligence ──────────────────────────────────────────────

class CompetitorProfile(BaseModel):
    company: str
    domain: str
    pricing_signals: list[str] = Field(default_factory=list)
    messaging_themes: list[str] = Field(default_factory=list)
    recent_job_postings: list[str] = Field(default_factory=list)
    recent_news: list[str] = Field(default_factory=list)
    hiring_signals: list[str] = Field(default_factory=list)
    summary: str = ""
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class BuyingSignal(BaseModel):
    source: str
    signal_type: str  # "job_posting" | "news" | "social" | "search_trend"
    description: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    url: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class GTMIntelligence(BaseModel):
    target: str
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    buying_signals: list[BuyingSignal] = Field(default_factory=list)
    market_positioning: str = ""
    action_items: list[str] = Field(default_factory=list)
    summary: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Track 2 — Finance & Market Intelligence ─────────────────────────────────

class PriceDataPoint(BaseModel):
    source: str
    product: str
    price: Optional[float] = None
    currency: str = "USD"
    metadata: dict[str, Any] = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class RegulatoryAlert(BaseModel):
    source: str
    title: str
    summary: str
    url: Optional[str] = None
    impact: str = ""
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class AlternativeDataSignal(BaseModel):
    signal_type: str  # "job_posting_velocity" | "web_traffic" | "review_sentiment"
    company: str
    value: Any
    interpretation: str
    collected_at: datetime = Field(default_factory=datetime.utcnow)


class FinanceIntelligence(BaseModel):
    target: str
    pricing_data: list[PriceDataPoint] = Field(default_factory=list)
    regulatory_alerts: list[RegulatoryAlert] = Field(default_factory=list)
    alternative_signals: list[AlternativeDataSignal] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0.0, le=10.0)
    investment_summary: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Track 3 — Security & Compliance ─────────────────────────────────────────

class ThreatIndicator(BaseModel):
    indicator_type: str  # "domain" | "ip" | "hash" | "keyword" | "credential_leak"
    value: str
    source: str
    severity: Severity
    description: str
    url: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ComplianceChange(BaseModel):
    regulation: str
    jurisdiction: str
    summary: str
    effective_date: Optional[str] = None
    action_required: str = ""
    source_url: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class VendorRiskProfile(BaseModel):
    vendor: str
    domain: str
    risk_score: float = Field(ge=0.0, le=10.0)
    risk_factors: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class SecurityIntelligence(BaseModel):
    target: str
    threat_indicators: list[ThreatIndicator] = Field(default_factory=list)
    compliance_changes: list[ComplianceChange] = Field(default_factory=list)
    vendor_risks: list[VendorRiskProfile] = Field(default_factory=list)
    brand_exposure: list[str] = Field(default_factory=list)
    overall_severity: Severity = Severity.LOW
    summary: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Unified Intelligence Report ─────────────────────────────────────────────

class IntelligenceReport(BaseModel):
    """Final cross-track report delivered to the caller."""
    request_id: str
    target: str
    tracks_run: list[Track]
    gtm: Optional[GTMIntelligence] = None
    finance: Optional[FinanceIntelligence] = None
    security: Optional[SecurityIntelligence] = None
    executive_summary: str = ""
    top_priorities: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
