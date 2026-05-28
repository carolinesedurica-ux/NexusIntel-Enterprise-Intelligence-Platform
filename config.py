"""
NexusIntel — Configuration
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"

    # Bright Data
    brightdata_api_key: str = ""
    brightdata_customer_id: str = ""
    brightdata_unlocker_zone: str = "unlocker"
    brightdata_unlocker_password: str = ""
    brightdata_scraping_browser_zone: str = "scraping_browser"
    brightdata_scraping_browser_password: str = ""

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    env: str = "development"
    log_level: str = "INFO"

    # Alerts
    alert_webhook_urls: str = ""

    # ── Derived helpers ──────────────────────────────────────────

    @property
    def unlocker_proxy_url(self) -> str:
        """HTTP proxy URL for Bright Data Web Unlocker."""
        return (
            f"http://brd-customer-{self.brightdata_customer_id}"
            f"-zone-{self.brightdata_unlocker_zone}"
            f":{self.brightdata_unlocker_password}"
            f"@brd.superproxy.io:22225"
        )

    @property
    def scraping_browser_wss(self) -> str:
        """WSS endpoint for Bright Data Scraping Browser (Playwright)."""
        return (
            f"wss://brd-customer-{self.brightdata_customer_id}"
            f"-zone-{self.brightdata_scraping_browser_zone}"
            f":{self.brightdata_scraping_browser_password}"
            f"@brd.superproxy.io:9222"
        )

    @property
    def scraping_browser_https(self) -> str:
        """HTTPS endpoint for Bright Data Scraping Browser (CDP over HTTPS)."""
        return (
            f"https://brd-customer-{self.brightdata_customer_id}"
            f"-zone-{self.brightdata_scraping_browser_zone}"
            f":{self.brightdata_scraping_browser_password}"
            f"@brd.superproxy.io:9515"
        )

    @property
    def alert_webhooks(self) -> list[str]:
        return [u.strip() for u in self.alert_webhook_urls.split(",") if u.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
