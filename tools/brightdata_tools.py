"""
NexusIntel — Bright Data Tools
================================
Wraps four Bright Data products used across all three intelligence tracks:

  1. Web Unlocker   — proxy-based scraping that bypasses bot detection
  2. SERP API       — structured real-time search results (via unlocker proxy)
  3. Web Scraper API — pre-built dataset scrapers (LinkedIn, Amazon, etc.)
  4. Scraping Browser — Playwright-powered remote browser for JS-heavy pages

All methods return plain dicts / strings that the agent layer structures
into typed Pydantic models.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import ssl
import time
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Async-safe runner ────────────────────────────────────────────────────────

def run_coro(coro, timeout: int = 90) -> Any:
    """
    Run an async coroutine safely from synchronous code.

    Uses a dedicated ThreadPoolExecutor thread so the coroutine always runs
    in a fresh event loop — avoiding the 'This function is not supported in
    asynchronous context' error that occurs when asyncio.run() is called
    from inside FastAPI's already-running event loop.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


# ─── Retry policy (shared) ───────────────────────────────────────────────────

_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


# ─── 1. Web Unlocker ─────────────────────────────────────────────────────────

class WebUnlocker:
    """
    Fetches any public URL through Bright Data Web Unlocker REST API.
    Handles CAPTCHAs, geo-blocks, and bot protection transparently.

    Uses the modern Bearer-token endpoint:
      POST https://api.brightdata.com/request
      Authorization: Bearer {BRIGHTDATA_API_KEY}

    Bright Data docs: https://docs.brightdata.com/scraping-automation/web-unlocker
    """

    _ENDPOINT = "https://api.brightdata.com/request"

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {settings.brightdata_api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=60.0)

    def fetch(self, url: str, country: Optional[str] = None) -> str:
        """
        Return HTML/text of `url`.
        Primary: Bright Data Web Unlocker (bypass bot protection).
        Fallback: direct httpx request (when no zone is configured).
        """
        if settings.brightdata_unlocker_zone:
            try:
                logger.info("[WebUnlocker] Unlocking via Bright Data: %s", url)
                payload: dict[str, Any] = {
                    "zone": settings.brightdata_unlocker_zone,
                    "url": url,
                    "format": "raw",
                }
                if country:
                    payload["country"] = country
                resp = self._client.post(self._ENDPOINT, headers=self._headers, json=payload)
                if resp.status_code == 400 and "not found" in resp.text.lower():
                    raise RuntimeError(f"Bright Data zone not found — falling back to direct fetch")
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                logger.warning("[WebUnlocker] BD failed (%s), using direct HTTP", exc)

        # Direct fallback
        logger.info("[WebUnlocker] Direct fetch: %s", url)
        resp = self._client.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text

    def fetch_json(self, url: str, country: Optional[str] = None) -> Any:
        """Fetch URL and parse JSON response."""
        return json.loads(self.fetch(url, country=country))

    @_retry
    def fetch_with_retry(self, url: str) -> str:
        return self.fetch(url)


# ─── 2. SERP API ─────────────────────────────────────────────────────────────

class SERPApi:
    """
    Returns structured real-time search results from Google/Bing.
    Routes requests through Bright Data Web Unlocker (REST API).

    Docs: https://docs.brightdata.com/scraping-automation/serp-api
    """

    _GOOGLE_URL = "https://www.google.com/search"
    _BING_URL = "https://www.bing.com/search"

    def __init__(self) -> None:
        self._unlocker = WebUnlocker()

    @_retry
    def google_search(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us",
    ) -> list[dict[str, Any]]:
        """
        Perform a Google search and return a list of structured results.

        Each result dict has: title, url, snippet, position.
        """
        url = (
            f"{self._GOOGLE_URL}?q={quote_plus(query)}"
            f"&num={num_results}&gl={country}&hl=en"
        )
        html = self._unlocker.fetch(url, country=country)
        return self._parse_google_html(html, query)

    @_retry
    def bing_search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Perform a Bing search and return structured results."""
        url = f"{self._BING_URL}?q={quote_plus(query)}&count={num_results}"
        html = self._unlocker.fetch(url)
        return self._parse_bing_html(html, query)

    def _parse_google_html(self, html: str, query: str) -> list[dict[str, Any]]:
        """
        Parse Google search result HTML using regex patterns that match
        Google's current rendered output from Bright Data.
        """
        import re
        results: list[dict[str, Any]] = []

        try:
            # Pattern 1: /url?q= links with h3 titles (classic Google structure)
            pattern1 = re.findall(
                r'href="/url\?q=(https?://[^&"]+)[^"]*"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
                html, re.DOTALL
            )
            for url, title in pattern1:
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                clean_url = url.split("&")[0]
                if clean_title and clean_url and "google.com" not in clean_url:
                    results.append({
                        "title": clean_title,
                        "url": clean_url,
                        "snippet": "",
                        "position": len(results) + 1,
                    })

            # Pattern 2: data-ved anchors with jsname (modern Google structure)
            if not results:
                pattern2 = re.findall(
                    r'<a[^>]+href="(https?://(?!www\.google)[^"]+)"[^>]*jsname[^>]*>'
                    r'.*?<h3[^>]*class="[^"]*LC20lb[^"]*"[^>]*>(.*?)</h3>',
                    html, re.DOTALL
                )
                for url, title in pattern2[:10]:
                    clean_title = re.sub(r"<[^>]+>", "", title).strip()
                    if clean_title:
                        results.append({
                            "title": clean_title,
                            "url": url,
                            "snippet": "",
                            "position": len(results) + 1,
                        })

            # Pattern 3: extract all external hrefs from <a> tags + nearby text
            if not results:
                links = re.findall(
                    r'href="(https?://(?!www\.google\.com|accounts\.google)[^"]{10,})"',
                    html
                )
                seen: set[str] = set()
                for url in links:
                    domain = re.sub(r"https?://(?:www\.)?", "", url).split("/")[0]
                    if domain and domain not in seen and len(domain) > 4:
                        seen.add(domain)
                        results.append({
                            "title": domain,
                            "url": url,
                            "snippet": "",
                            "position": len(results) + 1,
                        })
                    if len(results) >= 10:
                        break

            # Extract snippets from <span> or <div> near each result
            snippets = re.findall(r'<span[^>]*class="[^"]*VwiC3b[^"]*"[^>]*>(.*?)</span>', html, re.DOTALL)
            for i, snippet_html in enumerate(snippets[:len(results)]):
                results[i]["snippet"] = re.sub(r"<[^>]+>", "", snippet_html).strip()[:200]

        except Exception as exc:
            logger.warning("[SERP] HTML parse failed: %s", exc)

        logger.info("[SERP] Google parsed %d results for '%s'", len(results), query)
        return results[:10]

    def _parse_bing_html(self, html: str, query: str) -> list[dict[str, Any]]:
        """Parse Bing result HTML."""
        results: list[dict[str, Any]] = []
        try:
            import re
            matches = re.findall(r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html)
            for i, (url, title) in enumerate(matches[:10]):
                results.append({"title": title, "url": url, "snippet": "", "position": i + 1})
        except Exception as exc:
            logger.warning("[SERP] Bing parse failed: %s", exc)
        return results


# ─── 3. Web Scraper API ───────────────────────────────────────────────────────

class WebScraperApi:
    """
    Bright Data Web Scraper API — pre-built structured scrapers for
    660+ sites including LinkedIn, Amazon, Google Maps, and more.

    Workflow:
      1. trigger_collection(dataset_id, inputs)  → snapshot_id
      2. poll_snapshot(snapshot_id)              → "ready" or "running"
      3. get_snapshot_data(snapshot_id)          → list[dict]

    Docs: https://docs.brightdata.com/scraping-automation/web-scraper-api
    """

    _BASE_URL = "https://api.brightdata.com/datasets/v3"

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {settings.brightdata_api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=90.0)

    def trigger_collection(
        self,
        dataset_id: str,
        inputs: list[dict[str, Any]],
        notify_webhook: Optional[str] = None,
    ) -> str:
        """
        Trigger a dataset collection.

        Common dataset IDs:
          gd_l1vikfch901nx3by4  — Google SERP
          gd_l1vikfch9l3vvkz96  — LinkedIn Company
          gd_lz11l67o2cb3r0lkj3 — LinkedIn People
          gd_lwhherz21dqcscf0zb — Amazon Products
          gd_m794d2gkl1kl8oo3j  — Indeed Jobs

        Returns a snapshot_id to poll.
        """
        params = {"dataset_id": dataset_id, "include_errors": "true"}
        if notify_webhook:
            params["notify"] = notify_webhook

        logger.info("[WebScraperAPI] Triggering dataset %s with %d inputs", dataset_id, len(inputs))
        resp = self._client.post(
            f"{self._BASE_URL}/trigger",
            headers=self._headers,
            params=params,
            json=inputs,
        )
        resp.raise_for_status()
        data = resp.json()
        snapshot_id = data.get("snapshot_id", "")
        logger.info("[WebScraperAPI] snapshot_id=%s", snapshot_id)
        return snapshot_id

    def poll_snapshot(self, snapshot_id: str, timeout_seconds: int = 120) -> str:
        """
        Poll until snapshot is ready. Returns 'ready' or raises TimeoutError.
        """
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            resp = self._client.get(
                f"{self._BASE_URL}/snapshot/{snapshot_id}/progress",
                headers=self._headers,
            )
            resp.raise_for_status()
            status = resp.json().get("status", "")
            logger.debug("[WebScraperAPI] snapshot %s status=%s", snapshot_id, status)
            if status == "ready":
                return "ready"
            if status == "failed":
                raise RuntimeError(f"Snapshot {snapshot_id} failed")
            time.sleep(3)
        raise TimeoutError(f"Snapshot {snapshot_id} timed out after {timeout_seconds}s")

    def get_snapshot_data(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Fetch the collected data rows for a completed snapshot."""
        resp = self._client.get(
            f"{self._BASE_URL}/snapshot/{snapshot_id}",
            headers=self._headers,
            params={"format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("results", [])

    def collect_and_wait(
        self,
        dataset_id: str,
        inputs: list[dict[str, Any]],
        timeout_seconds: int = 120,
    ) -> list[dict[str, Any]]:
        """Convenience: trigger, poll, return data in one call."""
        snapshot_id = self.trigger_collection(dataset_id, inputs)
        self.poll_snapshot(snapshot_id, timeout_seconds)
        return self.get_snapshot_data(snapshot_id)


# ─── 4. Scraping Browser ──────────────────────────────────────────────────────

class ScrapingBrowser:
    """
    Bright Data Scraping Browser — remote Chromium via Playwright.
    Handles JS rendering, infinite scroll, SPAs, and interactive sites.

    Docs: https://docs.brightdata.com/scraping-automation/scraping-browser
    """

    def __init__(self) -> None:
        self._wss = settings.scraping_browser_wss

    async def fetch_js_page(self, url: str, wait_selector: Optional[str] = None) -> str:
        """
        Navigate to `url` in Bright Data's remote browser and return full HTML.
        Optionally wait for a CSS selector before extracting content.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            raise RuntimeError("playwright not installed — run: pip install playwright && playwright install chromium")

        logger.info("[ScrapingBrowser] Connecting to remote browser for %s", url)
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(self._wss)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=25000)
                if wait_selector:
                    await page.wait_for_selector(wait_selector, timeout=15000)
                html = await page.content()
                return html
            finally:
                await browser.close()

    async def extract_text(self, url: str, selector: str = "body") -> str:
        """Return inner text of the first matching element."""
        html = await self.fetch_js_page(url)
        try:
            from playwright.async_api import async_playwright  # type: ignore
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:5000]
        except Exception:
            return html[:2000]
