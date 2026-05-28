"""
NexusIntel — LangGraph Multi-Track Orchestrator
=================================================
Manages the full intelligence pipeline across all three tracks:

  [gtm_node]      ─┐
  [finance_node]  ─┼─→  [synthesis_node]  →  END
  [security_node] ─┘

All three track nodes execute in parallel (LangGraph fan-out).
The synthesis node runs after all track nodes complete.

Each node wraps the corresponding agent and handles errors gracefully —
a single track failure never aborts the whole pipeline.

Usage:
    from orchestrator import run_intelligence_pipeline

    report = run_intelligence_pipeline(IntelligenceRequest(
        target="Salesforce",
        tracks=[Track.GTM, Track.FINANCE, Track.SECURITY],
    ))
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.gtm_agent import GTMAgent
from agents.finance_agent import FinanceAgent
from agents.security_agent import SecurityAgent
from agents.synthesis_agent import SynthesisAgent
from models.schemas import (
    FinanceIntelligence,
    GTMIntelligence,
    IntelligenceReport,
    IntelligenceRequest,
    SecurityIntelligence,
    Track,
)

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Shared LangGraph state ───────────────────────────────────────────────────

class NexusState(TypedDict, total=False):
    # Input
    request_id: str
    target: str
    tracks: list[Track]
    competitors: list[str]
    vendors: list[str]
    context: Optional[str]
    webhook_url: Optional[str]

    # Track outputs
    gtm_result: Optional[GTMIntelligence]
    finance_result: Optional[FinanceIntelligence]
    security_result: Optional[SecurityIntelligence]

    # Final
    report: Optional[IntelligenceReport]
    errors: list[str]


# ─── Node functions ───────────────────────────────────────────────────────────

def gtm_node(state: NexusState) -> NexusState:
    if Track.GTM not in state.get("tracks", []):
        logger.info("[Node] gtm_node skipped")
        return {}
    logger.info("[Node] gtm_node: '%s'", state["target"])
    try:
        agent = GTMAgent()
        result = agent.run(
            target=state["target"],
            competitors=state.get("competitors"),
            context=state.get("context"),
        )
        return {"gtm_result": result}
    except Exception as exc:
        logger.error("[Node] gtm_node failed: %s", exc)
        return {"errors": state.get("errors", []) + [f"gtm:{exc}"]}


def finance_node(state: NexusState) -> NexusState:
    if Track.FINANCE not in state.get("tracks", []):
        logger.info("[Node] finance_node skipped")
        return {}
    logger.info("[Node] finance_node: '%s'", state["target"])
    try:
        agent = FinanceAgent()
        result = agent.run(
            target=state["target"],
            context=state.get("context"),
        )
        return {"finance_result": result}
    except Exception as exc:
        logger.error("[Node] finance_node failed: %s", exc)
        return {"errors": state.get("errors", []) + [f"finance:{exc}"]}


def security_node(state: NexusState) -> NexusState:
    if Track.SECURITY not in state.get("tracks", []):
        logger.info("[Node] security_node skipped")
        return {}
    logger.info("[Node] security_node: '%s'", state["target"])
    try:
        agent = SecurityAgent()
        result = agent.run(
            target=state["target"],
            vendors=state.get("vendors"),
            context=state.get("context"),
        )
        return {"security_result": result}
    except Exception as exc:
        logger.error("[Node] security_node failed: %s", exc)
        return {"errors": state.get("errors", []) + [f"security:{exc}"]}


def synthesis_node(state: NexusState) -> NexusState:
    logger.info("[Node] synthesis_node: '%s'", state["target"])
    try:
        agent = SynthesisAgent()

        # Merge webhook from request into synthesis delivery
        import os as _os
        webhook_urls_env = _os.environ.get("ALERT_WEBHOOK_URLS", "")
        if state.get("webhook_url"):
            _os.environ["ALERT_WEBHOOK_URLS"] = state["webhook_url"]

        report = agent.run(
            request_id=state["request_id"],
            target=state["target"],
            tracks_run=state.get("tracks", []),
            gtm=state.get("gtm_result"),
            finance=state.get("finance_result"),
            security=state.get("security_result"),
            errors=state.get("errors", []),
        )

        if state.get("webhook_url"):
            _os.environ["ALERT_WEBHOOK_URLS"] = webhook_urls_env

        return {"report": report}
    except Exception as exc:
        logger.error("[Node] synthesis_node failed: %s", exc)
        return {"errors": state.get("errors", []) + [f"synthesis:{exc}"]}


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build the NexusIntel LangGraph.

    The three track nodes run in parallel (fan-out via Send API is not
    required here since each node checks state.tracks itself).
    LangGraph executes nodes without declared edges in parallel by default
    when they share a common predecessor (START).
    """
    graph = StateGraph(NexusState)

    # Register nodes
    graph.add_node("gtm_node", gtm_node)
    graph.add_node("finance_node", finance_node)
    graph.add_node("security_node", security_node)
    graph.add_node("synthesis_node", synthesis_node)

    # Fan-out from START to all three track nodes
    graph.set_entry_point("gtm_node")

    # All three track nodes converge into synthesis
    graph.add_edge("gtm_node", "synthesis_node")
    graph.add_edge("finance_node", "synthesis_node")
    graph.add_edge("security_node", "synthesis_node")
    graph.add_edge("synthesis_node", END)

    # Add finance and security as parallel branches from the same start
    # LangGraph handles this via explicit parallel entry when we use add_node
    # We also wire START → finance_node and START → security_node
    from langgraph.graph import START
    graph.add_edge(START, "gtm_node")
    graph.add_edge(START, "finance_node")
    graph.add_edge(START, "security_node")

    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_intelligence_pipeline(request: IntelligenceRequest) -> IntelligenceReport:
    """
    Execute the full NexusIntel multi-track intelligence pipeline.

    Args:
        request: IntelligenceRequest with target, tracks, and options.

    Returns:
        IntelligenceReport — the unified cross-track intelligence brief.
    """
    request_id = str(uuid.uuid4())
    logger.info("NexusIntel pipeline START | request_id=%s target='%s' tracks=%s",
                request_id, request.target, [t.value for t in request.tracks])

    graph = build_graph()

    initial_state: NexusState = {
        "request_id": request_id,
        "target": request.target,
        "tracks": request.tracks,
        "competitors": [],
        "vendors": [],
        "context": request.context,
        "webhook_url": request.webhook_url,
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    report: Optional[IntelligenceReport] = final_state.get("report")
    if report is None:
        # Build a minimal report if synthesis failed entirely
        report = IntelligenceReport(
            request_id=request_id,
            target=request.target,
            tracks_run=request.tracks,
            errors=final_state.get("errors", ["Pipeline produced no report"]),
        )

    logger.info(
        "NexusIntel pipeline COMPLETE | request_id=%s errors=%d",
        request_id,
        len(report.errors),
    )
    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <target> [gtm,finance,security]")
        sys.exit(1)

    target_arg = sys.argv[1]
    tracks_arg = sys.argv[2] if len(sys.argv) > 2 else "gtm,finance,security"
    selected_tracks = [Track(t.strip()) for t in tracks_arg.split(",")]

    req = IntelligenceRequest(target=target_arg, tracks=selected_tracks)
    result = run_intelligence_pipeline(req)
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
