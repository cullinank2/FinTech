"""
ARCHITECTURE: PCA + Knowledge Graph + Narrative System

ARCH: TIER2_CHATBOT
ARCH: KG_SERIALIZATION_ONLY
ARCH: NO_LIVE_GRAPH_IN_CHATBOT

====================================================================
TIER 2 — CHATBOT (LLM, Non-Deterministic, Exploratory)
====================================================================

BOUNDARY:
- Chatbot is a USER-INTERACTION TOOL only
- Outputs are NOT governance-grade
- Used for exploration, not reporting

AUTHORIZED FLOW:
- app.py → render_chatbot_section(...)
- chatbot.py → StockAnalysisChatbot

RULES:
- Must NEVER be used as a source of truth
- Must NOT override Narrative Engine conclusions
- Must NOT access raw system objects directly

CONTEXT INPUTS (STRICTLY CONTROLLED):
- stock_context (dict)
- kg_subgraph (serialized dict ONLY)

CRITICAL CONSTRAINT:
- Chatbot MUST NEVER receive:
    ❌ live KnowledgeGraph object
    ❌ raw PCA model
    ❌ unrestricted dataset access

It ONLY receives:
    ✅ bounded snapshots (JSON-serializable)

====================================================================
KNOWLEDGE GRAPH INTEGRATION (TIER 2)
====================================================================

- Graph must ALWAYS be passed as a SERIALIZED DICT
- Injected via:
    set_stock_context(..., kg_subgraph=...)

RULE:
- NEVER pass live graph instances across boundaries
"""

import os
from typing import Dict, List, Optional
import pandas as pd

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Chatbot will be disabled.")

from config import (
    OPENAI_API_KEY_PLACEHOLDER,
    OPENAI_MODEL,
    CHATBOT_SYSTEM_PROMPT,
    PC1_INTERPRETATION,
    PC2_INTERPRETATION,
    QUADRANTS
)


class StockAnalysisChatbot:
    """
    Chatbot for answering questions about PCA cluster analysis.

    Attributes:
        client:               OpenAI client instance
        conversation_history: List of conversation messages
        stock_context:        Current stock context for responses
        _kg_subgraph:         Serialized KG subgraph dict (Tier 2 context)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the chatbot with OpenAI API key.

        Args:
            api_key: OpenAI API key. If None, tries to get from environment.
        """
        self.client = None
        self.conversation_history: List[Dict] = []
        self.stock_context: Dict = {}
        self._kg_subgraph: Optional[Dict] = None   # Phase 4: KG context

        if not OPENAI_AVAILABLE:
            return

        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or OPENAI_API_KEY_PLACEHOLDER

        if self.api_key and self.api_key != OPENAI_API_KEY_PLACEHOLDER:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

    def is_available(self) -> bool:
        """Check if the chatbot is available (OpenAI client initialized)."""
        return self.client is not None

    def set_stock_context(
        self,
        ticker: str,
        permno: str,
        cluster: int,
        quadrant: str,
        pc1: float,
        pc2: float,
        factor_data: Dict,
        percentiles: Dict,
        peer_count: int,
        cluster_summary: pd.DataFrame,
        # ── Phase 4: KG context (Tier 2) ──────────────────────────────────────
        kg_subgraph: Optional[Dict] = None,
    ):
        """
        Set the current stock context for the chatbot.

        Args:
            ticker:          Stock ticker symbol
            permno:          PERMNO identifier
            cluster:         Cluster number
            quadrant:        Quadrant (Q1-Q4)
            pc1:             PC1 score
            pc2:             PC2 score
            factor_data:     Dictionary of factor values
            percentiles:     Dictionary of percentile rankings
            peer_count:      Number of peers in same quadrant
            cluster_summary: Summary statistics for all clusters
            kg_subgraph:     Serialized KG subgraph dict from
                             KnowledgeGraph.serialize_subgraph() — optional.
                             When provided, structural graph facts are injected
                             into the LLM context window.
        """
        self.stock_context = {
            'ticker':          ticker,
            'permno':          permno,
            'cluster':         cluster,
            'quadrant':        quadrant,
            'pc1':             pc1,
            'pc2':             pc2,
            'factors':         factor_data,
            'percentiles':     percentiles,
            'peer_count':      peer_count,
            'cluster_summary': (
                cluster_summary.to_dict()
                if isinstance(cluster_summary, pd.DataFrame)
                else cluster_summary
            ),
        }

        # Store KG subgraph — validated to be a dict or None
        self._kg_subgraph = kg_subgraph if isinstance(kg_subgraph, dict) else None

        # Reset conversation when stock changes
        self.conversation_history = []

    def _build_context_message(self) -> str:
        """
        Build a context message from the current stock data.

        Phase 4: appends a structured KG STRUCTURAL CONTEXT block when
        a serialized subgraph is available. This block gives the LLM explicit
        structural facts — regime crowding scores, Procrustes transitions,
        quadrant positions, cluster membership — without exposing the live
        graph object (Tier 2 boundary maintained).
        """
        if not self.stock_context:
            return "No stock is currently selected."

        ctx          = self.stock_context
        quadrant_info = QUADRANTS.get(ctx['quadrant'], {})

        context = f"""
CURRENT STOCK ANALYSIS CONTEXT:

Stock: {ctx['ticker']} (PERMNO: {ctx['permno']})
Cluster: {ctx['cluster']}
Quadrant: {ctx['quadrant']} - {quadrant_info.get('name', 'Unknown')}

PCA Scores:
- PC1 (Profitability & Quality): {ctx['pc1']:.3f}
  {'Positive = High quality, profitable, financially strong' if ctx['pc1'] >= 0 else 'Negative = Riskier, lower quality, more volatile'}
- PC2 (Valuation Style): {ctx['pc2']:.3f}
  {'Positive = Value-oriented (high BM, SP, EY)' if ctx['pc2'] >= 0 else 'Negative = Growth-oriented (low BM, SP, EY)'}

Quadrant Characteristics:
{', '.join(quadrant_info.get('characteristics', []))}

Number of peers in same quadrant: {ctx['peer_count']}

Factor Values:
"""
        if ctx.get('factors'):
            for category, features in ctx['factors'].items():
                context += f"\n{category}:\n"
                for feature, value in features.items():
                    percentile = ctx.get('percentiles', {}).get(feature, 'N/A')
                    if isinstance(percentile, (int, float)):
                        context += f"  - {feature}: {value:.4f} (Percentile: {percentile:.1f}%)\n"
                    else:
                        context += f"  - {feature}: {value:.4f}\n"

        if ctx.get('cluster_summary'):
            context += "\nCluster Summary (means):\n"
            for cluster_id, data in ctx['cluster_summary'].items():
                if isinstance(data, dict):
                    context += f"Cluster {cluster_id}: "
                    if 'count' in data:
                        context += f"{data.get('count', 'N/A')} stocks\n"

        # ── Phase 4: KG structural context block ──────────────────────────────
        if self._kg_subgraph:
            context += self._build_kg_context_block()

        return context

    def _build_kg_context_block(self) -> str:
        """
        Render the serialized KG subgraph as a structured plain-text block
        for LLM context injection.

        Governance note: this method receives a bounded JSON dict (never a
        live graph object). Output is appended to the system context message —
        it is Tier 2 (practitioner tool), not Tier 1 (governance artifact).
        """
        if not self._kg_subgraph:
            return ""

        nodes  = self._kg_subgraph.get("nodes", [])
        edges  = self._kg_subgraph.get("edges", [])
        meta   = self._kg_subgraph.get("meta", {})

        if not nodes:
            return ""

        lines = [
            "",
            "=" * 60,
            "KG STRUCTURAL CONTEXT (Knowledge Graph — Tier 2 context)",
            f"Nodes: {meta.get('node_count', len(nodes))}  "
            f"Edges: {meta.get('edge_count', len(edges))}",
            "=" * 60,
        ]

        # Group nodes by type for readable presentation
        regime_nodes  = [n for n in nodes if n.get("node_type") == "regime"]
        factor_nodes  = [n for n in nodes if n.get("node_type") == "factor"]
        stock_nodes   = [n for n in nodes if n.get("node_type") == "stock"]
        cluster_nodes = [n for n in nodes if n.get("node_type") == "cluster"]
        quadrant_nodes = [n for n in nodes if n.get("node_type") == "quadrant"]
        other_nodes   = [
            n for n in nodes
            if n.get("node_type") not in
            {"regime", "factor", "stock", "cluster", "quadrant"}
        ]

        # Regime nodes — highest information density
        if regime_nodes:
            lines.append("\nREGIME NODES:")
            for n in regime_nodes:
                label      = n.get("label", n.get("id", ""))
                crowding   = n.get("crowding_score", "N/A")
                risk       = n.get("crowding_risk", "N/A")
                start      = n.get("start_date", "")
                end        = n.get("end_date", "")
                date_range = f" ({start} → {end})" if start and end else ""
                lines.append(
                    f"  {label}{date_range} | "
                    f"Crowding: {crowding} ({risk})"
                )

        # Quadrant nodes
        if quadrant_nodes:
            lines.append("\nQUADRANT NODES:")
            for n in quadrant_nodes:
                lines.append(
                    f"  {n.get('label', n.get('id', ''))} | "
                    f"PC1 {n.get('pc1_sign', '?')}, PC2 {n.get('pc2_sign', '?')}"
                )

        # Stock node for the queried ticker
        if stock_nodes:
            lines.append("\nSTOCK NODE:")
            for n in stock_nodes:
                lines.append(
                    f"  {n.get('label', n.get('id', ''))} | "
                    f"Sector: {n.get('gics_sector', 'Unknown')}"
                )

        # Cluster nodes
        if cluster_nodes:
            lines.append("\nCLUSTER NODES:")
            for n in cluster_nodes:
                lines.append(f"  {n.get('label', n.get('id', ''))}")

        # Factor nodes (condensed)
        if factor_nodes:
            factor_labels = [n.get("label", n.get("id", "")) for n in factor_nodes]
            lines.append(f"\nFACTOR NODES ({len(factor_nodes)}): "
                         + ", ".join(factor_labels[:8])
                         + ("..." if len(factor_labels) > 8 else ""))

        # Other nodes
        if other_nodes:
            lines.append("\nOTHER NODES:")
            for n in other_nodes:
                lines.append(
                    f"  [{n.get('node_type', '?')}] {n.get('label', n.get('id', ''))}"
                )

        # Edges — show regime transitions and crowding edges prominently
        if edges:
            transition_edges = [
                e for e in edges if e.get("edge_type") == "regime_transition"
            ]
            crowding_edges = [
                e for e in edges if e.get("edge_type") == "crowding_level"
            ]
            migration_edges = [
                e for e in edges if e.get("edge_type") == "migrates_to"
            ]
            other_edges = [
                e for e in edges
                if e.get("edge_type") not in
                {"regime_transition", "crowding_level", "migrates_to"}
            ]

            if transition_edges:
                lines.append("\nREGIME TRANSITIONS:")
                for e in transition_edges:
                    disp   = e.get("procrustes_disparity", "N/A")
                    sev    = e.get("severity", "")
                    major  = " [MAJOR BREAK]" if e.get("is_major_break") else ""
                    mig    = e.get("migration_pct", 0)
                    common = e.get("common_tickers", 0)
                    src    = e.get("src", "").replace("regime:", "")
                    tgt    = e.get("tgt", "").replace("regime:", "")
                    lines.append(
                        f"  {src} → {tgt} | "
                        f"Procrustes: {disp:.3f} ({sev}){major} | "
                        f"Migration: {mig:.1f}% | "
                        f"Common tickers: {common:,}"
                    )

            if crowding_edges:
                lines.append("\nCROWDING EDGES:")
                for e in crowding_edges:
                    src   = e.get("src", "").replace("regime:", "")
                    score = e.get("score", "N/A")
                    risk  = e.get("risk_level", "N/A")
                    lines.append(f"  {src} | Score: {score} | Risk: {risk}")

            if migration_edges:
                lines.append("\nQUADRANT MIGRATION EDGES:")
                for e in migration_edges[:6]:   # cap at 6 for context window
                    src   = e.get("src", "").replace("quadrant:", "")
                    tgt   = e.get("tgt", "").replace("quadrant:", "")
                    count = e.get("count", "?")
                    from_p = e.get("from_period", "")
                    to_p   = e.get("to_period", "")
                    lines.append(
                        f"  {src} → {tgt} | {count} stocks | {from_p} → {to_p}"
                    )

            if other_edges:
                lines.append(
                    f"\nOTHER EDGES ({len(other_edges)}): "
                    + ", ".join(set(e.get("edge_type", "?") for e in other_edges))
                )

        if meta.get("missing_nodes"):
            lines.append(
                f"\nNote: {len(meta['missing_nodes'])} requested node(s) not found "
                f"in graph: {', '.join(meta['missing_nodes'][:3])}"
            )

        lines += [
            "",
            "Note: KG context is Tier 2 (practitioner tool). For the deterministic",
            "governance version of this structural analysis, refer to the",
            "Tier 1 Narrative Engine output.",
            "=" * 60,
        ]

        return "\n".join(lines)

    def _get_system_prompt(self) -> str:
        """Get the system prompt with PC interpretation context."""
        base_prompt = CHATBOT_SYSTEM_PROMPT

        pc1_details = f"""
PC1 Interpretation ({PC1_INTERPRETATION['name']}):
- Explains ~{PC1_INTERPRETATION['variance_explained']}% of variance
- High values indicate: {', '.join(PC1_INTERPRETATION['high_meaning'])}
- Low values indicate: {', '.join(PC1_INTERPRETATION['low_meaning'])}
"""
        pc2_details = f"""
PC2 Interpretation ({PC2_INTERPRETATION['name']}):
- Explains ~{PC2_INTERPRETATION['variance_explained']}% of variance
- High values indicate: {', '.join(PC2_INTERPRETATION['high_meaning'])}
- Low values indicate: {', '.join(PC2_INTERPRETATION['low_meaning'])}
"""
        # Phase 4: append KG awareness note to system prompt
        kg_note = """
STRUCTURAL KNOWLEDGE GRAPH CONTEXT:
When a 'KG STRUCTURAL CONTEXT' block appears in the stock context message,
use those structural facts to inform your responses. These include regime
crowding scores, Procrustes disparity scores (structural distance between
regimes), quadrant migration rates, and cluster membership. Treat these as
authoritative data points — they come from a deterministic graph, not inference.
When discussing structural characteristics (e.g. factor crowding, regime breaks,
peer clusters), prefer citing these KG facts over general commentary.
"""
        return f"{base_prompt}\n\n{pc1_details}\n{pc2_details}\n{kg_note}"

    def get_response(self, user_message: str) -> str:
        """
        Get a response from the chatbot.

        Args:
            user_message: User's question or message

        Returns:
            Chatbot's response string
        """
        if not self.is_available():
            return "⚠️ Chatbot is not available. Please configure your OpenAI API key."

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "system", "content": self._build_context_message()},
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            formatted_input = []

            for m in messages:
                formatted_input.append({
                    "role": m["role"],
                    "content": [
                        {"type": "text", "text": m["content"]}
                    ]
                })

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                input=formatted_input,
                max_output_tokens=1000
            )

            assistant_message = ""

            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for c in item.content:
                            if hasattr(c, "type") and c.type == "output_text":
                                assistant_message += getattr(c, "text", "")

            assistant_message = assistant_message.strip() or "⚠️ Empty model response"

            self.conversation_history.append({"role": "user",      "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return assistant_message

        except Exception as e:
            return f"⚠️ Error getting response: {str(e)}"

    def get_quick_analysis(self) -> str:
        """
        Get a quick automatic analysis of the current stock.
        Does not make an API call — uses stored context only.
        """
        if not self.stock_context:
            return "No stock selected for analysis."

        ctx          = self.stock_context
        quadrant_info = QUADRANTS.get(ctx['quadrant'], {})
        kg_note       = (
            f"\n**Structural KG context loaded:** "
            f"{self._kg_subgraph['meta']['node_count']} nodes, "
            f"{self._kg_subgraph['meta']['edge_count']} edges available for queries."
            if self._kg_subgraph else ""
        )

        analysis = f"""
### Quick Analysis for {ctx['ticker']}

**Cluster Assignment:** Cluster {ctx['cluster']}

**Position:** Quadrant {ctx['quadrant']} - {quadrant_info.get('name', 'Unknown')}

**PCA Interpretation:**
- **PC1 Score ({ctx['pc1']:.3f}):** {'Above average quality/stability' if ctx['pc1'] >= 0 else 'Below average quality/stability'}
- **PC2 Score ({ctx['pc2']:.3f}):** {'Value-oriented (high BM, SP, EY)' if ctx['pc2'] >= 0 else 'Growth-oriented (low BM, SP, EY)'}

**Quadrant Characteristics:**
{chr(10).join(['• ' + c for c in quadrant_info.get('characteristics', [])])}

**Peer Group:** {ctx['peer_count']} other stocks share this quadrant{kg_note}
"""
        return analysis

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def clear_kg_context(self):
        """Remove KG subgraph from context (e.g. when switching regimes)."""
        self._kg_subgraph = None


    def call_llm_structural(self, system_prompt: str, user_prompt: str) -> str:
        """
        LLM callable for Structural Analyst (strict contract).

        Accepts:
            system_prompt, user_prompt

        Returns:
            raw text response (should be JSON)
        """
        if not self.is_available():
            return ""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,  # deterministic for structural analysis
                max_tokens=1500,
            )

            return response.choices[0].message.content or ""

        except Exception:
            return ""


def create_chatbot(api_key: Optional[str] = None) -> StockAnalysisChatbot:
    """
    Factory function to create a chatbot instance.

    Args:
        api_key: Optional OpenAI API key

    Returns:
        Configured StockAnalysisChatbot instance
    """
    return StockAnalysisChatbot(api_key=api_key)


# =============================================================================
# SAMPLE QUESTIONS FOR UI
# =============================================================================

SAMPLE_QUESTIONS = [
    "Which cluster does this stock belong to?",
    "How does this stock compare to others in its cluster?",
    "What does the PC1 score tell me about this stock?",
    "Is this stock considered high quality or risky?",
    "What are the key financial characteristics of this stock's cluster?",
    "How does this stock's leverage compare to peers?",
    "What makes this stock different from others in its quadrant?",
    "Should I be concerned about this stock's volatility?",
    # Phase 4: KG-aware questions
    "What is the crowding score for the current regime?",
    "How did the factor structure change from the prior regime?",
    "Which factors reversed sign across the last regime transition?",
    "What is the structural distance between the current and prior regime?",
]
