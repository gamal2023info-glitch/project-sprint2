"""Shared graph state, A2A hand-off messages and memory structures.

* ``A2AMessage``  - Pydantic model mirroring ``schemas/a2a_message_schema.json`` plus a small
  envelope (sender / recipient / message_type / iteration). Every hand-off between the Analyst
  and Reviewer is validated through this model.
* ``ReviewVerdict`` - the Reviewer's structured output (``is_approved``, ``feedback_notes``).
* ``GraphState``  - the ``TypedDict`` that LangGraph threads through the workflow. Anything that
  must survive a checkpoint is stored as plain JSON-able data (dicts / strings), never as live
  objects, so that the SQLite checkpointer can restore it after a process restart.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- A2A protocol
AgentName = Literal["analyst", "reviewer", "human_director"]
MessageType = Literal["draft_submission", "review_feedback"]


class ScratchpadEntry(BaseModel):
    """One intermediate reasoning / tool step (schema: ``scratchpad_memory`` items)."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    tool_input: str
    observation: str


class MarketContext(BaseModel):
    """Schema: ``market_context``. All fields optional, as in the JSON schema."""

    model_config = ConfigDict(extra="forbid")

    live_egp_usd: float | None = None
    egx30_trend: str | None = None  # kept for schema compatibility; see README on what it contains
    cbe_policy_rate: float | None = None


class A2AMessage(BaseModel):
    """Agent-to-Agent hand-off message.

    The four fields required by ``a2a_message_schema.json`` are ``draft_report``,
    ``sources_cited`` (min 1), ``reviewer_feedback`` and ``approval_status``. The envelope
    fields (sender, recipient, ...) are additive: the JSON schema does not forbid extra keys.
    """

    model_config = ConfigDict(extra="forbid")

    # envelope
    sender: AgentName
    recipient: AgentName
    message_type: MessageType
    iteration: int = Field(ge=1)
    thread_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # payload - exactly the JSON-schema fields
    draft_report: str = Field(min_length=1)
    sources_cited: list[str] = Field(min_length=1)
    reviewer_feedback: str = ""
    approval_status: bool = False
    scratchpad_memory: list[ScratchpadEntry] = Field(default_factory=list)
    market_context: MarketContext | None = None

    @field_validator("sources_cited")
    @classmethod
    def _no_blank_sources(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("sources_cited must not contain blank entries")
        return v

    def schema_payload(self) -> dict[str, Any]:
        """Dict restricted to the fields of the published JSON schema (for jsonschema validation)."""
        data = self.model_dump(
            mode="json",
            include={"draft_report", "sources_cited", "reviewer_feedback", "approval_status",
                     "scratchpad_memory", "market_context"},
            exclude_none=True,
        )
        return data


class ReviewVerdict(BaseModel):
    """Structured output of the Reviewer (LLM-as-a-Judge)."""

    model_config = ConfigDict(extra="forbid")

    is_approved: bool = Field(description="True only if the draft meets every regulatory criterion.")
    feedback_notes: str = Field(description="Concrete, rule-referenced fixes; 'APPROVED' text when approved.")


class HumanDecision(BaseModel):
    """What the human director injects when resuming the interrupted thread."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["publish", "reject"]
    reviewer: str = Field(default="unknown", description="Name / ID of the human director")
    comment: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v


# --------------------------------------------------------------------------- graph state
class GraphState(TypedDict, total=False):
    # conversational memory (trimmed / summarised by the memory_manager node)
    messages: Annotated[list[AnyMessage], add_messages]
    history_summary: str

    # scratchpad memory: append-only list of ScratchpadEntry dicts
    scratchpad: Annotated[list[dict], operator.add]
    # audit trail of RAG retrievals (trace links); append-only
    audit_links: Annotated[list[dict], operator.add]

    # working data
    thread_id: str
    iteration: int
    retrieved_context: list[dict]   # [{query, section, text}]
    facts: dict                     # ground truth used by the Reviewer (rates, FX, EGX moves)
    draft_report: str
    sources_cited: list[str]
    a2a_message: dict               # latest validated A2AMessage (model_dump, JSON mode)
    review: dict                    # latest ReviewVerdict + automated check results
    review_history: Annotated[list[dict], operator.add]

    # HITL outcome
    human_decision: dict
    status: str   # drafting | in_review | pre_publish | published | rejected | blocked
    published_report: str
