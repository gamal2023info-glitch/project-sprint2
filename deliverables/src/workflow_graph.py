"""The CapitalTrust Analyst / Reviewer LangGraph workflow (Evaluator-Optimizer + HITL gate).

START -> analyst -> reviewer --approved--------------------------> human_gate -> finalize -> END
                       |  rejected (iteration < max)                  ^   (interrupt: publish / reject)
                       +--> memory_manager --> analyst                |
                       |  rejected (iteration >= max): escalate ------+  (a human can only reject)

Sections
  1. configuration / paths
  2. embeddings + CBE ingestion pipeline (loader -> chunking -> Chroma)
  3. audit: hash-chained local log + Langfuse tracing around every RAG retrieval
  4. EGX market context from the CSV
  5. LLMs: real (Groq via langchain-groq) when GROQ_API_KEY is set, otherwise a clearly-labelled offline stand-in
  6. Reviewer: deterministic regulatory checks (+ LLM-as-a-Judge when a real LLM is available)
  7. graph nodes, sliding-window / summarising memory, routing
  8. graph construction, persistent checkpointer, resume helper
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, str(Path(__file__).parent))
from mcp_tools import get_exchange_rate_tool  # noqa: E402
from state_schema import (A2AMessage, GraphState, HumanDecision, MarketContext, ReviewVerdict,  # noqa: E402
                          ScratchpadEntry)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chroma

# =============================================================================== 1. configuration
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "runtime"          # chroma db, sqlite checkpoints, audit log (git-ignorable)
CBE_FILE = DATA_DIR / "cbe_monetary_policy_guidelines.md"
EGX_FILE = DATA_DIR / "egx_daily_summaries.csv"
DEFAULT_DB = RUNTIME_DIR / "checkpoints.sqlite"
DEFAULT_AUDIT = RUNTIME_DIR / "audit_log.jsonl"

try:                                    # pick up GROQ_API_KEY / LANGFUSE_* from deliverables/.env if present
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MAX_ITERATIONS = 3          # Analyst drafts before we stop looping and escalate to the human
HISTORY_WINDOW = 8          # messages kept verbatim; older ones are summarised
RAG_QUERIES = [
    "CBE policy rates overnight deposit rate overnight lending rate main operation discount rate",
    "exchange rate regime EGP USD foreign exchange directives",
    "AI automated advisory human-in-the-loop pre-publish audit trail requirements",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================== 2. embeddings + ingestion
class LocalHashEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words (hashing trick) embeddings.

    Last-resort fallback (no fastembed / no model download possible). Lexical, not semantic.
    """

    DIM = 768
    _STOP = {"the", "of", "and", "to", "a", "in", "for", "on", "is", "are", "be", "by", "or", "as", "with"}

    def _vec(self, text: str) -> list[float]:
        toks = [t for t in re.findall(r"[a-z0-9\.]+", text.lower()) if t not in self._STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.DIM
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbeddings(Embeddings):
    """Real semantic embeddings that run locally (ONNX, no API key, no torch).

    Groq does not offer an embeddings endpoint, so the vector store uses fastembed instead.
    The model (~130 MB) is downloaded once on first use.
    """

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding
        self.model_name = model_name or os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_EMB_CACHE: dict[str, tuple[Embeddings, str]] = {}


def get_embeddings() -> tuple[Embeddings, str]:
    """(embeddings, kind). EMBEDDINGS=fastembed (default) | hash. Falls back to hash embeddings if fastembed
    cannot be imported or its model cannot be downloaded."""
    choice = os.getenv("EMBEDDINGS", "fastembed").lower()
    if choice in _EMB_CACHE:
        return _EMB_CACHE[choice]
    if choice != "hash":
        try:
            _EMB_CACHE[choice] = (FastEmbedEmbeddings(), "fastembed")
            return _EMB_CACHE[choice]
        except Exception as exc:  # noqa: BLE001
            print(f"[embeddings] fastembed unavailable ({type(exc).__name__}: {str(exc)[:80]}); "
                  "falling back to LocalHashEmbeddings (lexical).")
    _EMB_CACHE[choice] = (LocalHashEmbeddings(), "local")
    return _EMB_CACHE[choice]


def load_and_chunk_cbe(path: Path = CBE_FILE) -> list[Document]:
    """LangChain loader -> header-aware split -> size-bounded chunks, with audit-friendly metadata."""
    raw_docs = TextLoader(str(path), encoding="utf-8").load()
    text = raw_docs[0].page_content
    circ = re.search(r"CIRCULAR NO:\*\*\s*(\S+)", text)
    issued = re.search(r"DATE OF ISSUANCE:\*\*\s*([^\n]+?)\s{0,2}\n", text)
    doc_hash = hashlib.sha256(text.encode()).hexdigest()
    header_split = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "section"), ("####", "subsection")],
        strip_headers=False)
    sections = header_split.split_text(text)
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(sections)
    for i, c in enumerate(chunks):
        label = c.metadata.get("subsection") or c.metadata.get("section") or c.metadata.get("h2", "preamble")
        c.metadata.update({
            "source": path.name, "chunk_index": i, "section_label": label,
            "circular_id": circ.group(1) if circ else "UNKNOWN",
            "issued": issued.group(1).strip() if issued else "UNKNOWN",
            "doc_sha256": doc_hash,
        })
    return chunks


def build_vectorstore(rebuild: bool = False, persist_dir: Path | None = None) -> Chroma:
    """Ingest the CBE guidelines into a persistent Chroma collection (idempotent)."""
    emb, kind = get_embeddings()
    persist_dir = persist_dir or (RUNTIME_DIR / "chroma")
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs = Chroma(collection_name=f"cbe_guidelines_{kind}", embedding_function=emb,
                persist_directory=str(persist_dir))
    if rebuild:
        vs.delete_collection()
        vs = Chroma(collection_name=f"cbe_guidelines_{kind}", embedding_function=emb,
                    persist_directory=str(persist_dir))
    if vs._collection.count() == 0:
        chunks = load_and_chunk_cbe()
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.metadata['doc_sha256']}:{c.metadata['chunk_index']}"))
               for c in chunks]
        vs.add_documents(chunks, ids=ids)
    return vs


# =============================================================================== 3. audit + tracing
class AuditLog:
    """Append-only JSONL log; each entry embeds the SHA-256 of the previous one (tamper-evident chain).

    Satisfies the 'immutable / traceable records' requirement in CBE circular s.4.2(4) in spirit; for
    production, ship these entries to WORM storage. ``verify()`` recomputes the chain.
    """

    def __init__(self, path: Path = DEFAULT_AUDIT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "GENESIS"
        with open(self.path, "rb") as f:
            last = f.read().splitlines()[-1]
        return json.loads(last)["hash"]

    def append(self, event: str, thread_id: str, payload: dict) -> dict:
        entry = {"ts": _now(), "event": event, "thread_id": thread_id, "payload": payload,
                 "prev_hash": self._last_hash()}
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self, thread_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        return [r for r in rows if thread_id is None or r["thread_id"] == thread_id]

    def verify(self) -> tuple[bool, int]:
        prev, n = "GENESIS", 0
        for row in self.entries():
            body = {k: v for k, v in row.items() if k != "hash"}
            if row["prev_hash"] != prev or hashlib.sha256(
                    json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != row["hash"]:
                return False, n
            prev, n = row["hash"], n + 1
        return True, n


class RetrievalCaptureHandler(BaseCallbackHandler):
    """LangChain callback that records what the retriever was asked and what it returned."""

    def __init__(self):
        self.query: str | None = None
        self.docs: list[dict] = []

    def on_retriever_start(self, serialized, query, **kwargs):
        self.query = query

    def on_retriever_end(self, documents, **kwargs):
        self.docs = [{"section": d.metadata.get("section_label"), "chunk_index": d.metadata.get("chunk_index"),
                      "sha256_16": hashlib.sha256(d.page_content.encode()).hexdigest()[:16]} for d in documents]


class RetrievalAuditor:
    """Wraps every retriever call with callbacks and yields a trace link.

    * Langfuse configured (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY [+ LANGFUSE_HOST/BASE_URL]):
      the call runs inside a Langfuse trace with the LangChain CallbackHandler; the link is the real
      Langfuse trace URL.
    * Otherwise: a local fallback link ``local-audit://<thread>/<trace-id>`` pointing at the hash-chained
      audit log. This is NOT a Langfuse link - the notebook says which mode it ran in.
    """

    def __init__(self, audit_log: AuditLog):
        self.audit = audit_log
        self.langfuse_enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self.mode = "langfuse" if self.langfuse_enabled else "local-audit-fallback"

    def retrieve(self, retriever, query: str, thread_id: str, iteration: int) -> tuple[list[Document], dict]:
        capture = RetrievalCaptureHandler()
        link, trace_id, lf_error = None, uuid.uuid4().hex, None
        docs: list[Document] | None = None
        if self.langfuse_enabled:
            try:
                from langfuse import Langfuse, get_client
                from langfuse.langchain import CallbackHandler
                lf = get_client()
                trace_id = Langfuse.create_trace_id()
                ctx = getattr(lf, "start_as_current_observation", None) or lf.start_as_current_span
                with ctx(trace_context={"trace_id": trace_id}, name=f"rag_retrieval[{thread_id}#{iteration}]"):
                    docs = retriever.invoke(query, config={"callbacks": [CallbackHandler(), capture]})
                lf.flush()
                link = lf.get_trace_url(trace_id=trace_id)
            except Exception as exc:  # never let tracing break the run; record the failure instead
                lf_error = f"{type(exc).__name__}: {exc}"
        if docs is None:
            docs = retriever.invoke(query, config={"callbacks": [capture]})
        if link is None:
            link = f"local-audit://{thread_id}/{trace_id}"
        entry = self.audit.append("rag_retrieval", thread_id, {
            "iteration": iteration, "query": query, "trace_id": trace_id, "trace_link": link,
            "tracer": self.mode if lf_error is None else f"local (langfuse error: {lf_error})",
            "retrieved": capture.docs})
        info = {"iteration": iteration, "query": query, "trace_link": link, "audit_hash": entry["hash"][:12],
                "tracer": self.mode if lf_error is None else "local (langfuse error)"}
        print(f"[RAG #{iteration}] {query[:58]!r} -> {len(docs)} chunks | trace: {link}")
        return docs, info


# =============================================================================== 4. EGX context
def load_egx_context(path: Path = EGX_FILE) -> dict:
    """Latest-session moves per ticker, derived only from the CSV (no outside data)."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values(["Ticker", "Date"])
    dates = sorted(df["Date"].dt.strftime("%Y-%m-%d").unique())
    latest = dates[-1]
    moves: dict[str, dict] = {}
    for tk, g in df.groupby("Ticker"):
        g = g.reset_index(drop=True)
        last, prev = g.iloc[-1], g.iloc[-2]
        prior_vol = statistics.mean(g["Volume"].iloc[:-1])
        moves[tk] = {
            "sector": last["Sector"], "close": float(last["Close"]), "prev_close": float(prev["Close"]),
            "pct_change": round((last["Close"] / prev["Close"] - 1) * 100, 1),
            "volume": int(last["Volume"]), "avg_prior_volume": round(prior_vol),
            "vol_vs_prior_pct": round((last["Volume"] / prior_vol - 1) * 100, 1),
        }
    basket = round(statistics.mean(m["pct_change"] for m in moves.values()), 1)
    out = {"latest_date": latest, "first_date": dates[0], "sessions": len(dates), "moves": moves,
           "basket_avg_pct": basket}
    return json.loads(json.dumps(out, default=float))   # plain Python types only: state must be checkpoint-serialisable


def egx_prompt_block(egx: dict) -> str:
    lines = [f"EGX daily summaries file: {egx['sessions']} sessions, {egx['first_date']} to {egx['latest_date']}."]
    for tk, m in egx["moves"].items():
        lines.append(f"{tk}: close {m['close']:.2f}, prev close {m['prev_close']:.2f}, "
                     f"{m['pct_change']:+.1f}% day-on-day, volume {m['volume']:,} "
                     f"({m['vol_vs_prior_pct']:+.1f}% vs mean of prior sessions)")
    lines.append(f"Equal-weighted basket day-on-day change: {egx['basket_avg_pct']:+.1f}%")
    return "\n".join(lines)


_RATE_PATTERNS = {
    "overnight_deposit": r"Overnight Deposit Rate:\*\*\s*([\d.]+)%",
    "overnight_lending": r"Overnight Lending Rate:\*\*\s*([\d.]+)%",
    "main_operation": r"Rate of the Main Operation:\*\*\s*([\d.]+)%",
    "discount": r"Discount Rate:\*\*\s*([\d.]+)%",
}


def parse_policy_rates(text: str) -> dict[str, float]:
    out = {}
    for k, pat in _RATE_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
    return out


# =============================================================================== 5. LLMs
class OfflineDemoChatModel(BaseChatModel):
    """Deterministic stand-in for a chat model, used when GROQ_API_KEY is absent.

    It is NOT a language model. It (a) emits a ``fetch_egp_rate`` tool call when tools are bound and no tool
    result is present yet, and (b) assembles the brief from the facts found in the prompt. On its first
    draft (no reviewer feedback in the prompt) it deliberately reproduces three classic LLM failure modes
    so the evaluator loop can be demonstrated; on revisions it fixes exactly the rule IDs the reviewer cited.
    """

    @property
    def _llm_type(self) -> str:
        return "offline-demo"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if kwargs.get("tools") and not any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="", tool_calls=[{"name": "fetch_egp_rate", "args": {"base_currency": "USD"},
                                                     "id": "call_fx_" + uuid.uuid4().hex[:8], "type": "tool_call"}])
        else:
            msg = AIMessage(content=_offline_compose(messages))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _offline_compose(messages: list[BaseMessage]) -> str:
    prompt = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    fx = json.loads(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage)))
    egx = json.loads(re.search(r"<<EGX_JSON>>(.*?)<<END_EGX_JSON>>", prompt, re.S).group(1))
    m_id = re.search(r"CBE-MPD-\d{4}-\d{3}", prompt)
    circ_id = m_id.group(0) if m_id else "UNKNOWN"
    issued = re.search(r"issued ([A-Z][a-z]+ \d{1,2}, \d{4})", prompt)
    issued = issued.group(1) if issued else "date not retrieved"
    rates = parse_policy_rates(prompt)
    feedback = prompt.split("REVIEWER FEEDBACK:", 1)[1] if "REVIEWER FEEDBACK:" in prompt else ""
    first_pass = not feedback
    fix = lambda code: (not first_pass) and f"[{code}]" in feedback  # noqa: E731

    cite = (f"CBE Circular {circ_id}" if (fix("R1") or not first_pass) else "CBE Circular No. 45/2023")
    L = ["CAPITALTRUST MORNING BRIEF - DRAFT",
         "Status: DRAFT. Pending human director approval; this brief must not be distributed to clients before release.",
         "", "1. Monetary policy (CBE)"]
    if rates:
        L.append(f"Per {cite} (issued {issued}), the overnight deposit rate is {rates['overnight_deposit']:.2f}% and the "
                 f"overnight lending rate is {rates['overnight_lending']:.2f}%; the rate of the main operation and the "
                 f"discount rate are both {rates['main_operation']:.2f}%.")
    L.append("The circular describes the MPC stance as restrictive and data-dependent, and reaffirms a flexible exchange rate regime.")
    L.append(f"Note: the circular is dated {issued}; confirm that no later MPC decision exists before release.")
    L += ["", "2. EGP exchange rate",
          f"The MCP exchange-rate tool returned {fx['rate']:.2f} EGP per 1 {fx['base']} (source: {fx['source']}, "
          f"as of {fx['last_updated']}).",
          "This is a mocked, indicative rate and not a live market quote." if "MOCK" in fx["source"] else
          "Rate as returned by the tool at the time shown.", "",
          f"3. EGX market context (sample of {len(egx['moves'])} tickers; latest session {egx['latest_date']})"]
    for tk, m in egx["moves"].items():
        L.append(f"- {tk} ({m['sector']}): closed at {m['close']:.2f}, {m['pct_change']:+.1f}% vs previous close "
                 f"{m['prev_close']:.2f}; volume {m['volume']:,}, {m['vol_vs_prior_pct']:+.1f}% vs the mean of prior sessions.")
    L.append(f"Equal-weighted basket move on {egx['latest_date']}: {egx['basket_avg_pct']:+.1f}%.")
    L.append(f"The file covers only {egx['sessions']} sessions ({egx['first_date']} to {egx['latest_date']}); "
             "index-level data and longer averaging windows are not in the dataset.")
    if first_pass or not fix("R4"):
        L.append("- COMI and FWRY recorded a trading volume spike of 15% above the 30-day average.")
    if first_pass or not fix("R5"):
        L.append("Real estate names such as TMGH may experience short-term volatility until secondary CBE guidelines are released.")
    else:
        L.append("Analyst Speculation: TMGH (real estate) may see continued volatility given its price range across "
                 "the sessions in the file; this is an opinion, not a CBE statement or a reported fact.")
    L += ["", "Sources: CBE circular (vector store), EGX daily summaries file, MCP exchange-rate tool."]
    return "\n".join(L)


GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
_GROQ_MODEL_CACHE: dict[str, str] = {}


def pick_groq_model(requested: str | None = None) -> str:
    """Return a Groq model that this API key can actually use.

    Groq retires models regularly (llama-3.3-70b-versatile was shut down on 2026-08-16). We ask the Models API
    which ids are live; if the requested one is missing we warn and take the first available fallback. If the
    Models API itself cannot be reached we simply return the requested id (and let the real call report errors).
    """
    requested = requested or os.getenv("GROQ_MODEL") or GROQ_FALLBACK_MODELS[0]
    if requested in _GROQ_MODEL_CACHE:
        return _GROQ_MODEL_CACHE[requested]
    chosen = requested
    try:
        import requests
        r = requests.get("https://api.groq.com/openai/v1/models", timeout=10,
                         headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"})
        r.raise_for_status()
        live = {m["id"] for m in r.json().get("data", [])}
        if live and requested not in live:
            alt = next((m for m in GROQ_FALLBACK_MODELS if m in live), None)
            if alt:
                print(f"[llm] WARNING: Groq model {requested!r} is not available for this key; using {alt!r} instead. "
                      "Set GROQ_MODEL (check .env and `echo $env:GROQ_MODEL`) to silence this.")
                chosen = alt
            else:
                print(f"[llm] WARNING: {requested!r} is not available and none of {GROQ_FALLBACK_MODELS} are either; "
                      f"live models: {sorted(live)}")
    except Exception as exc:  # noqa: BLE001 - offline / transient: keep the requested id
        print(f"[llm] could not list Groq models ({type(exc).__name__}); using {requested!r}")
    _GROQ_MODEL_CACHE[requested] = chosen
    return chosen


def get_llm() -> tuple[BaseChatModel, str]:
    """(llm, mode). mode is 'online' when Groq is used, 'offline' for the deterministic stand-in.

    Env: GROQ_API_KEY (required for online), GROQ_MODEL (default openai/gpt-oss-120b; auto-replaced if retired).
    """
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=pick_groq_model(), temperature=0, max_retries=6, timeout=90), "online"
        except ImportError:
            print("[llm] GROQ_API_KEY is set but langchain-groq is not installed (pip install langchain-groq); "
                  "using the offline stand-in.")
    return OfflineDemoChatModel(), "offline"


# =============================================================================== 6. Reviewer: regulatory criteria
CRITERIA = {
    "R1": "Cite the real CBE circular ID (CBE-MPD-2023-049); never an unknown circular number.",
    "R2": "Every CBE policy rate quoted must equal the circular's figure.",
    "R3": "The EGP/USD figure must equal the MCP tool output and disclose its source (mock vs live).",
    "R4": "EGX statistics must be derivable from the EGX CSV; no 30-day / EGX30 claims the data cannot support.",
    "R5": "Any forward-looking or hedged statement must be explicitly labelled 'Analyst Speculation'.",
    "R6": "No guarantees, price predictions stated as fact, or buy/sell directives.",
    "R7": "Must be marked DRAFT / pending human director approval (HITL pre-publish).",
    "R8": "sources_cited must be non-empty and limited to the real inputs (circular, EGX file, MCP tool).",
    "R9": "Must state the as-of dates: EGX latest session and the circular's issue date.",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\-])|\n+")
_HEDGE = re.compile(r"\b(may|might|could|likely|probably|expected to|anticipate[sd]?|potentially|should see)\b", re.I)
_PROHIBITED = re.compile(r"(guarantee[sd]?|risk-free|\bwill (?:rise|fall|increase|decrease|appreciate|depreciate|"
                         r"surge|rally|crash)\b|you should (?:buy|sell)|\b(?:strong )?(?:buy|sell) recommendation)", re.I)


def run_regulatory_checks(draft: str, sources: list[str], facts: dict) -> list[dict]:
    """Returns the list of violated rules ([] means all checks passed). Ground truth = ``facts``, built from
    the retrieved circular, the CSV and the MCP tool output, never from the draft itself."""
    issues: list[dict] = []
    add = lambda rid, msg: issues.append({"rule": rid, "criterion": CRITERIA[rid], "problem": msg})  # noqa: E731
    sents = [s.strip() for s in _SENT_SPLIT.split(draft) if s.strip()]
    low = draft.lower()

    # R1
    cid = facts.get("circular_id")
    if not cid or cid not in draft:
        add("R1", f"The draft does not cite the circular ID {cid}.")
    for ref in set(re.findall(r"Circular\s+(?:No\.?|Number)\s*([A-Za-z0-9\-/]+)", draft)):
        if ref != cid:
            add("R1", f"Cites 'Circular No. {ref}', which is not in the retrieved CBE material; the retrieved circular is {cid}.")
    # R2
    pol = facts.get("policy_rates", {})
    kw = {"overnight_deposit": "overnight deposit", "overnight_lending": "overnight lending",
          "main_operation": "main operation", "discount": "discount rate"}
    if not pol:
        add("R2", "Ground-truth policy rates could not be parsed from the retrieved circular text.")
    elif "overnight deposit" not in low:
        add("R2", "The overnight deposit rate is not stated.")
    for key, phrase in kw.items():
        for s in sents:
            if key in pol and phrase in s.lower() and not re.search(rf"(?<![\d.]){pol[key]:.2f}\s*%", s):
                add("R2", f"Sentence mentions the {phrase} but not the circular's figure {pol[key]:.2f}%: '{s[:90]}'")
    # R3
    fx = facts.get("fx", {})
    if fx:
        if f"{fx['rate']:.2f}" not in draft:
            add("R3", f"The tool rate {fx['rate']:.2f} EGP/{fx['base']} is not stated.")
        if "MOCK" in str(fx.get("source", "")).upper() and "mock" not in low:
            add("R3", "The rate comes from a mock source but the draft does not disclose that.")
        for s in sents:
            if re.search(r"EGP/USD|EGP per|exchange rate", s, re.I):
                for num in re.findall(r"(?<![\d.])\d{2,3}\.\d{2}(?![\d])", s):
                    if abs(float(num) - fx["rate"]) > 0.005:
                        add("R3", f"Quotes {num} for the exchange rate; the MCP tool returned {fx['rate']:.2f}.")
    else:
        add("R3", "No exchange-rate tool output is available to verify against.")
    # R4
    egx = facts.get("egx", {})
    allowed = {abs(m[k]) for m in egx.get("moves", {}).values() for k in ("pct_change", "vol_vs_prior_pct")}
    allowed.add(abs(egx.get("basket_avg_pct", 0.0)))
    tickers = list(egx.get("moves", {}))
    for s in sents:
        if re.search(r"30[- ]day|30[- ]session|EGX\s?30", s, re.I) and not re.search(r"not in the|not available|unavailable", s, re.I):
            add("R4", f"References a 30-day / EGX30 metric that the 3-session CSV cannot support: '{s[:90]}'")
        if any(re.search(rf"\b{t}\b", s) for t in tickers):
            for p in re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", s):
                if not any(abs(abs(float(p)) - a) <= 0.06 for a in allowed):
                    add("R4", f"Percentage {p}% in '{s[:70]}...' does not match any figure derivable from the EGX CSV.")
    # R5
    for s in sents:
        if _HEDGE.search(s) and "analyst speculation" not in s.lower():
            add("R5", f"Hedged/forward-looking statement not labelled 'Analyst Speculation': '{s[:100]}'")
    # R6
    for s in sents:
        if _PROHIBITED.search(s):
            add("R6", f"Prohibited promotional / directive language: '{s[:100]}'")
    # R7
    if "draft" not in low or "pending human director approval" not in low:
        add("R7", "Missing the 'DRAFT ... Pending human director approval' marker.")
    # R8
    if not sources:
        add("R8", "sources_cited is empty.")
    for s in sources:
        if not re.match(r"^(CBE Circular CBE-MPD-\d{4}-\d{3}|EGX Daily Summaries|MCP Tool Response)", s):
            add("R8", f"Unrecognised / unverifiable source: '{s}'")
    # R9
    if egx.get("latest_date") and egx["latest_date"] not in draft:
        add("R9", f"The EGX as-of date {egx['latest_date']} is not stated.")
    if facts.get("circular_issued") and facts["circular_issued"] not in draft:
        add("R9", f"The circular's issue date ({facts['circular_issued']}) is not stated.")
    # de-duplicate
    seen, uniq = set(), []
    for i in issues:
        k = (i["rule"], i["problem"])
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    return uniq


def format_feedback(issues: list[dict]) -> str:
    lines = ["REJECTED. Revise the draft; fix every item and do not introduce new unsupported claims:"]
    for i in issues:
        lines.append(f"[{i['rule']}] {i['problem']}")
    return "\n".join(lines)


_UNICODE_FIXES = {"\u2011": "-", "\u2010": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
                  "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200a": " ", "\u2007": " ", "\u200b": "",
                  "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."}


def normalize_text(text: str) -> str:
    """Map look-alike Unicode (non-breaking hyphens/spaces, curly quotes) to ASCII. LLMs emit these; they would
    otherwise slip past the Reviewer's regex checks (e.g. '30<U+2011>day') and break Windows consoles."""
    return text.translate(str.maketrans(_UNICODE_FIXES))


def inject_demo_faults(draft: str, circular_id: str) -> str:
    """DEMO / TEST ONLY (``stress_first_draft=True``): deterministic fault injection into the first draft so the
    Reviewer's rejection path can be shown on demand, independent of how well-behaved the LLM is."""
    d = draft.replace(circular_id, "CBE Circular No. 45/2023")
    return (d.rstrip() + "\n- COMI and FWRY recorded a trading volume spike of 15% above the 30-day average.\n"
            "Real estate names such as TMGH may experience short-term volatility until secondary CBE guidelines "
            "are released.")


# =============================================================================== 7. nodes
ANALYST_SYSTEM = (
    "You are the CapitalTrust research Analyst drafting a morning brief for a regulated Egyptian investment "
    "firm. Use ONLY the supplied circular excerpts, EGX facts and the tool result. Never invent figures.\n"
    "Hard rules (an automated reviewer checks each one and rejects the draft if any is broken):\n"
    "1. First line: 'DRAFT - Pending human director approval' (this exact wording).\n"
    "2. Cite the circular by its exact ID (e.g. CBE-MPD-2023-049) and state its issue date exactly as printed in the "
    "excerpt header. Never cite any other circular number.\n"
    "3. Every sentence that names a CBE policy rate (overnight deposit, overnight lending, main operation, discount "
    "rate) must contain that rate's figure with two decimals, exactly as in the circular.\n"
    "4. State the EGP/USD rate exactly as returned by the tool (two decimals), its source and the word 'mock' if the "
    "source says MOCK.\n"
    "5. EGX: use only the percentages and volumes given in EGX FACTS; do not compute new percentages; state the latest "
    "session date in YYYY-MM-DD. Do not mention '30-day' averages or the EGX30 index.\n"
    "6. Avoid hedging words (may, might, could, likely, expected to, potentially). If you must offer an opinion about "
    "the future, put it in its own sentence that starts with 'Analyst Speculation:'.\n"
    "7. No guarantees, no price predictions stated as fact, no buy/sell advice.\n"
    "Output the brief text only."
)
JUDGE_SYSTEM = (
    "You are the Reviewer: an LLM-as-a-Judge enforcing CapitalTrust's regulatory criteria. Approve ONLY if every "
    "criterion is met and every figure is supported by the ground truth. Criteria:\n"
    + "\n".join(f"{k}: {v}" for k, v in CRITERIA.items())
    + "\nAn automated checker has already verified R1-R9 deterministically and its findings are provided. If that list "
      "is empty, approve unless you can point to a concrete factual error against the GROUND TRUTH; do not reject over "
      "style. If you reject, give specific, rule-referenced fixes."
)


def summarize_history(old: list[BaseMessage], prior: str, llm: BaseChatModel, mode: str) -> str:
    """Running summary of the messages being dropped from the window.

    The LLM is used only when there is enough text to be worth compressing; its answer is discarded if it is
    empty, too short or looks like a request for more input, and any provider error falls back as well. The
    deterministic summary below is therefore always available and never injects junk into the Analyst prompt."""
    text = "\n".join(f"{m.type}: {str(m.content)[:600]}" for m in old if str(m.content).strip())
    if mode == "online" and len(text) >= 1500:
        try:
            out = llm.invoke([SystemMessage(content="Summarise the Analyst/Reviewer exchange below in <=120 words, "
                                                    "keeping every reviewer rule ID raised and whether it was fixed. "
                                                    "Use only the text provided; never ask for more input."),
                              HumanMessage(content=f"Prior summary: {prior}\n\n{text}")])
            summary = str(out.content).strip()
            if len(summary) >= 40 and not re.search(r"\b(I need|please provide|could you)\b", summary, re.I):
                return summary
        except Exception:  # noqa: BLE001 - fall through to the deterministic summary
            pass
    task, rules = None, set()
    for m in old:
        if isinstance(m, HumanMessage) and m.name == "cio":
            task = str(m.content)[:90]
        if isinstance(m, HumanMessage) and str(m.content).startswith("[A2A reviewer"):
            rules |= set(re.findall(r"\[(R\d)\]", str(m.content)))
    drafts = sum(1 for m in old if isinstance(m, AIMessage) and m.content)
    return ((prior + " | ") if prior else "") + (
        f"{len(old)} earlier messages compacted ({drafts} drafts" + (f"; original task: {task}" if task else "")
        + (f"; reviewer rules raised: {', '.join(sorted(rules))}" if rules else "") + ").")


def compact_history(messages: list[BaseMessage], prior_summary: str, llm: BaseChatModel, mode: str,
                    window: int) -> dict:
    """Sliding window: keep the last ``window`` messages verbatim, replace older ones with a running summary.
    Returns a state update ({} when nothing to trim). Never leaves a ToolMessage without its AI tool-call."""
    if len(messages) <= window:
        return {}
    cut = len(messages) - window
    while cut > 0 and isinstance(messages[cut], ToolMessage):
        cut -= 1
    old = messages[:cut]
    if not old:
        return {}
    return {"messages": [RemoveMessage(id=m.id) for m in old],
            "history_summary": summarize_history(old, prior_summary, llm, mode)}


def build_nodes(*, llm: BaseChatModel, mode: str, retriever, auditor: RetrievalAuditor, audit: AuditLog,
                max_iterations: int, history_window: int, stress_first_draft: bool = False):
    fx_tool = get_exchange_rate_tool()
    llm_tools = llm.bind_tools([fx_tool])
    judge = llm.with_structured_output(ReviewVerdict) if mode == "online" else None

    # ----------------------------------------------------------------- analyst (prompt chaining)
    def analyst(state: GraphState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"]["thread_id"]
        it = state.get("iteration", 0) + 1
        step0 = len(state.get("scratchpad", []))
        pad: list[ScratchpadEntry] = []
        add_pad = lambda tool, inp, obs: pad.append(  # noqa: E731
            ScratchpadEntry(step=step0 + len(pad) + 1, tool_name=tool, tool_input=inp, observation=obs[:400]))

        # chain step 1 - agentic RAG over the CBE circular (each retrieval is traced / audited)
        seen, ctx, links = set(), [], []
        for q in RAG_QUERIES:
            docs, info = auditor.retrieve(retriever, q, thread_id, it)
            links.append(info)
            for d in docs:
                if d.metadata["chunk_index"] not in seen:
                    seen.add(d.metadata["chunk_index"])
                    ctx.append({"query": q, "section": d.metadata["section_label"], "text": d.page_content,
                                "circular_id": d.metadata["circular_id"], "issued": d.metadata["issued"]})
        add_pad("rag_cbe_search", " | ".join(RAG_QUERIES), f"{len(ctx)} unique chunks: " +
                ", ".join(sorted({c['section'] for c in ctx})))
        policy_text = "\n".join(c["text"] for c in ctx)
        rates = parse_policy_rates(policy_text)
        circ_id, issued = ctx[0]["circular_id"], ctx[0]["issued"]

        # chain step 2 - EGX context from the CSV
        egx = load_egx_context()
        add_pad("egx_daily_summary_query", str(EGX_FILE.name),
                f"{egx['sessions']} sessions to {egx['latest_date']}; basket {egx['basket_avg_pct']:+.1f}%")

        # chain step 3 - the LLM calls the MCP FX tool (bind_tools); fall back to forcing the call
        feedback = (state.get("a2a_message") or {}).get("reviewer_feedback", "")
        ctx_txt = "\n\n".join(f"[source: {c['circular_id']} | issued {c['issued']} | section: {c['section']}]\n{c['text']}"
                              for c in ctx)
        task_body = (f"CBE CIRCULAR EXCERPTS:\n{ctx_txt}\n\nEGX FACTS:\n{egx_prompt_block(egx)}\n"
                     f"<<EGX_JSON>>{json.dumps(egx)}<<END_EGX_JSON>>\n")
        if state.get("history_summary"):
            task_body += f"\nEARLIER CYCLES (summary): {state['history_summary']}\n"
        if feedback:
            task_body += f"\nPREVIOUS DRAFT:\n{state.get('draft_report', '')}\n\nREVIEWER FEEDBACK:\n{feedback}\n"
        head = f"Draft today's brief (iteration {it}).\n\n"
        chain = [SystemMessage(content=ANALYST_SYSTEM),
                 HumanMessage(content=head + "First call fetch_egp_rate for USD.\n\n" + task_body)]
        try:
            first = llm_tools.invoke(chain)
            calls = list(first.tool_calls)
        except Exception as exc:  # noqa: BLE001 - e.g. Groq 'tool_use_failed' on a malformed tool call
            first, calls = AIMessage(content=""), []
            add_pad("llm_tool_call_error", "fetch_egp_rate", f"{type(exc).__name__}: {str(exc)[:200]} -> forcing the MCP call")
        if not calls:       # the model did not (validly) call the tool: force the MCP call so the chain still runs
            calls = [{"name": "fetch_egp_rate", "args": {"base_currency": "USD"}, "id": "forced_fx_" + uuid.uuid4().hex[:8]}]
            first = AIMessage(content=first.content or "", tool_calls=[{**calls[0], "type": "tool_call"}])
        tool_msgs = [ToolMessage(content=fx_tool.invoke(c["args"]), tool_call_id=c["id"], name=c["name"]) for c in calls]
        fx = json.loads(tool_msgs[0].content)
        add_pad("fetch_egp_rate", json.dumps(calls[0]["args"]), f"{fx['rate']} EGP per {fx['base']} ({fx['source']})")

        # chain step 4 - draft
        if mode == "online":
            # the draft step needs no tool access: the MCP result is passed as plain text, which avoids provider
            # quirks around replaying tool-call turns
            draft_msg = llm.invoke([SystemMessage(content=ANALYST_SYSTEM), HumanMessage(content=(
                head + "The exchange rate was fetched with the MCP tool fetch_egp_rate; its JSON result is below.\n\n"
                + task_body + f"\nMCP TOOL RESULT (fetch_egp_rate):\n{tool_msgs[0].content}\n\n"
                "Now write the brief."))])
        else:
            draft_msg = llm_tools.invoke(chain + [first] + tool_msgs)
        draft = str(draft_msg.content).strip()
        draft = normalize_text(draft)
        if not draft:
            raise RuntimeError("The LLM returned an empty draft.")
        add_pad("draft_generation", f"iteration {it}" + (" with reviewer feedback" if feedback else ""),
                f"{len(draft)} chars drafted")
        if stress_first_draft and mode == "online" and it == 1:
            draft = inject_demo_faults(draft, circ_id)
            add_pad("demo_fault_injection", "stress_first_draft=True",
                    "DEMO: wrong circular number, unsupported 30-day claim and unlabelled speculation injected")

        sources = [f"CBE Circular {circ_id}: Monetary Policy and Regulatory Framework Guidelines (issued {issued})",
                  f"EGX Daily Summaries (egx_daily_summaries.csv), sessions {egx['first_date']} to {egx['latest_date']}",
                  f"MCP Tool Response: fetch_egp_rate {fx['base']} ({fx['source']}, {fx['last_updated']})"]
        facts = {"circular_id": circ_id, "circular_issued": issued, "policy_rates": rates, "fx": fx, "egx": egx}
        mc = MarketContext(live_egp_usd=fx["rate"], cbe_policy_rate=rates.get("overnight_deposit"),
                           egx30_trend=f"EGX sample basket (8 tickers) {egx['basket_avg_pct']:+.1f}% on "
                                       f"{egx['latest_date']}; EGX30 index not in dataset")
        msg = A2AMessage(sender="analyst", recipient="reviewer", message_type="draft_submission", iteration=it,
                         thread_id=thread_id, draft_report=draft, sources_cited=sources,
                         reviewer_feedback="", approval_status=False, scratchpad_memory=pad, market_context=mc)
        audit.append("analyst_draft", thread_id, {"iteration": it, "draft_sha256": hashlib.sha256(draft.encode()).hexdigest(),
                                                  "scratchpad": [p.model_dump() for p in pad]})
        return {
            "iteration": it, "thread_id": thread_id, "retrieved_context": ctx, "facts": facts,
            "draft_report": draft, "sources_cited": sources, "a2a_message": msg.model_dump(mode="json"),
            "scratchpad": [p.model_dump() for p in pad], "audit_links": links, "status": "in_review",
            "messages": [HumanMessage(content=f"[analyst task iteration {it}]" + (f" feedback addressed: {feedback[:200]}" if feedback else ""), name="orchestrator"),
                         first, *tool_msgs, AIMessage(content=draft, name="analyst")],
        }

    # ----------------------------------------------------------------- reviewer (LLM-as-a-Judge)
    def reviewer(state: GraphState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"]["thread_id"]
        it = state["iteration"]
        draft, sources, facts = state["draft_report"], state["sources_cited"], state["facts"]
        A2AMessage.model_validate(state["a2a_message"])                # inbound hand-off must be schema-valid
        issues = run_regulatory_checks(draft, sources, facts)
        if judge is not None:
            gt = json.dumps({k: facts[k] for k in ("circular_id", "policy_rates", "fx")} | {"egx": facts["egx"]})
            try:
                v: ReviewVerdict | None = judge.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=(
                    f"GROUND TRUTH:\n{gt}\n\nAUTOMATED CHECK RESULTS:\n{json.dumps(issues, indent=1)}\n\nDRAFT:\n{draft}"))])
                if v is None:
                    raise ValueError("judge returned no structured verdict")
                judge_name = getattr(llm, "model_name", "llm-judge")
            except Exception as exc:  # noqa: BLE001 - provider hiccup: decide on the hard checks alone, and say so
                v = ReviewVerdict(is_approved=not issues, feedback_notes=(
                    f"(LLM judge unavailable: {type(exc).__name__}; decision based on automated checks only.)"))
                judge_name = f"deterministic fallback ({type(exc).__name__})"
            approved = v.is_approved and not issues          # the LLM can never override a failed hard check
            notes = (format_feedback(issues) + "\nJudge notes: " + v.feedback_notes) if issues else v.feedback_notes
            if not approved and not issues:
                notes = "REJECTED by judge. " + v.feedback_notes
        else:
            approved = not issues
            notes = ("APPROVED: all 9 regulatory criteria (R1-R9) passed automated verification against the circular, "
                     "the EGX CSV and the MCP tool output." if approved else format_feedback(issues))
            judge_name = "deterministic-evaluator (offline)"
        verdict = ReviewVerdict(is_approved=approved, feedback_notes=notes)
        out_msg = A2AMessage(sender="reviewer", recipient="human_director" if approved else "analyst",
                             message_type="review_feedback", iteration=it, thread_id=thread_id, draft_report=draft,
                             sources_cited=sources, reviewer_feedback=verdict.feedback_notes,
                             approval_status=verdict.is_approved,
                             scratchpad_memory=[ScratchpadEntry(
                                 step=len(state.get("scratchpad", [])) + 1, tool_name="reviewer_evaluation",
                                 tool_input=f"iteration {it} draft vs R1-R9",
                                 observation=("Approved." if approved else f"Rejected: {sorted({i['rule'] for i in issues})}"))],
                             market_context=MarketContext.model_validate(state["a2a_message"].get("market_context") or {}))
        rec = {"iteration": it, "is_approved": approved, "feedback_notes": notes,
               "rules_failed": sorted({i["rule"] for i in issues}), "judge": judge_name}
        audit.append("reviewer_verdict", thread_id, rec)
        update: dict[str, Any] = {
            "review": verdict.model_dump() | {"issues": issues, "judge": judge_name},
            "review_history": [rec], "a2a_message": out_msg.model_dump(mode="json"),
            "scratchpad": [out_msg.scratchpad_memory[0].model_dump()],
            "status": "pre_publish" if approved else "in_review",
        }
        if not approved:
            update["messages"] = [HumanMessage(content=f"[A2A reviewer->analyst] {notes}", name="reviewer")]
        return update

    def route_after_review(state: GraphState) -> Literal["human_gate", "memory_manager"]:
        if state["review"]["is_approved"] or state["iteration"] >= max_iterations:
            return "human_gate"
        return "memory_manager"

    # ----------------------------------------------------------------- sliding window + summary
    def memory_manager(state: GraphState) -> dict:
        return compact_history(state["messages"], state.get("history_summary", ""), llm, mode, history_window)

    # ----------------------------------------------------------------- HITL gate + finalize
    def human_gate(state: GraphState) -> dict:
        approved = state["review"]["is_approved"]
        raw = interrupt({                                    # <- graph suspends here until Command(resume=...)
            "kind": "pre_publish_hold", "thread_id": state.get("thread_id"),
            "reviewer_approved": approved, "iterations_used": state["iteration"],
            "instructions": "Resume with {'action': 'publish'|'reject', 'reviewer': '<name>', 'comment': '...'}."
                            + ("" if approved else " WARNING: automated reviewer did NOT approve; only 'reject' is permitted."),
            "draft_report": state["draft_report"], "sources_cited": state["sources_cited"],
            "reviewer_feedback": state["review"]["feedback_notes"],
        })
        decision = HumanDecision.model_validate(raw if isinstance(raw, dict) else {"action": raw})
        return {"human_decision": decision.model_dump()}

    def finalize(state: GraphState) -> dict:
        d, approved = state["human_decision"], state["review"]["is_approved"]
        ts = _now()
        if d["action"] == "publish" and approved:
            status = "published"
            body = (state["draft_report"].replace("DRAFT", "FINAL", 1)
                    + f"\n\nReleased by {d['reviewer']} at {ts} (human approval recorded in audit log).")
        elif d["action"] == "publish":
            status, body = "blocked", ""
        else:
            status, body = "rejected", ""
        audit.append("human_decision", state.get("thread_id", "?"), {**d, "status": status, "decided_at": ts,
                     "draft_sha256": hashlib.sha256(state["draft_report"].encode()).hexdigest()})
        return {"status": status, "published_report": body}

    return analyst, reviewer, route_after_review, memory_manager, human_gate, finalize


# =============================================================================== 8. graph construction
def make_checkpointer(db_path: Path | str | None = DEFAULT_DB):
    """Persistent SQLite checkpointer (survives process restarts). ``None`` -> in-memory ``MemorySaver``
    (does NOT survive a restart; fine for unit tests)."""
    if db_path is None:
        return MemorySaver()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))


def build_graph(checkpointer, *, llm: BaseChatModel | None = None, mode: str | None = None, vectorstore=None,
                audit_log: AuditLog | None = None, max_iterations: int = MAX_ITERATIONS,
                history_window: int = HISTORY_WINDOW, stress_first_draft: bool = False):
    if llm is None:
        llm, mode = get_llm()
    mode = mode or "offline"
    vectorstore = vectorstore or build_vectorstore()
    audit = audit_log or AuditLog()
    auditor = RetrievalAuditor(audit)
    nodes = build_nodes(llm=llm, mode=mode, retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                        auditor=auditor, audit=audit, max_iterations=max_iterations, history_window=history_window,
                        stress_first_draft=stress_first_draft)
    analyst, reviewer, route, memory_manager, human_gate, finalize = nodes

    g = StateGraph(GraphState)
    g.add_node("analyst", analyst)
    g.add_node("reviewer", reviewer)
    g.add_node("memory_manager", memory_manager)
    g.add_node("human_gate", human_gate)
    g.add_node("finalize", finalize)
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "reviewer")
    g.add_conditional_edges("reviewer", route, {"human_gate": "human_gate", "memory_manager": "memory_manager"})
    g.add_edge("memory_manager", "analyst")
    g.add_edge("human_gate", "finalize")
    g.add_edge("finalize", END)
    graph = g.compile(checkpointer=checkpointer)
    graph.mode, graph.auditor, graph.audit = mode, auditor, audit   # convenience for the notebook
    return graph


def initial_input(thread_id: str) -> dict:
    return {"thread_id": thread_id, "iteration": 0, "status": "drafting", "history_summary": "",
            "messages": [HumanMessage(content="Draft the CapitalTrust morning brief for market open: CBE policy "
                                              "stance, EGP/USD rate and EGX movements.", name="cio")]}


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def resume_thread(thread_id: str, action: str, reviewer: str = "unknown", comment: str = "",
                  db_path: Path | str = DEFAULT_DB, **build_kwargs) -> dict:
    """Rebuild everything from disk (new process) and resume an interrupted thread with a human decision."""
    graph = build_graph(make_checkpointer(db_path), **build_kwargs)
    cfg = thread_config(thread_id)
    snap = graph.get_state(cfg)
    if not snap.next:
        raise RuntimeError(f"Thread {thread_id!r} is not waiting for input (next={snap.next}, "
                           f"status={snap.values.get('status')}).")
    graph.invoke(Command(resume={"action": action, "reviewer": reviewer, "comment": comment}), cfg)
    return graph.get_state(cfg).values


if __name__ == "__main__":      # python workflow_graph.py resume <thread_id> <publish|reject> <reviewer>
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) >= 4 and sys.argv[1] == "resume":
        vals = resume_thread(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "unknown")
        print(json.dumps({"status": vals["status"], "decision": vals["human_decision"]}, indent=2))
        