# CapitalTrust EGP Financial Analyst - Evaluator-Optimizer on LangGraph

Drafts the daily brief with an **Analyst** (agentic RAG over the CBE circular + MCP FX tool + EGX CSV), has a
**Reviewer** verify it against regulatory criteria, loops until approved, then **pauses** at a human pre-publish gate
(`interrupt()`) that only a human `publish` / `reject` can complete (CBE circular s.4.2).

```
START -> analyst -> reviewer --approved--------------------------> human_gate -> finalize -> END
                       |  rejected & iteration < max                 ^  interrupt(): publish | reject
                       +--> memory_manager --> analyst               |
                       |  rejected & iteration >= max: escalate -----+  (publish is blocked, reject only)
```

## Run (Groq)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key in it:  GROQ_API_KEY=gsk_...
cd notebooks && jupyter nbconvert --to notebook --execute --inplace system_execution.ipynb   # or open it in Jupyter
```
Environment variables (all optional except the key):
`GROQ_API_KEY` (real LLM; without it the offline stand-in runs), `GROQ_MODEL` (default `llama-3.3-70b-versatile`),
`EMBEDDINGS` (`fastembed` default, or `hash`), `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`.
Resume a paused thread from any shell: `python src/workflow_graph.py resume <thread_id> publish "<director name>"`.

## Files
| File | Purpose |
|---|---|
| `src/state_schema.py` | `GraphState` TypedDict (scratchpad, A2A, audit links), `A2AMessage`, `ReviewVerdict`, `HumanDecision` (Pydantic) |
| `src/mcp_egp_server.py` | MCP server (FastMCP, stdio) exposing `fetch_egp_rate` (mock by default, optional live mode with mock fallback) |
| `src/mcp_tools.py` | MCP client: `tools/list`, `tools/call`, LangChain tool for `llm.bind_tools` |
| `src/workflow_graph.py` | Ingestion (loader -> chunking -> Chroma, fastembed embeddings), audited retriever, Analyst (Groq), Reviewer, memory manager, HITL gate, graph, SQLite checkpointer |
| `notebooks/system_execution.ipynb` | Executed demo of every acceptance criterion |

## Design decisions worth knowing
* **Reviewer = hard checks + judge.** Nine rules (R1-R9) are verified programmatically against ground truth built from the
  retrieved circular, the CSV and the MCP output - not from the draft. With a real LLM, an LLM-as-a-Judge returns the
  `ReviewVerdict`, but it can never approve a draft that fails a hard check.
* **Persistent checkpointer = SQLite, not `MemorySaver`.** `MemorySaver` lives in process memory, so a paused thread would be
  lost on restart. `make_checkpointer(None)` still gives `MemorySaver` for tests.
* **Sources are computed, not generated.** `sources_cited` is built from what was actually retrieved/called, so the model cannot invent citations.
* **Sliding window + summary** (`compact_history`) keeps the last `HISTORY_WINDOW` messages and summarises the rest; the Analyst prompt also receives only the latest feedback + summary.
* **Audit:** every retrieval writes to a SHA-256 hash-chained JSONL log (`runtime/audit_log.jsonl`) and, if configured, a Langfuse trace; the human decision (who, when, draft hash) is logged too.
* `publish` only records the release in state/audit log - delivery to clients (email etc.) is intentionally out of scope.

## Groq specifics
* **LLM:** `langchain_groq.ChatGroq` (tool calling for `bind_tools`, structured output for the Reviewer's `ReviewVerdict`).
* **Embeddings:** Groq has no embeddings API, so Chroma uses local `fastembed` (`BAAI/bge-small-en-v1.5`, downloaded once). If it cannot load, the code prints a warning and falls back to lexical hash embeddings.
* **Robustness added for a real model:** a malformed/missing tool call forces the MCP call; the draft step receives the tool result as plain text; if the judge call fails, the Reviewer decides on the deterministic checks alone and says so in `judge`; `max_retries=6` for rate limits.
* **Rate limits:** Groq's free tier limits tokens per minute/day. A full notebook run makes a few dozen calls; if you hit 429s, wait and re-run, or set another `GROQ_MODEL`.
* **`stress_first_draft=True`** (demo switch, default off, used by the notebook for threads A and C): the first prompt asks the model to write carelessly, so that reject -> revise -> approve is demonstrable on demand. Without it a strong model may pass on the first try, and then no rejection would occur.

## Honest limitations
* **Not executed with Groq by me.** This environment has no network or packages, so the Groq changes were only syntax-checked. Notebook outputs were cleared; run it once with your key to produce the real outputs and trace links.
* **Offline mode is a stand-in.** Without `GROQ_API_KEY`, `OfflineDemoChatModel` assembles the brief from facts in the prompt and *deliberately* seeds three errors in its first draft so the loop is demonstrable.
* **Langfuse links:** without Langfuse credentials the notebook prints `local-audit://...` fallback links, not Langfuse URLs. The Langfuse code path follows the v3+ SDK but has not been exercised - run once with your keys to confirm.
* **Data caveats:** the CSV has 3 sessions (2024-03-05..07), so no 30-day averages or EGX30 level exist; the circular is dated 2023-11-12, so briefs carry a staleness note; the FX rate is a mock (47.85).
* The schema example cites "Circular No. 45/2023" and "COMI.CA"; the actual circular is `CBE-MPD-2023-049` and the CSV uses bare tickers, so the system follows the real data (R1 rejects the other number).
* Starter bug fixed: `FastMCP(description=...)` is not a valid argument (it is `instructions`).
