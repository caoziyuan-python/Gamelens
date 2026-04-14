# Design Decisions

## 1. Per-game isolated RAG index vs shared global index

**Background**: Initial version used 6 AI-generated static industry documents.
Two problems: retrieval returned generic best practices unrelated to the specific
game; a shared index causes cross-game data contamination.

**Decision**: Each game gets its own FAISS index (`indices/{game_name}.faiss`)
built automatically from that game's real App Store reviews after each analysis run.

**Tradeoff**: Index build adds ~2-3s per run. Acceptable - analysis already takes
10-30s, and real review retrieval quality justifies the overhead.

## 2. 3 tabs vs 8 tabs

**Background**: Original 8-tab structure mapped to engineering modules, not PM workflows.

**Decision**: Reorganized around 3 questions a PM actually asks:
"How do I compare vs competitors?" / "What are users saying?" / "What should I build?"

**Tradeoff**: Advanced features (cache, feedback) moved to sidebar settings,
less discoverable. Acceptable - not core PM workflows.

## 3. Topic-aware sampling: 2000+ reviews -> ~80

**Background**: Full GPT-4o analysis on 2000+ reviews costs ~$2-3 per game,
takes 3-4 minutes.

**Decision**: Reviews classified by rule-based keyword topic tags, then sampled
proportionally from each topic bucket (~80 total). Maintains theme coverage
while reducing API calls.

**Tradeoff**: Topics with <2% frequency may be underrepresented. Validated by
confirming all major topic categories have at least one sample. 96% cost
reduction with 90%+ topic coverage.

## 4. Rule-based fallback for all LLM features

**Background**: LLM calls fail due to API limits, network issues, or cost constraints.

**Decision**: Every LLM feature has a rule-based fallback. App remains fully
functional without any LLM calls. UI indicates which mode is active.

**Tradeoff**: Fallback quality is lower but functional.

## 5. Azure OpenAI + local FAISS vs OpenAI API + cloud vector DB

**Background**: Direct OpenAI API has compliance and latency issues for CN-based users.
Cloud vector DBs add external dependency and ongoing cost.

**Decision**: Azure OpenAI for enterprise compliance and CN-region latency;
local FAISS for zero dependency and zero ongoing cost.

**Tradeoff**: No managed scaling. Acceptable - local FAISS handles 100K+ vectors
on a single machine, sufficient for game review scale.

## 6. Observable Agent execution

**Background**: Single LLM calls and multi-step Agents look identical to users.

**Decision**: Every Agent step executes and displays real results in real-time
via st.status(). Users see actual retrieved review counts, not pre-written messages.

**Tradeoff**: More verbose UI. Transparency builds trust and lets PMs verify
the Agent is working with relevant data before trusting output.

## 7. Chinese UI, English documentation

**Background**: Target users are CN-based game PMs managing overseas titles.
GitHub audience is technical interviewers.

**Decision**: UI and all LLM outputs in Chinese for PM usability; README and
design docs in English for professional GitHub presentation.

**Tradeoff**: None - the two audiences have different needs and different entry points.
