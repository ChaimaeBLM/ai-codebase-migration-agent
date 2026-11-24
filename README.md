# ⚡ Codebase Migration & Refactor Planner, grounded in real code

A multi-agent codebase migration assistant built with **LangGraph** and **Streamlit**. It plans file-by-file migrations, asks a human to approve the plan, runs parallel refactoring workers, and produces a migration report with diffs and risk charts.

This is an enhanced version of the [Codebase Migration Agent](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/multi_agent_apps/ai_codebase_migration_agent) from awesome-llm-apps. **The main change: the agent now reads the target repository before planning and editing**, instead of working only from the repository's name.

## What I changed

| | Original | This version |
|---|---|---|
| Repository access | The repo URL is only passed to the LLM as text. Nothing is cloned or read | A new `repo_loader` node clones the repo (or uses a local folder) |
| Planner | Plans from the repo name alone. The prompt forces 3–6 tasks | Explores the repo with **tool calling**, then plans from what it found. 1–6 tasks, real paths only |
| Workers | Receive a file name and write a diff for code they never saw | Receive the **real file content** and write a strict diff against it. Missing files are skipped |
| Approval step | Bug: approving always failed (the paused run was lost on every Streamlit rerun) | Fixed: the graph is kept in `st.session_state` |
| UI | | New "🔍 Repository exploration" panel listing every tool call the planner made |

### The planner's tools

| Tool | What it does |
|---|---|
| `list_files()` | Lists the repository's files (skips `.git`, `node_modules`, virtualenvs, build folders, files over 1 MB) |
| `search_code(pattern)` | Text or regex search across files, including notebook code cells |
| `read_file(path)` | Reads one file. Notebooks are returned as their code cells |

The tools are **read-only**, refuse any path outside the repository, cap their output size, and the exploration loop stops after **8 rounds**. A plan revision reuses the first exploration, so it doesn't spend extra LLM calls.

The prompts were also rewritten: only use paths returned by the tools, include files that call the migrated code, and state whether the code is the **client** or the **provider** of the API being migrated.

## Results

Test case: [GitHub-Documentation-Rag-Agent](https://github.com/Mandoa-Labs/GitHub-Documentation-Rag-Agent), with the goal *psycopg2 → psycopg 3* and *Ollama `/api/embeddings` → `/api/embed`*. The correct answer is 3 files: `src/postgres_db.py`, `src/llama.py`, `requirements.txt`.

| Metric | Original | This version |
|---|---|---|
| Planned files that exist | 1 / 10 (2 runs) | 3 / 3 |
| Required files found (of 3) | 0 and 1 | 3 |
| API direction understood (client of Ollama) | No | Yes |
| Report diffs that apply (`git apply --check`) | None (target files don't exist) | 2 / 3 |
| Report diffs fully correct | 0 | 1 / 3 |

These come from a small number of runs with free models, so treat them as indicative rather than a benchmark.

## Known limitations

- **A diff can compile and still be wrong.** In the test run, the `src/llama.py` diff kept the old `"prompt"` field, which `/api/embed` doesn't accept. Only a test would catch this, and the test repo has none. Always review diffs before applying them.
- **The aggregator rewrites the diffs** when it builds the report, and can add details that don't exist in the code (class names, infrastructure).
- **Diffs are not validated automatically** yet.
- **Notebooks** get new cell contents instead of a diff, so they can't be checked with `git apply`.
- **The graph's memory is in RAM.** Restarting the app loses a plan that's waiting for approval.

## Next steps

1. Insert the workers' diffs into the report as-is, and let the aggregator write only the summary.
2. Validate each diff with `git apply --check` and a compile check, and show the result per file.
3. Generate characterization tests before migrating repos that have none.
4. Persistent checkpoints (e.g. SQLite), tracing, and cost per run.
5. Support a self-hosted model for code that can't leave a client's network.

## How to run

**Requirements:** Python 3.10+ and **git** (used to clone the target repository).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
cp .env.example .env    # then add your API key
streamlit run app.py
```

In the app, enter a repository URL (e.g. `github.com/Mandoa-Labs/GitHub-Documentation-Rag-Agent`) or a local folder path, plus a migration goal.

### LLM configuration

Set these in `.env` or in the app's sidebar:

```env
OPENAI_API_KEY=sk-your-key-here
# Optional:
MODEL_FAST=gpt-5-mini        # validator, planner, workers
MODEL_PRO=gpt-5.5            # final report
LLM_BASE_URL=                # any OpenAI-compatible API
```

**Using OpenRouter (free models available):**

```env
OPENAI_API_KEY=sk-or-v1-your-key
LLM_BASE_URL=https://openrouter.ai/api/v1
MODEL_FAST=<model-id>:free
MODEL_PRO=<model-id>:free
```

Choose models that support **tool calling** and **JSON mode**. Restart the app after editing `.env`.

## How it works

```
Repo + goal → Validator → Repo loader → Planner (explores with tools, then plans)
           → Human approval → Parallel workers (one per file, real content) → Aggregator → Report
```

1. **Query validator** checks that a repository and a meaningful goal were provided.
2. **Repo loader** clones the repository (cached between runs) or uses a local folder.
3. **Planner** explores the repo with its tools, writes down its findings, then produces a strategy and file-by-file tasks with risk levels.
4. **Approval gate** pauses the graph with LangGraph's `interrupt()`. You approve, or ask for changes in plain language.
5. **Workers** run in parallel with `Send()`, one per file, each with the file's real content.
6. **Aggregator** writes the final report, with charts rendered from data only (no generated code is executed).

## Tech stack

| Layer | Technology |
|---|---|
| Agent graph | LangGraph (`Send()` fan-out, `interrupt()` approval, `MemorySaver` checkpointer) |
| Repo exploration | LangChain tool calling, `git`, read-only file tools |
| LLMs | Any OpenAI-compatible API (OpenAI, OpenRouter, local) |
| Charts | Matplotlib, from validated data |
| UI | Streamlit |

## Credits and license

Based on the Codebase Migration Agent from [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) by Shubham Saboo, licensed under Apache-2.0. This version modifies `app.py` (repository loading, planner tools and prompts, workers, approval fix, UI panel) and this README. It is distributed under the same Apache-2.0 license.
