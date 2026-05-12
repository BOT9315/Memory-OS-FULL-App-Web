# Memory OS

> A life intelligence system that finds patterns in your journal, maps your relationships, tracks your emotional arc, and proactively surfaces insights — powered by Claude AI.

---

## What makes this different

This is not a notes app or a chatbot. It's a **second brain** that:

- Finds memories **by meaning**, not keywords — ask "when was I most motivated?" and it finds the right entries
- **Detects invisible patterns** — "you feel worse on Wednesdays", "your goal is stalling"
- Builds a **relationship constellation** — visual map of everyone you mention, with sentiment and frequency
- Shows your **emotional timeline** — mood graphed over months with life events marked
- Sends **proactive insights** — AI reaches out to you, not the other way around
- Supports **voice journaling** — speak instead of type (requires Whisper setup)

---

## Project Structure

```
Memory-os-full/
├── backend/
│   ├── main.py              ← FastAPI server (all 4 phases)
│   ├── database.py          ← SQLite with all tables
│   ├── vector_store.py      ← ChromaDB semantic search
│   ├── pattern_engine.py    ← Life pattern detection
│   ├── requirements.txt     ← Python dependencies
│   └── .env                 ← Your API keys (create this)
└── frontend/
    └── index.html           ← Full app UI (open in browser)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- An Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
- API credits added → [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing)

### 2. Set up the environment

```bash
cd backend
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install python-dotenv
pip install chromadb --upgrade
```

### 4. Create your `.env` file

Create a file named `.env` inside the `backend/` folder:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
DB_PATH=memory_os.db
```

No quotes. No spaces around `=`.

### 5. Fix the known bug in `main.py`

Open `main.py` and apply these two fixes:

**Fix 1 — line ~133:**
```python
# Change this:
context_parts.append(f"- {g['goal']} (set: {g['created'][:10]})")

# To this:
context_parts.append(f"- {g['goal']} (set: {g.get('created_at','')[:10]})")
```

**Fix 2 — line ~138:**
```python
# Change this:
context_parts.append(f"- {r['name']}: {r['sentiment']} sentiment, last mentioned {r['last_mentioned'][:10]}")

# To this:
last = (r['last_mentioned'] or '')[:10]
context_parts.append(f"- {r['name']}: {r['sentiment']} sentiment, last mentioned {last}")
```

### 6. Start the server

```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
DB ready: memory_os.db
Vector store: ChromaDB ready
Memory OS Full Stack — ready
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 7. Open the app

Open `frontend/index.html` directly in your browser. No extra server needed.

```bash
# Windows
start frontend/index.html

# Mac
open frontend/index.html
```

---

## The 5 Panels

| Panel | What it does |
|---|---|
| **Journal** | Core chat — every message stored with full memory context |
| **Insights** | Pattern analysis — themes, emotional arc, concerns, growth |
| **Timeline** | Mood graph over time + life events auto-extracted from entries |
| **People** | Relationship constellation — everyone you mention, with sentiment |
| **Goals** | Track goals — AI auto-detects goals from your journal entries |

---

## API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/chat` | Send a message, get AI reply with memory context |
| `GET` | `/stats/{user_id}` | Entries count, days active, streak, avg mood |
| `GET` | `/entries/{user_id}` | List recent journal entries |
| `GET` | `/search/{user_id}?q=...` | Semantic search by meaning |
| `GET` | `/patterns/{user_id}` | Full pattern analysis |
| `POST` | `/patterns/{user_id}/analyze` | Trigger re-analysis |
| `GET` | `/insights/{user_id}` | Proactive AI insights |
| `GET` | `/timeline/{user_id}?days=90` | Emotional timeline data |
| `GET` | `/timeline/{user_id}/life` | Life events from entries |
| `GET` | `/relationships/{user_id}` | Relationship map |
| `POST` | `/mood` | Log a mood check-in (1–10) |
| `GET` | `/mood/{user_id}/history` | Mood history |
| `POST` | `/goals` | Add a goal |
| `GET` | `/goals/{user_id}` | List active goals |
| `POST` | `/goals/{id}/complete` | Mark goal as complete |
| `GET` | `/export/{user_id}?format=markdown` | Export all data as Markdown |
| `DELETE` | `/entries/{user_id}/all` | Delete all data for a user |

Test all endpoints interactively at: **http://localhost:8000/docs**

---

## Enable Better Semantic Search (Optional)

By default, ChromaDB uses a local embedding model. For better results, add your OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-xxxxxxxx
```

This switches to `text-embedding-3-small` for state-of-the-art semantic search. Restart the server after adding the key.

---

## Enable Voice Journaling (Optional)

1. Add `OPENAI_API_KEY` to your `.env`
2. Install: `pip install openai`
3. Uncomment the Whisper code in the `/voice/{user_id}` endpoint in `main.py`
4. Click the 🎤 mic button in the Journal panel

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| `POST /chat` returns 500 | Check server terminal for traceback — usually API key or credits |
| `Your credit balance is too low` | Add credits at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing) |
| `ANTHROPIC_API_KEY` not found | Make sure `.env` is inside the `backend/` folder |
| `Address already in use` | Change port: `uvicorn main:app --port 8001` |
| ChromaDB errors | Run `pip install chromadb --upgrade` |
| Frontend shows CORS error | Make sure the server is running on port 8000 |
| `(venv)` not showing in terminal | Re-run `venv\Scripts\activate` from the `backend/` folder |

---

## How Patterns Work

Pattern analysis runs automatically every 10 journal entries, or you can trigger it manually from the Insights panel. Claude analyzes all your entries and returns:

- **Emotional arc** — improving / declining / volatile / stable
- **Top themes** — the 5 most recurring topics with frequency counts
- **Relationships** — people detected, their sentiment, how often mentioned
- **Goals detected** — goals mentioned in entries and their status
- **Behavioral patterns** — e.g. "feels energized after morning walks"
- **Concern areas** — recurring sources of stress
- **Growth moments** — specific moments of personal progress

---

## Data & Privacy

All data is stored **locally** on your machine:

- `backend/memory_os.db` — SQLite database with all entries, goals, moods, relationships
- `backend/chroma_db/` — Vector embeddings for semantic search

No data is sent anywhere except to the Anthropic API for generating responses. You can export everything as Markdown or delete all data from the app.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| AI | Anthropic Claude (claude-sonnet-4-20250514) |
| Database | SQLite via `sqlite3` |
| Vector Search | ChromaDB (local) |
| Embeddings | ChromaDB default / OpenAI text-embedding-3-small |
| Frontend | Pure HTML + CSS + JavaScript (no framework) |
| Fonts | DM Serif Display + DM Sans |
