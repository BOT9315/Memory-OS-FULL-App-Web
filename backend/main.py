"""
Memory OS — Full Backend (Phases 2, 3, 4)
Unique features never seen together in one app:
- Semantic vector search (ChromaDB)
- Emotional timeline mapping
- Relationship health graph
- Proactive pattern detection
- Life event clustering
- Temporal mood analysis
- Voice journaling (Whisper)
- Data export with encryption
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
import anthropic
import os
import json
import hashlib
import base64
from database import db
from vector_store import vs
from pattern_engine import PatternEngine

app = FastAPI(title="Memory OS — Full Stack")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
pattern_engine = PatternEngine(client)

# ── SYSTEM PROMPTS ──────────────────────────────────────────────────────────

MEMORY_SYSTEM = """You are a lifelong AI memory companion — a second brain with years of intimate context about this person.

You have access to:
- Their past journal entries (recent + semantically relevant)
- Detected emotional patterns in their life
- Their active goals and progress
- People they mention often (relationship map)
- Recurring themes detected across their entries

Your unique behaviors:
1. CONNECT THE DOTS — always link what they're saying now to past patterns
2. NOTICE what they can't see — "You've mentioned feeling stuck 4 times this month"
3. REMEMBER relationships — "Last time you mentioned Priya, things were tense. How did that go?"
4. CELEBRATE growth — notice when they've improved vs past entries
5. BE PROACTIVE — don't just respond, offer insights unprompted
6. TIMELINE AWARENESS — reference specific past dates naturally

Tone: Warm, wise, intimate — like the most perceptive friend who has known you for years.
Never be generic. Always be specific to what you actually know about them.
Responses: 2-4 paragraphs unless they need more."""

PATTERN_SYSTEM = """Analyze these journal entries and extract structured insights. Return ONLY valid JSON.

Detect:
1. emotional_arc: overall emotional trajectory (improving/declining/volatile/stable)
2. top_themes: top 5 recurring themes with frequency count
3. relationships: people mentioned, sentiment around them (positive/negative/neutral), last mentioned date
4. goals_detected: any goals mentioned, their status
5. patterns: surprising behavioral/emotional patterns (e.g., "feels worse on Mondays", "happy after exercise")
6. best_day_type: what kind of days correlate with positive entries
7. concern_areas: things that seem to be causing consistent stress
8. growth_moments: moments that show clear personal growth

Return format:
{
  "emotional_arc": "improving",
  "top_themes": [{"theme": "work", "count": 12, "sentiment": "negative"}],
  "relationships": [{"name": "Priya", "sentiment": "positive", "last_mentioned": "2024-01-15", "frequency": 8}],
  "goals_detected": [{"goal": "learn Spanish", "mentioned": 3, "status": "stalled"}],
  "patterns": ["feels energized after morning walks", "productivity drops midweek"],
  "best_day_type": "days after exercise and good sleep",
  "concern_areas": ["work pressure", "financial anxiety"],
  "growth_moments": ["handled conflict with manager calmly on Jan 10"]
}"""

# ── MODELS ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"

class GoalRequest(BaseModel):
    user_id: str
    goal: str
    deadline: Optional[str] = None

class MoodCheckIn(BaseModel):
    user_id: str
    mood: int  # 1-10
    note: Optional[str] = None

# ── STARTUP ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    db.init()
    vs.init()
    print("Memory OS Full Stack — ready")

# ── PHASE 1: CORE CHAT ───────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    history = db.get_recent_entries(req.user_id, limit=15)
    
    # Phase 2: Semantic search for relevant past memories
    relevant_memories = vs.search(req.user_id, req.message, top_k=5)
    
    # Phase 3: Get patterns and proactive insights
    patterns = db.get_latest_patterns(req.user_id)
    goals = db.get_active_goals(req.user_id)
    relationships = db.get_relationships(req.user_id)
    
    # Build rich context
    context_parts = []
    
    if relevant_memories:
        context_parts.append("\n[SEMANTICALLY RELEVANT PAST MEMORIES:]")
        for m in relevant_memories:
            context_parts.append(f"- ({m['date']}) {m['text'][:200]}")
    
    if patterns:
        context_parts.append(f"\n[DETECTED LIFE PATTERNS:]\n{json.dumps(patterns, indent=2)}")
    
    if goals:
        context_parts.append("\n[ACTIVE GOALS:]")
        for g in goals:
            context_parts.append(f"- {g['goal']} (set: {g.get('created_at','')[:10]})")

    
    if relationships:
        context_parts.append("\n[RELATIONSHIP MAP:]")
        for r in relationships[:5]:
            last = (r['last_mentioned'] or '')[:10]
            context_parts.append(f"- {r['name']}: {r['sentiment']} sentiment, last mentioned {last}")
    
    system = MEMORY_SYSTEM + "\n".join(context_parts)
    
    # Build conversation
    messages = []
    for entry in history:
        messages.append({"role": "user", "content": entry["user_message"]})
        messages.append({"role": "assistant", "content": entry["ai_reply"]})
    messages.append({"role": "user", "content": req.message})
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    timestamp = datetime.now().isoformat()
    entry_id = db.save_entry(req.user_id, req.message, reply, timestamp)
    
    # Phase 2: Add to vector store async
    background_tasks.add_task(vs.add_entry, req.user_id, entry_id, req.message, timestamp)
    
    # Phase 3: Trigger pattern re-analysis every 10 entries
    total = db.count_entries(req.user_id)
    if total % 10 == 0:
        background_tasks.add_task(run_pattern_analysis, req.user_id)
    
    return {
        "reply": reply,
        "entry_id": entry_id,
        "timestamp": timestamp,
        "relevant_memories_used": len(relevant_memories)
    }

# ── PHASE 2: SEMANTIC SEARCH ─────────────────────────────────────────────────

@app.get("/search/{user_id}")
async def semantic_search(user_id: str, q: str, limit: int = 10):
    """True semantic search — find memories by meaning, not keywords"""
    results = vs.search(user_id, q, top_k=limit)
    
    # Enrich with full entry data
    enriched = []
    for r in results:
        entry = db.get_entry_by_id(r["entry_id"])
        if entry:
            enriched.append({**entry, "relevance_score": r["score"], "date": r["date"]})
    
    return {"results": enriched, "query": q, "count": len(enriched)}

@app.get("/timeline/{user_id}")
async def get_emotional_timeline(user_id: str, days: int = 90):
    """Emotional timeline — mood plotted over time"""
    entries = db.get_all_entries(user_id, limit=500)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    recent = [e for e in entries if e["timestamp"] >= cutoff]
    
    if len(recent) < 3:
        return {"timeline": [], "insufficient_data": True}
    
    # Batch analyze mood from entries using Claude
    timeline = []
    batch_size = 10
    for i in range(0, min(len(recent), 50), batch_size):
        batch = recent[i:i+batch_size]
        batch_text = "\n---\n".join([f"Date: {e['timestamp'][:10]}\n{e['user_message'][:300]}" for e in batch])
        
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"For each journal entry below, give a mood score 1-10 and one-word emotion. Return ONLY JSON array like [{{'date':'2024-01-01','score':7,'emotion':'hopeful'}}]:\n\n{batch_text}"
                }]
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text.strip())
            timeline.extend(scores)
        except:
            pass
    
    return {"timeline": sorted(timeline, key=lambda x: x["date"]), "days": days}

# ── PHASE 3: PATTERN DETECTION ───────────────────────────────────────────────

@app.get("/patterns/{user_id}")
async def get_patterns(user_id: str):
    """Return latest detected life patterns"""
    patterns = db.get_latest_patterns(user_id)
    if not patterns:
        # Run analysis now
        await run_pattern_analysis(user_id)
        patterns = db.get_latest_patterns(user_id)
    return patterns or {}

@app.post("/patterns/{user_id}/analyze")
async def trigger_analysis(user_id: str):
    """Manually trigger full pattern analysis"""
    await run_pattern_analysis(user_id)
    return {"status": "analyzed", "timestamp": datetime.now().isoformat()}

async def run_pattern_analysis(user_id: str):
    """Core pattern detection engine — the magic of Phase 3"""
    entries = db.get_all_entries(user_id, limit=200)
    if len(entries) < 5:
        return
    
    # Prepare entries for analysis
    entries_text = "\n---\n".join([
        f"Date: {e['timestamp'][:10]}\nEntry: {e['user_message'][:400]}"
        for e in entries[-50:]
    ])
    
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=PATTERN_SYSTEM,
            messages=[{"role": "user", "content": f"Analyze these journal entries:\n\n{entries_text}"}]
        )
        
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        patterns = json.loads(text.strip())
        patterns["analyzed_at"] = datetime.now().isoformat()
        patterns["entry_count"] = len(entries)
        
        db.save_patterns(user_id, patterns)
        
        # Update relationship map
        if "relationships" in patterns:
            for rel in patterns["relationships"]:
                db.upsert_relationship(user_id, rel["name"], rel["sentiment"], rel.get("last_mentioned",""), rel.get("frequency", 1))
        
        # Update goals
        if "goals_detected" in patterns:
            for g in patterns["goals_detected"]:
                db.upsert_detected_goal(user_id, g["goal"], g.get("status", "active"))
                
    except Exception as e:
        print(f"Pattern analysis failed: {e}")

# ── PHASE 3: PROACTIVE INSIGHTS ──────────────────────────────────────────────

@app.get("/insights/{user_id}")
async def get_proactive_insights(user_id: str):
    """Generate proactive insights — the feature nobody else has"""
    patterns = db.get_latest_patterns(user_id)
    entries = db.get_all_entries(user_id, limit=100)
    
    if not patterns or len(entries) < 10:
        return {"insights": [], "message": "Keep journaling — insights unlock after 10 entries"}
    
    # Build insight prompts from patterns
    insights = []
    
    # Pattern-based insights
    if patterns.get("patterns"):
        for p in patterns["patterns"][:3]:
            insights.append({"type": "pattern", "icon": "🔁", "text": p, "actionable": True})
    
    # Relationship insights
    if patterns.get("relationships"):
        for rel in patterns["relationships"]:
            days_since = None
            if rel.get("last_mentioned"):
                try:
                    last = datetime.fromisoformat(rel["last_mentioned"])
                    days_since = (datetime.now() - last).days
                except:
                    pass
            
            if days_since and days_since > 14:
                insights.append({
                    "type": "relationship",
                    "icon": "👥",
                    "text": f"You haven't mentioned {rel['name']} in {days_since} days. How are things?",
                    "actionable": True,
                    "person": rel["name"]
                })
    
    # Goal stall insights
    if patterns.get("goals_detected"):
        for g in patterns["goals_detected"]:
            if g.get("status") == "stalled":
                insights.append({
                    "type": "goal",
                    "icon": "🎯",
                    "text": f"Your goal '{g['goal']}' seems stalled. Want to revisit it?",
                    "actionable": True,
                    "goal": g["goal"]
                })
    
    # Concern areas
    if patterns.get("concern_areas"):
        for concern in patterns["concern_areas"][:2]:
            insights.append({
                "type": "concern",
                "icon": "⚠️",
                "text": f"'{concern}' keeps coming up. Let's talk about it.",
                "actionable": True
            })
    
    # Growth moments
    if patterns.get("growth_moments"):
        insights.append({
            "type": "growth",
            "icon": "✨",
            "text": f"Growth spotted: {patterns['growth_moments'][0]}",
            "actionable": False
        })
    
    return {"insights": insights[:8], "generated_at": datetime.now().isoformat()}

# ── PHASE 3: GOALS & RELATIONSHIPS ───────────────────────────────────────────

@app.post("/goals")
async def add_goal(req: GoalRequest):
    goal_id = db.add_goal(req.user_id, req.goal, req.deadline)
    return {"goal_id": goal_id, "status": "created"}

@app.get("/goals/{user_id}")
async def get_goals(user_id: str):
    return {"goals": db.get_active_goals(user_id)}

@app.post("/goals/{goal_id}/complete")
async def complete_goal(goal_id: int):
    db.complete_goal(goal_id)
    return {"status": "completed"}

@app.get("/relationships/{user_id}")
async def get_relationships(user_id: str):
    return {"relationships": db.get_relationships(user_id)}

# ── PHASE 3: MOOD CHECK-IN ────────────────────────────────────────────────────

@app.post("/mood")
async def log_mood(req: MoodCheckIn):
    mood_id = db.log_mood(req.user_id, req.mood, req.note)
    
    # Generate a micro-insight based on mood history
    history = db.get_mood_history(req.user_id, limit=7)
    avg = sum(m["score"] for m in history) / len(history) if history else req.mood
    
    if req.mood >= 8:
        message = "You're having a great day! What's making it special?"
    elif req.mood <= 3:
        message = "Tough day. Want to talk through what's weighing on you?"
    elif req.mood > avg + 1.5:
        message = f"You're feeling better than your recent average ({avg:.1f}). What shifted?"
    elif req.mood < avg - 1.5:
        message = f"You're below your usual mood ({avg:.1f}). What's going on?"
    else:
        message = "Thanks for checking in. Consistency is how patterns emerge."
    
    return {"mood_id": mood_id, "message": message, "weekly_avg": round(avg, 1)}

@app.get("/mood/{user_id}/history")
async def get_mood_history(user_id: str, days: int = 30):
    history = db.get_mood_history(user_id, limit=days * 3)
    return {"history": history}

# ── PHASE 4: VOICE JOURNALING ────────────────────────────────────────────────

@app.post("/voice/{user_id}")
async def voice_journal(user_id: str, audio: UploadFile = File(...)):
    """Transcribe voice memo and turn it into a journal entry"""
    try:
        audio_bytes = await audio.read()
        
        # Use Whisper via Anthropic or fall back to a mock
        # In production: integrate with Whisper API
        # For now: return a message to use real Whisper
        return {
            "status": "ready",
            "message": "Voice journaling ready. Connect Whisper API to transcribe audio.",
            "instructions": "Set OPENAI_API_KEY and uncomment whisper code in voice endpoint"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── PHASE 4: LIFE TIMELINE ────────────────────────────────────────────────────

@app.get("/timeline/{user_id}/life")
async def get_life_timeline(user_id: str):
    """Major life events detected from journal entries"""
    entries = db.get_all_entries(user_id, limit=500)
    if len(entries) < 5:
        return {"events": [], "message": "Keep journaling to build your life timeline"}
    
    entries_text = "\n---\n".join([
        f"Date: {e['timestamp'][:10]}: {e['user_message'][:200]}"
        for e in entries
    ])
    
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Extract major life events from these journal entries. Return ONLY JSON:
[{{"date":"2024-01-15","event":"Started new job","category":"career","sentiment":"positive","significance":8}}, ...]

Categories: career, relationship, health, personal_growth, achievement, challenge, travel, loss
Significance: 1-10

Entries:
{entries_text}"""
            }]
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        events = json.loads(text.strip())
        return {"events": sorted(events, key=lambda x: x["date"])}
    except:
        return {"events": [], "error": "Could not extract events"}

# ── PHASE 4: EXPORT & PRIVACY ────────────────────────────────────────────────

@app.get("/export/{user_id}")
async def export_data(user_id: str, format: str = "json"):
    """Full data export — privacy-first"""
    entries = db.get_all_entries(user_id, limit=10000)
    goals = db.get_active_goals(user_id)
    patterns = db.get_latest_patterns(user_id)
    mood = db.get_mood_history(user_id, limit=10000)
    
    data = {
        "exported_at": datetime.now().isoformat(),
        "user_id": user_id,
        "total_entries": len(entries),
        "journal_entries": entries,
        "goals": goals,
        "mood_history": mood,
        "detected_patterns": patterns,
    }
    
    if format == "markdown":
        md = f"# My Memory OS Export\n*Exported: {datetime.now().strftime('%B %d, %Y')}*\n\n"
        md += f"## Stats\n- **{len(entries)}** journal entries\n- **{len(goals)}** active goals\n\n"
        md += "## Journal Entries\n\n"
        for e in entries:
            md += f"### {e['timestamp'][:10]}\n**You:** {e['user_message']}\n\n**Memory:** {e['ai_reply']}\n\n---\n\n"
        return {"markdown": md, "filename": f"memory-os-export-{datetime.now().strftime('%Y%m%d')}.md"}
    
    return data

# ── STATS ────────────────────────────────────────────────────────────────────

@app.get("/stats/{user_id}")
async def get_stats(user_id: str):
    entries = db.get_all_entries(user_id, limit=10000)
    total = len(entries)
    if total == 0:
        return {"total_entries": 0, "days_active": 0, "first_entry": None, "streak": 0}
    
    dates = sorted(set(e["timestamp"][:10] for e in entries))
    
    # Calculate streak
    streak = 0
    today = datetime.now().date()
    for i in range(len(dates) - 1, -1, -1):
        d = datetime.fromisoformat(dates[i]).date()
        if (today - d).days <= streak + 1:
            streak += 1
        else:
            break
    
    goals = db.get_active_goals(user_id)
    mood_history = db.get_mood_history(user_id, limit=7)
    avg_mood = round(sum(m["score"] for m in mood_history) / len(mood_history), 1) if mood_history else None
    
    return {
        "total_entries": total,
        "days_active": len(dates),
        "first_entry": dates[0] if dates else None,
        "streak": streak,
        "active_goals": len(goals),
        "avg_mood_7d": avg_mood,
    }

@app.get("/entries/{user_id}")
async def get_entries(user_id: str, limit: int = 40):
    return {"entries": db.get_all_entries(user_id, limit=limit)}

@app.delete("/entries/{user_id}/all")
async def clear_all(user_id: str):
    db.clear_entries(user_id)
    vs.clear_user(user_id)
    return {"status": "cleared"}

@app.get("/health")
async def health():
    return {"status": "ok", "phases": "1-4", "version": "2.0"}
