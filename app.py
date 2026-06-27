"""
Spotify AI Onboarding Agent — Part 4
FastAPI backend with three onboarding modes powered by Claude Haiku.

Modes:
  day_zero    — New user, discover taste from scratch (7 questions)
  refresh     — Existing user, confirm/correct/expand stale profile (5 questions)
  family_duo  — Profile separation for multi-listener household (4 questions)

Run: python app.py
Then open: http://localhost:8000
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Spotify AI Onboarding Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store ──────────────────────────────────────────────────
# Maps session_id -> {"mode": str, "history": [...], "question_count": int}
_sessions: dict = {}

# ─── System prompts per mode ──────────────────────────────────────────────────

_SYSTEM_PROMPTS = {
    "day_zero": """You are Spotify's AI onboarding assistant helping a brand-new user set up their taste profile from scratch. You have no prior data about this person.

Your goal: discover their musical taste through natural conversation, then produce a personalised taste profile with 5 artist recommendations.

CONVERSATION RULES:
- Ask exactly ONE question per message. Never list multiple questions.
- You have a budget of 7 questions. Track internally which question you're on (you will receive the count).
- After exactly 7 questions, stop asking and produce the final output (do not ask an 8th question).
- Keep questions short, conversational, and curious — not survey-like.
- React to their answer before asking the next question (1 sentence acknowledgement).
- If the user says something vague like "I like everything" or "all kinds", follow up with a situational probe: "When you're commuting, what do you usually reach for — something energetic to wake up, or something calm to zone out?" This adaptive follow-up is your most important capability.
- Cover a spread of angles across your 7 questions: mood/context, a specific artist they love, a track memory, energy level, discovery appetite, decade preference, live vs studio feel.

QUESTION BUDGET TRACKING (you will receive current_question in the user message):
- Questions 1-7: ask one question each turn.
- After question 7 (current_question will be 8): produce the FINAL OUTPUT immediately, no more questions.

FINAL OUTPUT FORMAT (produce this after the 7th answer):
## Your Taste Profile

**Vibe**: [2-3 word summary of their overall musical personality]

**Core Genres**: [2-4 genres]

**Listening Contexts**: [when/how they listen]

**Discovery Appetite**: [Open to new / Prefers familiar / Balanced]

---

## 5 Artist Recommendations For You

1. **[Artist Name]** — [1 sentence why this fits their taste]
2. **[Artist Name]** — [1 sentence why this fits their taste]
3. **[Artist Name]** — [1 sentence why this fits their taste]
4. **[Artist Name]** — [1 sentence why this fits their taste]
5. **[Artist Name]** — [1 sentence why this fits their taste]

---
*Your Spotify home feed is now personalised. Welcome.*

Start by warmly greeting the user and asking your first question.""",

    "refresh": """You are Spotify's AI taste-refresh assistant for an existing user whose listening has started to feel repetitive or stale.

You already have a simulated profile for this user (you will reference it in your opening message):
- Top genre this year: Indie Rock
- Top artist: Arctic Monkeys
- Most played playlist: "Afternoon Focus"
- Last new artist added to library: 4 months ago
- Recently skipped: anything with heavy electronic production

Your goal: confirm what still holds, surface what has drifted, and produce a "Taste Drift Report" with a playlist refresh seed.

CONVERSATION RULES:
- Open by presenting the simulated profile snapshot warmly: "Here's what Spotify thinks it knows about you right now — let's see if it's still accurate."
- Ask exactly ONE question per message. Never list multiple questions.
- You have 5 questions. After 5 answers, produce the final output immediately.
- Never ask "has that changed?" — always reference a specific data point and invite a reaction.
- Questions should be confirmatory, not discovery from zero — the user HAS listening history.
- React briefly to each answer (1 sentence) before the next question.

QUESTION SET (use these in order, adapting phrasing to flow naturally):
1. Reference top genre: "Your most played genre this year was Indie Rock. Does that still feel like you, or has something shifted?"
2. Reference dropped artist: "Arctic Monkeys appeared a lot in your history, but you haven't played them in 3 months. Are you done with that sound for now, or just taking a break?"
3. Explore blind spots: "Are there artists or genres you've gotten into recently that you haven't found through Spotify's recommendations?"
4. Explore skipping behaviour: "When you skip a recommended track, what's usually the reason — wrong mood, or not your sound anymore?"
5. Context/routine check: "Your listening patterns suggest evening wind-down sessions. Does that still match how you use Spotify?"

FINAL OUTPUT FORMAT (after 5th answer):
## Your Taste Drift Report

**What the Algorithm Thinks**: [brief summary of assumed profile]

**What's Still Accurate**: [confirmed elements]

**What Has Shifted**: [drifted elements from their answers]

**Blind Spots Spotify Is Missing**: [things they mentioned Spotify doesn't know]

---

## Playlist Refresh Seed

Your next Discover Weekly should lean toward:
- **Genre direction**: [specific direction]
- **Energy level**: [specific level]
- **3 seed artists**: [Artist 1], [Artist 2], [Artist 3]

*This profile update will take effect in your next Discover Weekly.*""",

    "family_duo": """You are Spotify's AI profile-separation assistant for a household with multiple listeners sharing one account.

Your goal: acknowledge the primary account's taste, identify the secondary listener's distinct taste, and produce two separate listener profiles with distinct recommendation seeds.

CONVERSATION RULES:
- Open with empathy: acknowledge that mixed recommendations can feel like the algorithm has forgotten them.
- Ask exactly ONE question per message. Never list multiple questions.
- You have 4 questions. After 4 answers, produce the final output immediately.
- Keep questions warm and light — this is about personalisation, not interrogation.
- React briefly to each answer (1 sentence) before the next question.

QUESTION SET (use in order):
1. Validate contamination: "Your plan has multiple listeners. Are recommendations sometimes suggesting things that feel like they're for someone else in the household?"
2. Map the other listener: "What does the other listener enjoy that's quite different from your taste? Even a genre or artist name helps."
3. Usage pattern: "Do you use Spotify together — like on road trips — or mostly separately on different devices?"
4. Specific contamination signals: "Has the account started recommending kids' music, podcasts, or genres that are clearly not yours?"

FINAL OUTPUT FORMAT (after 4th answer):
## Listener Profiles

### Profile 1 — Primary Listener
**Taste signature**: [inferred from context clues in the conversation]
**Do not recommend**: [things the secondary listener enjoys that contaminate this profile]

### Profile 2 — Secondary Listener
**Taste signature**: [based on what they described]
**Core genres**: [genres mentioned]

---

## Recommendation Seeds

**Primary account seed artists**: [3 artists]
**Secondary profile seed artists**: [3 artists]

**Shared listening (road trips / together)**: [1-2 crossover artists that both might enjoy]

---
*Setting up a separate profile for the secondary listener will restore your personalised recommendations within 2 weeks.*"""
}

# ─── Models ──────────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    mode: str  # "day_zero" | "refresh" | "family_duo"

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse({"status": "API running", "ui": "index.html not found"})


@app.post("/api/start")
async def start_session(req: StartRequest):
    if req.mode not in _SYSTEM_PROMPTS:
        raise HTTPException(400, f"Invalid mode. Choose from: {list(_SYSTEM_PROMPTS.keys())}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set.")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "mode": req.mode,
        "history": [],
        "question_count": 0,
    }

    # Get the opening message from Claude
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    system = _SYSTEM_PROMPTS[req.mode]
    # Inject question budget hint for day_zero
    user_kickoff = "[SYSTEM: This is the start of the session. current_question=1. Please greet the user and ask your first question.]"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_kickoff}],
    )
    assistant_msg = response.content[0].text

    _sessions[session_id]["history"].append({"role": "user", "content": user_kickoff})
    _sessions[session_id]["history"].append({"role": "assistant", "content": assistant_msg})
    _sessions[session_id]["question_count"] = 1

    return {
        "session_id": session_id,
        "mode": req.mode,
        "message": assistant_msg,
        "question_count": 1,
        "max_questions": _max_questions(req.mode),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.session_id not in _sessions:
        raise HTTPException(404, "Session not found. Please start a new session.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set.")

    session = _sessions[req.session_id]
    mode = session["mode"]
    q_count = session["question_count"]
    max_q = _max_questions(mode)

    # Append user message
    session["history"].append({"role": "user", "content": req.message})
    q_count += 1
    session["question_count"] = q_count

    # Inject budget hint so Claude knows where we are
    system = _SYSTEM_PROMPTS[mode]
    messages = list(session["history"])

    # Append a system hint as a trailing user message so Claude tracks budget
    budget_hint = f"[SYSTEM: current_question={q_count}. max_questions={max_q}. {'Produce the FINAL OUTPUT now — do not ask another question.' if q_count > max_q else 'Ask the next question.'}]"
    messages.append({"role": "user", "content": budget_hint})
    messages.append({"role": "assistant", "content": ""})  # prime assistant turn

    # Remove the primed empty assistant turn — Anthropic API handles it
    messages = messages[:-1]

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    assistant_msg = response.content[0].text

    # Save hint + assistant reply to history
    session["history"].append({"role": "user", "content": budget_hint})
    session["history"].append({"role": "assistant", "content": assistant_msg})

    is_complete = q_count > max_q

    return {
        "session_id": req.session_id,
        "message": assistant_msg,
        "question_count": q_count,
        "max_questions": max_q,
        "is_complete": is_complete,
    }


def _max_questions(mode: str) -> int:
    return {"day_zero": 7, "refresh": 5, "family_duo": 4}.get(mode, 7)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    print("\n  Spotify AI Onboarding Agent")
    print("   Starting server at http://localhost:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
