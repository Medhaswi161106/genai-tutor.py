# GenAI Tutor

A tutor that tracks what you actually know, refuses to hand you the answer, and adapts the next question to your gaps.

**10 of 10** — part of a GenAI project series. FastAPI backend, vanilla JS frontend.

## What it demonstrates

- Stateful pedagogy: a skill model updated per turn drives the next prompt
- Socratic constraint — the hard part is stopping the model from just answering
- The 'LLM proposes, code decides' pattern: the model assesses, code owns the state machine

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
# add your key to .env  (skip if this project needs no key)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Keys

| Key | Where to get it | Where it goes |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey (free) | `.env` |

## Stack

FastAPI · google-genai · JSON schema output

## How it works

Most 'AI tutor' projects are a chatbot with a teacher-flavoured system prompt. They fail the same way: the student says "I don't know" and the model, being helpful, just tells them. Nothing is learned and nothing is tracked.

This one splits the job in two:

**The model assesses.** Each answer goes to a call that returns strict JSON: correct or not, which concept the mistake belongs to, and a confidence score. It judges. It does not decide what happens next.

**Code decides.** A `SkillState` object holds a mastery score per concept. Right answers push a score up, wrong answers push it down and queue that concept for another pass. A concept is mastered at two correct answers in a row. The next question is chosen by the weakest unmastered concept — in Python, deterministically, not by asking the model to remember.

**The Socratic constraint** lives in the prompt and is enforced by the turn structure: on a wrong answer the tutor may only give a hint, and `hints_given` is tracked in state. After two hints the code — not the model — unlocks the explanation.

This is the same architecture as any serious agent: the LLM handles judgement calls that need language understanding, and deterministic code owns state and control flow. Swap the prompts and it is a compliance checker.

---
MIT
