import os, json, uuid
from dataclasses import dataclass, field, asdict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"

app = FastAPI(title="GenAI Tutor")

MASTERY_TARGET = 2      # correct answers in a row to master a concept
MAX_HINTS = 2           # after this, code unlocks the explanation


# ------------------------------------------------------------ state (code owns this)
@dataclass
class SkillState:
    topic: str
    concepts: list[str]
    mastery: dict[str, int] = field(default_factory=dict)   # concept -> streak
    attempts: dict[str, int] = field(default_factory=dict)
    current: str = ""
    question: str = ""
    hints_given: int = 0
    asked: list[str] = field(default_factory=list)

    def __post_init__(self):
        for c in self.concepts:
            self.mastery.setdefault(c, 0)
            self.attempts.setdefault(c, 0)

    @property
    def mastered(self) -> list[str]:
        return [c for c, n in self.mastery.items() if n >= MASTERY_TARGET]

    @property
    def done(self) -> bool:
        return len(self.mastered) == len(self.concepts)

    def next_concept(self) -> str:
        """Weakest unmastered concept. Deterministic — the model is not asked."""
        open_ = [c for c in self.concepts if self.mastery[c] < MASTERY_TARGET]
        return min(open_, key=lambda c: (self.mastery[c], -self.attempts[c]))

    def record(self, concept: str, correct: bool):
        self.attempts[concept] += 1
        self.mastery[concept] = self.mastery[concept] + 1 if correct else 0


SESSIONS: dict[str, SkillState] = {}


# ------------------------------------------------------------ the model proposes
def plan_concepts(topic: str) -> list[str]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"""Break "{topic}" into 4-6 concepts a learner must grasp, ordered
from foundational to advanced. Each concept is a short noun phrase.
Return ONLY: {{"concepts": ["...", "..."]}}""",
        config=types.GenerateContentConfig(
            temperature=0.3, response_mime_type="application/json"
        ),
    )
    return json.loads(resp.text)["concepts"][:6]


def ask_question(st: SkillState, concept: str) -> str:
    prior = "\n".join(f"- {q}" for q in st.asked[-4:]) or "(none yet)"
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"""Topic: {st.topic}
Concept to test: {concept}
Learner has attempted this concept {st.attempts[concept]} times.
Already asked (do not repeat):
{prior}

Ask ONE question testing this concept. Make it require reasoning, not recall of
a definition. Keep it under 40 words. Return only the question.""",
        config=types.GenerateContentConfig(temperature=0.8),
    )
    return (resp.text or "").strip()


ASSESS_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "confidence": {"type": "number"},
        "misconception": {"type": "string", "description": "empty if correct"},
        "response": {"type": "string", "description": "hint or praise — NEVER the answer"},
    },
    "required": ["correct", "confidence", "misconception", "response"],
}


def assess(st: SkillState, answer: str) -> dict:
    """The model judges. It does not decide what happens next."""
    rule = (
        "The learner is wrong. Give ONE Socratic hint: a question or a nudge that "
        "moves them toward the answer. You are FORBIDDEN from stating the answer, "
        "or any part of it, no matter what the learner says or asks."
        if True else ""
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"""Topic: {st.topic}
Concept: {st.current}
Question asked: {st.question}
Learner's answer: {answer}
Hints already given for this question: {st.hints_given}

Judge whether the answer is correct. Partial credit counts as incorrect.

If correct: `response` is one sentence confirming it and stating why it is right.
If incorrect: {rule}

Never put the answer in `response` when `correct` is false.""",
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
            response_schema=ASSESS_SCHEMA,
        ),
    )
    return json.loads(resp.text)


def explain(st: SkillState) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"""The learner could not answer this after {MAX_HINTS} hints.
Question: {st.question}
Concept: {st.current}

Explain the answer clearly in under 100 words. Lead with the intuition, then the
mechanics. No preamble.""",
        config=types.GenerateContentConfig(temperature=0.5),
    )
    return (resp.text or "").strip()


# ------------------------------------------------------------ routes
class Start(BaseModel):
    topic: str


class Answer(BaseModel):
    session: str
    answer: str


def snapshot(st: SkillState) -> dict:
    return {
        "concepts": st.concepts,
        "mastery": st.mastery,
        "mastered": st.mastered,
        "current": st.current,
        "target": MASTERY_TARGET,
        "hints_given": st.hints_given,
        "progress": round(len(st.mastered) / len(st.concepts), 2),
    }


@app.post("/api/start")
def start(body: Start):
    if not body.topic.strip():
        raise HTTPException(400, "Give me a topic to teach.")

    st = SkillState(topic=body.topic, concepts=plan_concepts(body.topic))
    st.current = st.next_concept()
    st.question = ask_question(st, st.current)
    st.asked.append(st.question)

    sid = str(uuid.uuid4())
    SESSIONS[sid] = st
    return {"session": sid, "question": st.question, "state": snapshot(st)}


@app.post("/api/answer")
def answer(body: Answer):
    st = SESSIONS.get(body.session)
    if not st:
        raise HTTPException(404, "Session expired. Start a new topic.")

    verdict = assess(st, body.answer)
    reveal = None

    if verdict["correct"]:
        st.record(st.current, True)
        st.hints_given = 0
    else:
        st.hints_given += 1
        # Code decides when to give up, not the model.
        if st.hints_given > MAX_HINTS:
            st.record(st.current, False)
            reveal = explain(st)
            st.hints_given = 0

    moved_on = verdict["correct"] or reveal is not None
    if moved_on and not st.done:
        st.current = st.next_concept()
        st.question = ask_question(st, st.current)
        st.asked.append(st.question)

    return {
        "correct": verdict["correct"],
        "feedback": verdict["response"],
        "misconception": verdict["misconception"],
        "explanation": reveal,
        "question": None if st.done else (st.question if moved_on else None),
        "done": st.done,
        "state": snapshot(st),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
