# app/orchestrator.py
import re
import json
import yaml
from typing import Optional, Dict, Any, List

from .llm_providers import call_llm
from .web_tools import web_answer
from .prompts import (
    TIER1_SYSTEM,
    TIER1_USER_TEMPLATE,
    TIER2_SYSTEM,
    TIER2_USER_TEMPLATE,
)

CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

class Orchestrator:
    """
    Tier-1: fast, conversational, uses RAG; returns JSON self-report.
    Tier-2: deeper reasoning; invoked on freshness or reasoning-intent cues,
            OR when Tier-1 is weak. Web is disabled if features.use_web = false.
    Cool-down prevents back-to-back escalations.
    """

    def __init__(self, rag):
        self.rag = rag
        self.policy = CONFIG["policy"]
        self.features = CONFIG.get("features", {})
        self.cooldown_turns = int(self.features.get("tier2_cooldown_turns", 2))
        self._session_meta: Dict[str, Dict[str, int]] = {}  # {user_id: {"turn": int, "last_tier2_turn": -999}}

    async def handle_chat(self, user_id: str, message: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        turn = self._bump_turn(user_id)

        # 1) Retrieve internal context
        docs = self.rag.retrieve(message, top_k=self.policy["top_k"], namespace=namespace)
        context_text = "\n\n".join([f"[{d['source']}] {d['text']}" for d in docs])
        sims = [d.get("score", 0.0) for d in docs]
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        # 2) Tier-1 attempt
        t1_prompt = {
            "system": TIER1_SYSTEM,
            "user": TIER1_USER_TEMPLATE.format(user_message=message, context=context_text),
        }
        t1_raw = await call_llm(tier="tier1", prompt=t1_prompt)
        t1_json = self._parse_tier1_json(t1_raw)
        conf = float(t1_json.get("confidence", 0.0) or 0.0)

        # 3) Decide escalation
        # Never escalate for greetings/thanks/closings
        if self._is_trivial_smalltalk(message) or self._is_gratitude_or_closing(message):
            return self._t1_response(t1_json, docs)

        reasons: List[str] = []
        # Weak T1 signals
        if avg_sim < self.policy["min_similarity"]:
            reasons.append(f"low_retrieval({avg_sim:.2f}<{self.policy['min_similarity']})")
        if conf < self.policy["min_self_confidence"]:
            reasons.append(f"low_confidence({conf:.2f}<{self.policy['min_self_confidence']})")
        if bool(t1_json.get("needs_web", False)):
            reasons.append("tier1_requested_web")

        # Cues
        fresh = self._looks_fresh(message) if self.features.get("tier2_on_freshness", True) else False
        intent_reason = self._looks_reasoning_intent(message) if self.features.get("tier2_on_reasoning_intent", True) else False

        # Cool-down: if within cooldown, be stricter (require strong cue + not purely trivial)
        if self._within_tier2_cooldown(user_id, turn, self.cooldown_turns):
            # only escalate if freshness OR reasoning intent AND (weak T1)
            weak_t1 = ("low_retrieval" in " ".join(reasons)) or ("low_confidence" in " ".join(reasons)) or ("tier1_requested_web" in reasons)
            should_escalate = (fresh or intent_reason) and weak_t1
        else:
            # OUTSIDE cooldown:
            # Escalate if:
            #   - freshness OR reasoning intent (regardless of T1 confidence), OR
            #   - weak T1 (low retrieval/confidence)
            weak_t1 = ("low_retrieval" in " ".join(reasons)) or ("low_confidence" in " ".join(reasons)) or ("tier1_requested_web" in reasons)
            should_escalate = fresh or intent_reason or weak_t1

        if not should_escalate:
            return self._t1_response(t1_json, docs)

        # 4) Tier-2 reasoning (web disabled unless features.use_web true)
        needs_web = bool(self.features.get("use_web", False)) and fresh
        web_ctx, web_citations = ("", [])
        if needs_web:
            web_ctx, web_citations = await web_answer(message)  # bulletproof

        difficulty = {
            "avg_retrieval_similarity": avg_sim,
            "tier1_confidence": conf,
            "reasons": sorted(set(reasons + (["freshness"] if fresh else []) + (["reasoning_intent"] if intent_reason else []))),
            "needs_web": needs_web,
        }

        t2_prompt = {
            "system": TIER2_SYSTEM,
            "user": TIER2_USER_TEMPLATE.format(
                user_message=message,
                retrieved_context=context_text,
                difficulty_report=json.dumps(difficulty),
                web_context=web_ctx,
            ),
        }
        t2 = await call_llm(tier="tier2", prompt=t2_prompt)
        t2 = _strip_think_blocks(t2)

        #advisory = "\n\n_Tier 2 AI answered this_" #-----------------------------------CHECK TO SEE WHEN TIER 2 AI IS BEING USED-----------------------------------------------------
        #if advisory.strip().lower() not in (t2 or "").lower(): #                                              |
            #t2 = (t2 or "").strip() + advisory #-------------------------------------------------------------------------------------------------------------------------------------

        self._mark_tier2(user_id, turn)

        return {
            "tier": "tier2",
            "answer": t2,
            "citations": [d.get("source") for d in docs] + web_citations,
            "difficulty_report": {"escalated": True, **difficulty},
        }

    # ----- helpers -----
    def _t1_response(self, t1_json: Dict[str, Any], docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "tier": "tier1",
            "answer": t1_json.get("answer", "") or "",
            "citations": [d.get("source") for d in docs],
            "difficulty_report": {"escalated": False, "reasons": t1_json.get("reasons", [])},
        }

    def _parse_tier1_json(self, t1_raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(t1_raw)
            if not isinstance(data, dict):
                raise ValueError("Tier-1 did not return a JSON object.")
            data.setdefault("answer", "")
            data.setdefault("confidence", 0.0)
            data.setdefault("needs_web", False)
            data.setdefault("reasons", [])
            return data
        except Exception:
            return {"answer": (t1_raw or "").strip() or "I'm not fully sure.", "confidence": 0.0, "needs_web": False, "reasons": ["unparseable self-report"]}

    def _looks_fresh(self, message: str) -> bool:
        for rx in self.policy.get("needs_freshness_patterns", []):
            if re.search(rx, message):
                return True
        return False

    def _looks_reasoning_intent(self, message: str) -> bool:
        for rx in self.policy.get("reasoning_intent_patterns", []):
            if re.search(rx, message):
                return True
        return False

    def _is_trivial_smalltalk(self, message: str) -> bool:
        m = message.strip().lower()
        common = {"hi", "hey", "hello", "hi there", "hey there", "yo", "sup", "hiya", "hola", "good morning", "good afternoon", "good evening"}
        return (len(m) <= 20 and m in common)

    def _is_gratitude_or_closing(self, message: str) -> bool:
        m = message.strip().lower()
        patterns = [r"^thanks[.!]?$", r"^thank you[.!]?$", r"^thx[.!]?$", r"^ty[.!]?$", r"^ok(ay)?[.!]?$", r"^got it[.!]?$", r"^cool[.!]?$", r"^bye[.!]?$", r"^goodbye[.!]?$", r"^see you[.!]?$"]
        return any(re.match(p, m) for p in patterns)

    def _bump_turn(self, user_id: str) -> int:
        meta = self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        meta["turn"] += 1
        return meta["turn"]

    def _mark_tier2(self, user_id: str, turn: int):
        self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        self._session_meta[user_id]["last_tier2_turn"] = turn

    def _within_tier2_cooldown(self, user_id: str, turn: int, cooldown_turns: int) -> bool:
        meta = self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        last_t2 = meta.get("last_tier2_turn", -999)
        return (turn - last_t2) <= cooldown_turns
