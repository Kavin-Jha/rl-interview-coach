# src/agent.py
# DSA Interview Coach Agent
# Orchestrates UCB bandit (problem selection) + REINFORCE (coaching strategy)
# with GPT as the conversational brain

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from src.problems import PROBLEMS, get_problem_by_id
from src.bandit import UCBBandit
from src.policy_gradient import REINFORCECoach, ACTIONS

load_dotenv()

TOPICS = [
    "Arrays & Hashing", "Sliding Window", "Stack",
    "Binary Search", "Linked Lists", "Graphs",
    "Dynamic Programming", "Trees"
]

class DSACoachAgent:
    def __init__(self, use_gpt: bool = True):
        self.use_gpt = use_gpt
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if use_gpt else None
        self.bandit = UCBBandit(n_problems=len(PROBLEMS))
        self.coach = REINFORCECoach()

        # Student state
        self.scores = []
        self.streak = 0
        self.attempts = 0
        self.current_problem = None
        self.session_log = []

    def _get_topic_id(self, topic: str) -> int:
        return TOPICS.index(topic) if topic in TOPICS else 0

    def _compute_reward(self, score: float, attempts: int, used_hint: bool) -> float:
        """Reward signal for both RL components."""
        reward = score
        if attempts == 1:
            reward += 0.2   # bonus for first try
        if used_hint:
            reward -= 0.1   # small penalty for needing hint
        return float(np.clip(reward, 0.0, 1.0))

    def _get_state(self, difficulty: int, topic: str) -> np.ndarray:
        avg_score = float(np.mean(self.scores)) if self.scores else 0.5
        return self.coach.encode_state(
            avg_score=avg_score,
            streak=self.streak,
            difficulty=difficulty,
            topic_id=self._get_topic_id(topic),
            attempts=self.attempts
        )

    def _ask_gpt(self, system_prompt: str, user_message: str) -> str:
        if not self.use_gpt:
            return "mock response"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def present_problem(self) -> dict:
        """UCB bandit selects next problem."""
        seen_ids = [log["problem_id"] for log in self.session_log]
        problem_id = self.bandit.select_problem(excluded_ids=seen_ids)
        self.current_problem = get_problem_by_id(problem_id)
        self.attempts = 0
        return self.current_problem

    def get_coaching_response(self, student_answer: str, score: float) -> str:
        """REINFORCE selects coaching action, GPT executes it."""
        p = self.current_problem
        state = self._get_state(p["difficulty"], p["topic"])
        action_id, _ = self.coach.select_action(state)
        action_name = self.coach.get_action_name(action_id)

        used_hint = action_name == "give_hint"
        reward = self._compute_reward(score, self.attempts + 1, used_hint)

        # Store transition for later update
        self.coach.store_transition(state, action_id, reward)

        # Build GPT prompt based on chosen action
        action_instructions = {
            "give_hint":           f"Give a helpful hint for this problem. Hint: {p['hint']}",
            "ask_followup":        "Ask a probing follow-up question about their approach.",
            "increase_difficulty": "Encourage them and suggest they are ready for a harder problem.",
            "decrease_difficulty": "Be encouraging, suggest trying a slightly easier problem first.",
            "explain_solution":    f"Explain the solution approach clearly. Approach: {p['solution_approach']}",
            "encourage":           "Give motivational encouragement and ask them to try again.",
        }

        system_prompt = f"""You are an expert DSA interview coach. 
Problem: {p['title']} ({p['topic']}, difficulty {p['difficulty']}/3)
Student score on this attempt: {score:.1f}/1.0
Your coaching action: {action_instructions[action_name]}
Be concise (2-3 sentences), warm, and specific to the problem."""

        response = self._ask_gpt(system_prompt, f"Student answer: {student_answer}")

        self.attempts += 1
        self.scores.append(score)
        if score >= 0.7:
            self.streak += 1
        else:
            self.streak = 0

        return f"[Coach action: {action_name}]\n{response}"

    def evaluate_answer(self, student_answer: str) -> float:
        """Ask GPT to score the student's answer 0.0-1.0."""
        if not self.use_gpt:
            return round(np.random.uniform(0.3, 0.9), 2)
        p = self.current_problem
        system_prompt = f"""You are evaluating a DSA interview answer.
Problem: {p['title']}
Correct approach: {p['solution_approach']}
Score the student's answer from 0.0 (completely wrong) to 1.0 (perfect).
Respond with ONLY a float between 0.0 and 1.0. Nothing else."""

        score_str = self._ask_gpt(system_prompt, student_answer)
        try:
            return float(score_str.strip())
        except:
            return 0.5

    def end_session(self):
        """Run REINFORCE update at end of session."""
        avg_return = self.coach.update()
        bandit_stats = self.bandit.get_stats()
        return {
            "avg_return": avg_return,
            "total_problems": len(self.session_log),
            "avg_score": float(np.mean(self.scores)) if self.scores else 0,
            "bandit_stats": bandit_stats,
            "policy_stats": self.coach.get_policy_stats()
        }

    def log_problem_result(self, score: float):
        reward = self._compute_reward(score, self.attempts, False)
        self.bandit.update(self.current_problem["id"], reward)
        self.session_log.append({
            "problem_id": self.current_problem["id"],
            "title": self.current_problem["title"],
            "score": score,
            "attempts": self.attempts
        })