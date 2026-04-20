# main.py
# Simulation runner for the DSA Interview Coach Agent
# Runs N sessions, tracks learning progress, saves results

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
import matplotlib.pyplot as plt
from src.agent import DSACoachAgent
from src.problems import PROBLEMS

# ── Simulated student ──────────────────────────────────────────────────────────
def simulate_student_answer(problem: dict, session_num: int) -> tuple[str, float]:
    """
    Simulate a student who gradually improves over sessions.
    Score increases with session number, harder problems score lower.
    """
    base_score = 0.3 + (session_num / 40.0)
    difficulty_penalty = (problem["difficulty"] - 1) * 0.15
    noise = np.random.normal(0, 0.08)
    score = float(np.clip(base_score - difficulty_penalty + noise, 0.0, 1.0))

    answer_templates = [
        f"I would use a {problem['topic']} approach to solve this.",
        f"My solution iterates through the input and tracks state.",
        f"I think we can optimize this using the key insight in {problem['topic']}.",
    ]
    answer = np.random.choice(answer_templates)
    return answer, score

# ── Single session ─────────────────────────────────────────────────────────────
def run_session(session_num: int, n_problems: int = 3, use_gpt: bool = True) -> dict:
    agent = DSACoachAgent(use_gpt=use_gpt)
    session_scores = []

    for _ in range(n_problems):
        problem = agent.present_problem()
        answer, score = simulate_student_answer(problem, session_num)

        if use_gpt:
            score = agent.evaluate_answer(answer)
            coaching = agent.get_coaching_response(answer, score)
            print(f"  [{problem['title']}] score={score:.2f} | {coaching[:80]}...")
        else:
            agent.get_coaching_response(answer, score)

        agent.log_problem_result(score)
        session_scores.append(score)

    stats = agent.end_session()
    stats["session_num"] = session_num
    stats["session_scores"] = session_scores
    return stats

# ── Plot learning curves ───────────────────────────────────────────────────────
def plot_results(all_stats: list):
    sessions = [s["session_num"] for s in all_stats]
    avg_scores = [s["avg_score"] for s in all_stats]
    avg_returns = [s["avg_return"] for s in all_stats]

    def smooth(y, w=3):
        return np.convolve(y, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("DSA Interview Coach — RL Learning Curves", fontsize=14, fontweight='bold')

    axes[0].plot(sessions, avg_scores, alpha=0.4, color='steelblue', label='Raw')
    if len(avg_scores) >= 3:
        axes[0].plot(sessions[1:-1], smooth(avg_scores), color='steelblue',
                     linewidth=2, label='Smoothed')
    axes[0].set_title("Avg Student Score per Session")
    axes[0].set_xlabel("Session")
    axes[0].set_ylabel("Score (0-1)")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].plot(sessions, avg_returns, alpha=0.4, color='coral', label='Raw')
    if len(avg_returns) >= 3:
        axes[1].plot(sessions[1:-1], smooth(avg_returns), color='coral',
                     linewidth=2, label='Smoothed')
    axes[1].set_title("REINFORCE Avg Return per Session")
    axes[1].set_xlabel("Session")
    axes[1].set_ylabel("Return")
    axes[1].legend()

    last_stats = all_stats[-1]
    counts = last_stats["bandit_stats"]["counts"]
    titles = [p["title"][:12] for p in PROBLEMS]
    axes[2].bar(range(len(counts)), counts, color='mediumseagreen')
    axes[2].set_title("UCB Problem Selection Counts")
    axes[2].set_xlabel("Problem")
    axes[2].set_ylabel("Times Selected")
    axes[2].set_xticks(range(len(titles)))
    axes[2].set_xticklabels(titles, rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=150, bbox_inches='tight')
    print("\nSaved: results/learning_curves.png")
    plt.show()

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_SESSIONS = 20
    USE_GPT = True   # set False for fast simulation without API calls

    print(f"Running {N_SESSIONS} sessions (use_gpt={USE_GPT})...\n")
    all_stats = []

    for i in range(N_SESSIONS):
        print(f"Session {i+1}/{N_SESSIONS}")
        stats = run_session(session_num=i+1, n_problems=3, use_gpt=USE_GPT)
        all_stats.append(stats)
        print(f"  -> avg_score={stats['avg_score']:.2f}, avg_return={stats['avg_return']:.2f}\n")

    with open("results/stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("Saved: results/stats.json")

    plot_results(all_stats)
    print("\nDone.")