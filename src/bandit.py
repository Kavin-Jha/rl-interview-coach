# src/bandit.py
# Upper Confidence Bound (UCB) Contextual Bandit
# Selects which DSA problem to present next based on estimated learning value

import numpy as np

class UCBBandit:
    """
    UCB1 Bandit that treats each problem as an arm.
    Balances exploitation (show problems with high learning reward)
    with exploration (try problems not seen enough yet).
    """

    def __init__(self, n_problems: int, c: float = 1.5):
        self.n_problems = n_problems
        self.c = c  # exploration coefficient

        self.counts = np.zeros(n_problems)       # how many times each problem shown
        self.rewards = np.zeros(n_problems)      # cumulative reward per problem
        self.q_values = np.zeros(n_problems)     # estimated mean reward per problem
        self.total_steps = 0
        self.history = []  # track (problem_id, reward) over time

    def select_problem(self, excluded_ids: list = None) -> int:
        """Select next problem using UCB1 formula."""
        excluded = set(excluded_ids or [])

        # Always try unseen problems first (counts == 0)
        for i in range(self.n_problems):
            if i not in excluded and self.counts[i] == 0:
                return i

        # UCB1: Q(a) + c * sqrt(ln(t) / N(a))
        ucb_values = np.full(self.n_problems, -np.inf)
        for i in range(self.n_problems):
            if i not in excluded:
                exploration_bonus = self.c * np.sqrt(
                    np.log(self.total_steps + 1) / (self.counts[i] + 1e-5)
                )
                ucb_values[i] = self.q_values[i] + exploration_bonus

        return int(np.argmax(ucb_values))

    def update(self, problem_id: int, reward: float):
        """Update estimates after observing a reward."""
        self.counts[problem_id] += 1
        self.total_steps += 1
        self.rewards[problem_id] += reward

        # Incremental mean update
        n = self.counts[problem_id]
        self.q_values[problem_id] += (reward - self.q_values[problem_id]) / n

        self.history.append((problem_id, reward))

    def get_stats(self) -> dict:
        """Return current bandit statistics."""
        return {
            "counts": self.counts.tolist(),
            "q_values": self.q_values.tolist(),
            "total_steps": self.total_steps,
            "history": self.history
        }