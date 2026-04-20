# src/policy_gradient.py
# REINFORCE Policy Gradient
# Learns optimal coaching actions given the student's current state

import numpy as np

# Coaching actions the agent can take
ACTIONS = [
    "give_hint",          # 0: provide a hint
    "ask_followup",       # 1: ask a follow-up question
    "increase_difficulty",# 2: move to a harder problem
    "decrease_difficulty",# 3: move to an easier problem
    "explain_solution",   # 4: walk through the solution
    "encourage",          # 5: motivational nudge, try again
]

class REINFORCECoach:
    """
    REINFORCE policy gradient for coaching action selection.
    State: [avg_score, streak, difficulty, topic_encoded, attempts]
    Policy: softmax over linear weights -> action probabilities
    """

    def __init__(self, n_actions: int = 6, state_dim: int = 5, lr: float = 0.01):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.lr = lr

        # Linear policy weights: state_dim -> n_actions
        self.weights = np.zeros((state_dim, n_actions))

        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        self.episode_history = []  # for analysis

    def encode_state(self, avg_score: float, streak: int,
                     difficulty: int, topic_id: int, attempts: int) -> np.ndarray:
        """Normalize state into a fixed-size vector."""
        return np.array([
            avg_score,                  # 0.0 - 1.0
            min(streak, 10) / 10.0,     # normalized streak
            (difficulty - 1) / 2.0,     # 0.0, 0.5, 1.0
            topic_id / 8.0,             # normalized topic
            min(attempts, 5) / 5.0      # normalized attempts
        ])

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)  # numerical stability
        exp = np.exp(logits)
        return exp / exp.sum()

    def select_action(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        """Sample action from policy distribution."""
        logits = state @ self.weights
        probs = self.softmax(logits)
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs

    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """Store a step for end-of-episode update."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def compute_returns(self, gamma: float = 0.95) -> np.ndarray:
        """Compute discounted returns G_t for each timestep."""
        T = len(self.episode_rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = self.episode_rewards[t] + gamma * G
            returns[t] = G
        # Normalize for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, gamma: float = 0.95):
        """REINFORCE gradient update at end of episode."""
        if not self.episode_states:
            return 0.0

        returns = self.compute_returns(gamma)
        total_loss = 0.0

        for t in range(len(self.episode_states)):
            state = self.episode_states[t]
            action = self.episode_actions[t]
            G = returns[t]

            logits = state @ self.weights
            probs = self.softmax(logits)

            # Policy gradient: grad = G * (1 - p(a)) for chosen action
            grad = np.outer(state, -probs)
            grad[:, action] += state * G

            self.weights += self.lr * grad
            total_loss += -G * np.log(probs[action] + 1e-8)

        avg_return = np.mean(self.episode_rewards)
        self.episode_history.append({
            "avg_return": avg_return,
            "n_steps": len(self.episode_states)
        })

        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        return avg_return

    def get_action_name(self, action_id: int) -> str:
        return ACTIONS[action_id]

    def get_policy_stats(self) -> dict:
        return {
            "weights_norm": float(np.linalg.norm(self.weights)),
            "episode_history": self.episode_history
        }