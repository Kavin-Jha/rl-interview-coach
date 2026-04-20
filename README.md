# DSA Interview Coach — Reinforcement Learning Agent

An agentic AI system that learns to optimally coach students through Data Structures & Algorithms interview preparation using two reinforcement learning approaches.

## Overview

The system combines:
- **UCB Contextual Bandit** — learns which problems to present next based on estimated learning value
- **REINFORCE Policy Gradient** — learns optimal coaching strategies (hints, follow-ups, difficulty adjustment)
- **GPT-4o-mini** — executes coaching actions as natural language responses

## Project Structure

```
rl_interview_coach/
├── src/
│   ├── problems.py        # DSA problem bank (12 problems, 3 difficulty levels)
│   ├── bandit.py          # UCB1 bandit for problem selection
│   ├── policy_gradient.py # REINFORCE policy gradient for coaching actions
│   └── agent.py           # Main agent orchestrating both RL components
├── results/
│   ├── learning_curves.png
│   └── stats.json
├── main.py                # Simulation runner
└── README.md
```

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install openai numpy matplotlib pandas python-dotenv
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

## Running

Full simulation (20 sessions, GPT-powered):
```bash
python3 main.py
```

Fast dry run (no API calls):
```bash
# In main.py, set USE_GPT = False
python3 main.py
```

## Reinforcement Learning Design

### 1. UCB Contextual Bandit (Problem Selection)
Each of the 12 DSA problems is treated as a bandit arm. The UCB1 algorithm balances:
- **Exploitation**: showing problems with historically high learning rewards
- **Exploration**: trying problems not yet seen enough

Reward signal: student score + first-attempt bonus − hint penalty

### 2. REINFORCE Policy Gradient (Coaching Strategy)
The coach learns a policy over 6 actions:
- `give_hint` — provide a targeted hint
- `ask_followup` — probe their reasoning
- `increase_difficulty` — advance to harder problems
- `decrease_difficulty` — step back to easier problems
- `explain_solution` — walk through the approach
- `encourage` — motivate and retry

State vector: `[avg_score, streak, difficulty, topic_id, attempts]`

Policy update uses discounted returns (γ=0.95) with normalized advantages for training stability.

## Results

After 20 sessions:
- Student scores trend upward (~0.4 → ~0.6)
- REINFORCE policy converges to stable return range
- UCB explores all problems before exploiting high-reward ones

See `results/learning_curves.png` for full learning curves.