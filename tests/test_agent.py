# tests/test_agent.py
# Unit tests for the DSA Interview Coach RL Agent

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from src.bandit import UCBBandit
from src.policy_gradient import REINFORCECoach, ACTIONS
from src.problems import PROBLEMS, get_problem_by_id, get_all_problem_ids
from src.agent import DSACoachAgent


class TestUCBBandit(unittest.TestCase):

    def setUp(self):
        self.bandit = UCBBandit(n_problems=12)

    def test_initial_selection_is_unseen(self):
        pid = self.bandit.select_problem()
        self.assertEqual(self.bandit.counts[pid], 0)

    def test_update_increments_count(self):
        pid = self.bandit.select_problem()
        self.bandit.update(pid, 0.8)
        self.assertEqual(self.bandit.counts[pid], 1)

    def test_q_value_updated_correctly(self):
        self.bandit.update(0, 0.8)
        self.assertAlmostEqual(self.bandit.q_values[0], 0.8, places=5)

    def test_q_value_incremental_mean(self):
        self.bandit.update(0, 0.6)
        self.bandit.update(0, 1.0)
        self.assertAlmostEqual(self.bandit.q_values[0], 0.8, places=5)

    def test_excluded_ids_not_selected(self):
        excluded = list(range(11))
        pid = self.bandit.select_problem(excluded_ids=excluded)
        self.assertEqual(pid, 11)

    def test_history_tracks_updates(self):
        self.bandit.update(2, 0.7)
        self.bandit.update(5, 0.4)
        self.assertEqual(len(self.bandit.history), 2)
        self.assertEqual(self.bandit.history[0], (2, 0.7))

    def test_total_steps_increments(self):
        self.bandit.update(0, 0.5)
        self.bandit.update(1, 0.5)
        self.assertEqual(self.bandit.total_steps, 2)

    def test_ucb_prefers_higher_reward(self):
        # Give problem 0 high reward many times, problem 1 low reward
        for _ in range(10):
            self.bandit.update(0, 1.0)
            self.bandit.update(1, 0.0)
        # After many updates, UCB should prefer problem 0
        # (exclude all others to force comparison)
        excluded = list(range(2, 12))
        scores = []
        for _ in range(20):
            pid = self.bandit.select_problem(excluded_ids=excluded)
            scores.append(pid)
        self.assertGreater(scores.count(0), scores.count(1))


class TestREINFORCECoach(unittest.TestCase):

    def setUp(self):
        self.coach = REINFORCECoach()

    def test_encode_state_shape(self):
        state = self.coach.encode_state(0.5, 2, 2, 3, 1)
        self.assertEqual(state.shape, (5,))

    def test_encode_state_normalized(self):
        state = self.coach.encode_state(1.0, 10, 3, 8, 5)
        for v in state:
            self.assertLessEqual(v, 1.0)
            self.assertGreaterEqual(v, 0.0)

    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 0.5, -1.0, 3.0, 0.0])
        probs = self.coach.softmax(logits)
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_select_action_valid(self):
        state = self.coach.encode_state(0.5, 1, 2, 3, 0)
        action, probs = self.coach.select_action(state)
        self.assertIn(action, range(6))
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_store_and_update_clears_episode(self):
        state = self.coach.encode_state(0.5, 1, 2, 3, 0)
        action, _ = self.coach.select_action(state)
        self.coach.store_transition(state, action, 0.7)
        self.assertEqual(len(self.coach.episode_states), 1)
        self.coach.update()
        self.assertEqual(len(self.coach.episode_states), 0)

    def test_update_returns_float(self):
        state = self.coach.encode_state(0.5, 1, 2, 3, 0)
        action, _ = self.coach.select_action(state)
        self.coach.store_transition(state, action, 0.8)
        result = self.coach.update()
        self.assertIsInstance(result, float)

    def test_weights_change_after_update(self):
        weights_before = self.coach.weights.copy()
        state = self.coach.encode_state(0.6, 2, 1, 4, 1)
        action, _ = self.coach.select_action(state)
        self.coach.store_transition(state, action, 1.0)
        self.coach.update()
        self.assertFalse(np.allclose(self.coach.weights, weights_before))

    def test_action_names_valid(self):
        for i in range(6):
            name = self.coach.get_action_name(i)
            self.assertIn(name, ACTIONS)


class TestProblems(unittest.TestCase):

    def test_problem_count(self):
        self.assertEqual(len(PROBLEMS), 12)

    def test_all_problems_have_required_fields(self):
        for p in PROBLEMS:
            for field in ["id", "title", "topic", "difficulty", "description", "hint", "solution_approach"]:
                self.assertIn(field, p)

    def test_difficulty_range(self):
        for p in PROBLEMS:
            self.assertIn(p["difficulty"], [1, 2, 3])

    def test_get_problem_by_id(self):
        p = get_problem_by_id(0)
        self.assertEqual(p["title"], "Two Sum")

    def test_get_problem_by_invalid_id(self):
        p = get_problem_by_id(999)
        self.assertIsNone(p)

    def test_all_problem_ids_unique(self):
        ids = get_all_problem_ids()
        self.assertEqual(len(ids), len(set(ids)))


class TestDSACoachAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DSACoachAgent(use_gpt=False)

    def test_present_problem_returns_dict(self):
        p = self.agent.present_problem()
        self.assertIsInstance(p, dict)
        self.assertIn("title", p)

    def test_present_problem_no_repeats_in_session(self):
        seen = set()
        for _ in range(12):
            p = self.agent.present_problem()
            self.assertNotIn(p["id"], seen)
            self.agent.log_problem_result(0.5)
            seen.add(p["id"])

    def test_evaluate_answer_returns_float_without_gpt(self):
        self.agent.present_problem()
        score = self.agent.evaluate_answer("I would use a hashmap.")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_coaching_response_contains_action(self):
        self.agent.present_problem()
        response = self.agent.get_coaching_response("my answer", 0.5)
        self.assertIn("[Coach action:", response)

    def test_end_session_returns_stats(self):
        self.agent.present_problem()
        self.agent.get_coaching_response("answer", 0.7)
        self.agent.log_problem_result(0.7)
        stats = self.agent.end_session()
        self.assertIn("avg_score", stats)
        self.assertIn("bandit_stats", stats)
        self.assertIn("policy_stats", stats)

    def test_streak_increments_on_high_score(self):
        self.agent.present_problem()
        self.agent.get_coaching_response("answer", 0.9)
        self.assertEqual(self.agent.streak, 1)

    def test_streak_resets_on_low_score(self):
        self.agent.present_problem()
        self.agent.get_coaching_response("answer", 0.9)
        self.agent.get_coaching_response("answer", 0.3)
        self.assertEqual(self.agent.streak, 0)

    def test_compute_reward_clipped(self):
        reward = self.agent._compute_reward(1.0, 1, False)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)