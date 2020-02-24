import numpy as np

import ProblemModel


def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
This class represents a greedy benchmark that picks the K arms with highest means.
"""


class Benchmark:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, budget):
        self.num_rounds = problem_model.num_rounds
        self.budget = budget
        self.problem_model = problem_model

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)

        for t in range(1, self.num_rounds + 1):
            available_arms = self.problem_model.get_available_arms(t)
            available_arms.sort(key=lambda x: x.true_mean)
            slate = available_arms[-self.budget:]
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

        return total_reward_arr, regret_arr
