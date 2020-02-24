import numpy as np

import ProblemModel
import random_algo


def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
This class represents the CC-MAB algorithm.
"""


class CCMAB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, budget, context_dim):  # Assumes a 1 x 1 x ... x 1 context space
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        self.hT = np.ceil(self.num_rounds ** (1 / (3 + context_dim)))
        self.cube_length = 1 / self.hT
        self.budget = budget
        self.problem_model = problem_model

    def get_hypercube_of_context(self, context):
        return tuple((context / self.cube_length).astype(int))

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        hypercube_played_counter_dict = {}
        avg_reward_dict = {}  # maps hypercube to avg reward

        for t in range(1, self.num_rounds + 1):
            arrived_cube_arms_dict = {}
            available_arms = self.problem_model.get_available_arms(t)

            # Hypercubes that the arrived arms belong to
            arrived_cube_set = set()
            for available_arm in available_arms:
                hypercube = self.get_hypercube_of_context(available_arm.context)
                if hypercube not in arrived_cube_arms_dict:
                    arrived_cube_arms_dict[hypercube] = list()
                arrived_cube_arms_dict[hypercube].append(available_arm)
                arrived_cube_set.add(hypercube)

            # Identify underexplored hypercubes
            underexplored_arm_set = set()
            for cube in arrived_cube_set:
                if hypercube_played_counter_dict.get(cube, 0) <= t ** (2 / (3 + self.context_dim)) * np.log(t):
                    underexplored_arm_set.update(arrived_cube_arms_dict[cube])

            # Play arms
            if len(underexplored_arm_set) >= self.budget:
                slate = random_algo.sample(underexplored_arm_set, self.budget)
            else:
                slate = []
                slate.extend(underexplored_arm_set)
                not_chosen_arms = list(set(available_arms) - underexplored_arm_set)
                i = 0
                conf_list = np.empty(len(not_chosen_arms))
                for arm in not_chosen_arms:
                    conf_list[i] = avg_reward_dict.get(self.get_hypercube_of_context(arm.context), 0)
                    i += 1
                arm_indices = self.problem_model.oracle(self.budget - len(slate), conf_list)
                for index in arm_indices:
                    selected_arm = not_chosen_arms[index]
                    slate.append(selected_arm)

            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Update the counters
            for reward in rewards:
                cube_with_context = self.get_hypercube_of_context(reward.context)
                new_counter = hypercube_played_counter_dict[cube_with_context] = hypercube_played_counter_dict.get(
                    cube_with_context, 0) + 1
                avg_reward_dict[cube_with_context] = (avg_reward_dict.get(cube_with_context, 0) * (
                        new_counter - 1) + reward.quality) / new_counter

        return total_reward_arr, regret_arr
