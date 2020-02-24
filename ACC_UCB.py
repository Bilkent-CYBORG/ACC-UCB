import math
from math import sqrt

import numpy as np

import ProblemModel
import random_algo
from UcbNode import UcbNode


def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
This class represents the ACC-UCB algorithm that is presented in the paper.
"""


class ACCUCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, v1, v2, N, rho, budget, initial_hypercube):
        self.initial_hypercube = initial_hypercube
        if N != 2:
            print('ACC-UCB ONLY works when N = 2')
            exit(1)
        self.N = N
        self.num_rounds = problem_model.num_rounds
        self.budget = budget
        self.rho = rho
        self.v2 = v2
        self.v1 = v1
        self.problem_model = problem_model

    def run_algorithm(self):
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        leaves = [UcbNode(None, 0, [self.initial_hypercube])]
        node_played_counter_dict = {}
        avg_reward_dict = {}

        for t in range(1, self.num_rounds + 1):
            available_arms = self.problem_model.get_available_arms(t)
            index_list = np.zeros(len(available_arms))
            i = 0

            # Check if only root node is available
            if len(leaves) == 1:
                arm_indices_to_play = random_algo.sample(range(len(available_arms)), self.budget)
            else:
                for available_arm in available_arms:
                    node = find_node_containing_context(available_arm.context, leaves)
                    index_list[i] = self.get_arm_index(node, node_played_counter_dict, avg_reward_dict)
                    i += 1

                arm_indices_to_play = self.problem_model.oracle(self.budget, index_list)

            selected_nodes = set()
            slate = []
            for index in arm_indices_to_play:
                selected_arm = available_arms[index]
                slate.append(selected_arm)
                selected_nodes.add(find_node_containing_context(selected_arm.context, leaves))
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Update the counters
            for reward in rewards:
                node_with_context = find_node_containing_context(reward.context, selected_nodes)
                new_counter = node_played_counter_dict[node_with_context] = node_played_counter_dict.get(
                    node_with_context, 0) + 1
                avg_reward_dict[node_with_context] = (avg_reward_dict.get(node_with_context, 0) * (
                            new_counter - 1) + reward.quality) / new_counter

            for selected_node in selected_nodes:
                # Split the node if needed
                if self.calc_confidence(
                        node_played_counter_dict[selected_node]) <= self.v1 * self.rho ** selected_node.h:
                    produced_nodes = selected_node.reproduce()
                    leaves.remove(selected_node)
                    leaves.extend(produced_nodes)
        return total_reward_arr, regret_arr

    def get_arm_index(self, node, node_played_counter_dict, avg_reward_dict):
        num_times_node_played = node_played_counter_dict.get(node, 0)
        avg_reward_of_node = avg_reward_dict.get(node, 0)
        num_times_parent_node_played = node_played_counter_dict.get(node.parent_node, 0)
        avg_reward_of_parent_node = avg_reward_dict.get(node.parent_node, 0)

        node_index = min(avg_reward_of_node + self.calc_confidence(num_times_node_played),
                         avg_reward_of_parent_node + self.calc_confidence(num_times_parent_node_played) +
                         self.v1 * self.rho ** (node.h - 1)) + self.v1 * self.rho ** node.h

        return node_index + self.N * self.v1 / self.v2 * self.v1 * self.rho ** node.h

    def calc_confidence(self, num_times_node_played):
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(self.num_rounds) / num_times_node_played)
