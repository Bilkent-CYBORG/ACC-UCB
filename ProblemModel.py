from abc import ABC

"""
This abstract class represents a problem model that the ACC-UCB algorithm will run on.
"""


class ProblemModel(ABC):

    def get_available_arms(self, t):
        pass

    def oracle(self, K, g_list):
        pass

    def play_arms(self, t, slate):
        pass

    def get_total_reward(self, rewards):
        pass
