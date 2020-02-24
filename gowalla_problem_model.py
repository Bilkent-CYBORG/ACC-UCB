import pickle
import random

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import gowallaLoader
from Arm import Arm
from ProblemModel import ProblemModel
from Reward import Reward

"""
This file contains code for the Gowalla problem model. The GowallaProblemModel class had functions to 
provide available arms, play the arms, calculate regret, and etc.
"""

saved_df_name = 'simulation_df'  # file where the saved simulation-ready dataframe will be saved


def context_to_mean_fun(context):
    """
    context[0] = norm of distance between worker and task location
    context[1] = worker battery
    """
    return norm.pdf(context[0], loc=0, scale=1) * context[1] ** 2 / norm.pdf(0, loc=0, scale=1)


class GowallaProblemModel(ProblemModel):
    def __init__(self, num_rounds, available_arms_mean, budget, use_saved):
        self.num_rounds = num_rounds
        if not use_saved:
            self.df = self.initialize_df(available_arms_mean, budget)
            self.df.set_index('time', inplace=True)
            with open(saved_df_name, 'wb') as output:
                pickle.dump(self.df, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(saved_df_name, 'rb') as input_file:
                self.df = pickle.load(input_file)

    def get_available_arms(self, t):
        # Construct a list of Arm objects
        arm_list = []
        for _, row in self.df.loc[t].iterrows():
            arm_list.append(Arm(len(arm_list), row['context'], row['true_mean']))
        return arm_list

    def get_regret(self, t, budget, slate):
        df = self.df.loc[t]
        highest_means = df['true_mean'].nlargest(budget)
        algo_mean_prod = 1
        bench_mean_prod = 1
        for arm in slate:
            algo_mean_prod *= 1 - df.iloc[arm.unique_id]['true_mean']
        for mean in highest_means:
            bench_mean_prod *= 1 - mean

        return algo_mean_prod - bench_mean_prod

    def get_total_reward(self, rewards):
        reward_sum = 0
        for reward in rewards:
            reward_sum += reward.quality  # Total reward is lin sum
        if reward_sum >= 1:
            return 1
        return 0

    def play_arms(self, t, slate):
        reward_list = []
        df = self.df.loc[t]
        for arm in slate:
            quality = np.random.binomial(1, df.iloc[arm.unique_id]['true_mean'])
            reward_list.append(Reward(arm, quality))
        return reward_list

    def oracle(self, K, g_list):
        return np.argsort(g_list)[-K:]

    def initialize_df(self, available_arms_mean, budget):
        print("Generating workers...")
        with open(gowallaLoader.saved_df_filename, 'rb') as input_file:
            df = pickle.load(input_file)
        user_index_set = set()
        row_list = []
        for time in tqdm(range(1, self.num_rounds + 1)):
            # Define task
            task_location = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
            num_available_arms = np.random.poisson(available_arms_mean)
            while num_available_arms <= budget:
                num_available_arms = np.random.poisson(available_arms_mean)
            for i in range(num_available_arms):
                worker_battery = np.random.uniform(0, 1)

                # Randomly assign user location to worker
                index = random.randint(0, len(df) - 1)
                while index in user_index_set:
                    index = random.randint(0, len(df) - 1)
                user_index_set.add(index)
                user_row = df.iloc[index]
                worker_location = np.array([user_row['latitude'], user_row['longitude']])
                context = np.array([np.linalg.norm(worker_location - task_location) / np.sqrt(2), worker_battery])
                true_mean = context_to_mean_fun(context)
                row_list.append((time, context, true_mean))
        return pd.DataFrame(row_list, columns=['time', 'context', 'true_mean'])
