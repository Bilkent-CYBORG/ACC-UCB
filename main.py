import multiprocessing
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from CC_MAB import CCMAB
from ACC_UCB import ACCUCB
from Hypercube import Hypercube
from benchmark_algo import Benchmark
from gowalla_problem_model import GowallaProblemModel
from random_algo import Random

"""
This python script is responsible for running ACC-UCB, CC-MAB, Random, and benchmark on the Gowalla dataset
for a given number of times.
!!! Note that when the parameter below is set to True, the script will use the arm-pairs that were used to produce the 
figures in the paper. When set to False, the script will generate arm-pairs (i.e., number of arms, worker batteries, etc)
and run the simulations on them. Therefore, to reproduce the results in the paper, it must be set to True.!!!
"""
use_generated_workers_in_paper = True

sns.set(style='whitegrid')
num_threads_to_use = -1  # number of threads to run the simulation on. When set to -1, will run on all available threads
use_saved_data = True  # when True, the script simply plots the data of the most recently ran simulation, if available
# this means that no simulations are run when True.

available_arms_mean = 50
num_times_to_run = 10
num_rounds = 50000
num_std_to_show = 5
budgets = [2, 4]
line_style_dict = {2: '-',
                   4: ':'}

v1 = np.sqrt(5)
v2 = 1
rho = 0.5
N = 2  # changing this HAS NO EFFECT and the ACC-UCB class will not work when N != 2
root_context = Hypercube(1, np.array([0.5, 0.5]))  # this is called x_{0,1} in the paper


def run_one_try(problem_model, num_run, budget):
    random_algo = Random(problem_model, budget)
    bench_algo = Benchmark(problem_model, budget)
    cc_mab_algo = CCMAB(problem_model, budget, root_context.get_dimension())
    cc_ucb_algo = ACCUCB(problem_model, v1, v2, N, rho, budget, root_context)

    ucb_reward, ucb_regret = cc_ucb_algo.run_algorithm()
    bench_reward, bench_regret = bench_algo.run_algorithm()
    random_reward, random_regret = random_algo.run_algorithm()
    mab_reward, mab_regret = cc_mab_algo.run_algorithm()
    print("Run done: " + str(num_run))

    return {'bench_reward': bench_reward,
            'random_reward': random_reward,
            'random_regret': random_regret,
            'ucb_reward': ucb_reward,
            'mab_reward': mab_reward,
            'ucb_regret': ucb_regret,
            'mab_regret': mab_regret}


if __name__ == '__main__':
    ucb_reward_runs_arr = np.zeros((num_times_to_run, num_rounds))
    mab_reward_runs_arr = np.zeros((num_times_to_run, num_rounds))
    random_reward_runs_arr = np.zeros((num_times_to_run, num_rounds))
    bench_reward_runs_arr = np.zeros((num_times_to_run, num_rounds))

    ucb_regret_runs_arr = np.zeros((num_times_to_run, num_rounds))
    mab_regret_runs_arr = np.zeros((num_times_to_run, num_rounds))
    random_regret_runs_arr = np.zeros((num_times_to_run, num_rounds))
    if not use_saved_data:
        problem_model = GowallaProblemModel(num_rounds, available_arms_mean, max(budgets),
                                            use_generated_workers_in_paper)
        if num_threads_to_use == -1:
            num_threads_to_use = int(multiprocessing.cpu_count())
        print("Running on {thread_count} threads".format(thread_count=num_threads_to_use))
        for budget in budgets:
            print("Doing budget {budget}...".format(budget=budget))
            parallel_results = Parallel(n_jobs=num_threads_to_use)(
                delayed(run_one_try)(problem_model, i, budget) for i in tqdm(range(num_times_to_run)))

            with open('parallel_results_budget_{budget}'.format(budget=budget), 'wb') as output:
                pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)

    for budget in budgets:
        with open('parallel_results_budget_{budget}'.format(budget=budget), 'rb') as input_file:
            parallel_results = pickle.load(input_file)

        i = 0
        # Load the ucb, mab, random, and bench rewards and regrets
        for entry in parallel_results:
            ucb_reward_runs_arr[i] = pd.Series(entry['ucb_reward']).expanding().mean().values
            mab_reward_runs_arr[i] = pd.Series(entry['mab_reward']).expanding().mean().values
            random_reward_runs_arr[i] = pd.Series(entry['random_reward']).expanding().mean().values
            bench_reward_runs_arr[i] = pd.Series(entry['bench_reward']).expanding().mean().values

            ucb_regret_runs_arr[i] = np.cumsum(entry['ucb_regret'])
            mab_regret_runs_arr[i] = np.cumsum(entry['mab_regret'])
            random_regret_runs_arr[i] = np.cumsum(entry['random_regret'])
            i += 1

        # Find the mean and std of the regrets and rewards
        ucb_avg_reward = np.mean(ucb_reward_runs_arr, axis=0)
        mab_avg_reward = np.mean(mab_reward_runs_arr, axis=0)
        random_avg_reward = np.mean(random_reward_runs_arr, axis=0)
        bench_avg_reward = np.mean(bench_reward_runs_arr, axis=0)

        ucb_std_reward = np.std(ucb_reward_runs_arr, axis=0)
        mab_std_reward = np.std(mab_reward_runs_arr, axis=0)
        random_std_reward = np.std(random_reward_runs_arr, axis=0)
        bench_std_reward = np.std(bench_reward_runs_arr, axis=0)

        ucb_avg_regret = np.mean(ucb_regret_runs_arr, axis=0)
        mab_avg_regret = np.mean(mab_regret_runs_arr, axis=0)
        random_avg_regret = np.mean(random_regret_runs_arr, axis=0)

        ucb_std_regret = np.std(ucb_regret_runs_arr, axis=0)
        mab_std_regret = np.std(mab_regret_runs_arr, axis=0)
        random_std_regret = np.std(random_regret_runs_arr, axis=0)

        # PLOT CUMULATIVE REGRET
        # Only show a few error bars
        for i in range(len(mab_std_regret)):
            if i == 0 or i % int(num_rounds / num_std_to_show) != 0 and i != len(mab_std_regret) - 1:
                ucb_std_regret[i] = mab_std_regret[i] = random_std_regret[i] = None

        plt.figure(1)
        plt.errorbar(range(1, num_rounds + 1), ucb_avg_regret, yerr=ucb_std_regret,
                     label="ACC-UCB with $K = {budget}$".format(budget=budget), capsize=2, color='r',
                     linestyle=line_style_dict[budget], linewidth=2)
        plt.errorbar(range(1, num_rounds + 1), mab_avg_regret, yerr=mab_std_regret,
                     label="CC-MAB with $K = {budget}$".format(budget=budget), capsize=2, color='g',
                     linestyle=line_style_dict[budget], linewidth=2)
        plt.errorbar(range(1, num_rounds + 1), random_avg_regret, yerr=random_std_regret,
                     label="Random with $K = {budget}$".format(budget=budget), capsize=2, color='b',
                     linestyle=line_style_dict[budget], linewidth=2)

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend()
        plt.xlabel("Arriving task $(t)$")
        plt.ylabel("Cumulative regret up to $t$")
        plt.tight_layout()
        plt.savefig("cum_regret.pdf", bbox_inches='tight', pad_inches=0.01)

        # PLOT AVERAGE REWARD
        # Only show a few error bars
        for i in range(len(mab_std_regret)):
            if i == 0 or i % int(num_rounds / num_std_to_show) != 0 and i != len(mab_std_regret) - 1:
                ucb_std_reward[i] = mab_std_reward[i] = random_std_reward[i] = bench_std_reward[i] = None
        plt.figure(2)
        plt.errorbar(range(1, num_rounds + 1), ucb_avg_reward, yerr=ucb_std_reward,
                     label="ACC-UCB with $K = {budget}$".format(budget=budget), capsize=2, color='r',
                     linestyle=line_style_dict[budget], linewidth=2)
        plt.errorbar(range(1, num_rounds + 1), mab_avg_reward, yerr=mab_std_reward,
                     label="CC-MAB with $K = {budget}$".format(budget=budget), capsize=2, color='g',
                     linestyle=line_style_dict[budget], linewidth=2)
        plt.errorbar(range(1, num_rounds + 1), random_avg_reward, yerr=random_std_reward,
                     label="Random with $K = {budget}$".format(budget=budget), capsize=2, color='b',
                     linestyle=line_style_dict[budget], linewidth=2)

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend()
        plt.ylim(0.1, 1)  # We need to do this b/c otherwise the legend was not seen
        plt.xlabel("Arriving task $(t)$")
        plt.ylabel("Average task reward up to $t$")
        plt.tight_layout()
        plt.savefig("avg_reward.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()
