# ACC-UCB
Implementation of the Adaptive Contextual Combinatorial Upper Confidence Bound (ACC-UCB) algorithm for the contextual combinatorial volatile multi-armed bandit setting.

# How to run
The main script is the "main.py" file. Depending on the setting of the "use_generated_workers_in_paper" parameter, it will either run the simulations using the arm-pairs used in the paper simulation (saved in the "simulation_df" file), or it will generate the arm-pairs (i.e., workers) from the Gowalla dataset as explained in the paper, and then run the simulations. In other words, by setting the "use_generated_workers_in_paper" to True, you will get the results presented in the paper.

Note that if you set "use_generated_workers_in_paper" to False and want to run the simulations from scratch, then the Gowalla checkins txt file must be downloaded to the same directory as the script and saved as "Gowalla_totalCheckins.txt". Then, the "gowallaLoader.py" script MUST BE RUN before the "main.py" file. The checkins txt file can be downloaded from the following link:
https://snap.stanford.edu/data/loc-Gowalla.html.

Therefore, the "Gowalla_totalCheckins.txt" file is processed by the "gowallaLoader.py" script and saved as a DataFrame in a file called "gowalla_df". Then, the "gowalla_df" file is opened when running the simulations and from it, workers and tasks are generated. The generated arm-pairs are saved as a DataFrame in the "simulation_df" file and used in the simulations.


# On the implementation of ACC_UCB 
The implemented ACC-UCB class ONLY works when N=2 (i.e., binary tree of nodes/contexts). It will, however, work for any k-dimensional context, BUT v_1, v_2, \rho, and x_{0,1} must be rederived and updated in the "main.py" file.

# Specs of the PC used for the paper
A PC with the following specs was used for the simulations whose results/figures are presented in the paper:

CPU: Intel Core i7-7700 @ 3.60 GHz |
RAM: 32 GB |
OS: Ubuntu 18.10

# How to cite
Please cite this code as

A. Nika, S. Elahi and C. Tekin, "Contextual combinatorial volatile multi-armed bandit with adaptive discretization", to appear in 23rd International Conference on Artificial Intelligence and Statistics (AISTATS).
