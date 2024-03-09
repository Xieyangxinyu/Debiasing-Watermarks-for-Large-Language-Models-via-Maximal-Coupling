# Simulate the Higher Criticism null distribution

import numpy as np
from tqdm import tqdm
import pickle

# Set the random seed for reproducibility
np.random.seed(0)

# Set the number of simulations
n_sim = 2000

# Assuming n_sim and empirical_dists are defined
for n_sample in tqdm(range(1, 1028)):
    file_path = f'HC_simulate_data/{n_sample}.pkl'
    #if os.path.exists(file_path):
    #    continue

    empirical_dist = {}
    
    # Sample from uniform distribution
    P = np.random.uniform(0, 1, (n_sim, n_sample))
    #empirical_dist['P'] = P
    
    # Sort P along the last axis (samples within each simulation)
    P_sorted = np.sort(P, axis=1)
    
    # Compute t/n for each possible t value, considering broadcasting
    t_n = np.arange(1, n_sample + 1) / n_sample
    
    # Compute p_{(t)} - t/n for each p_{(t)} and corresponding t/n, again using broadcasting
    difference = t_n - P_sorted
    
    # Compute the denominator for the HC statistic
    denominator = np.sqrt(P_sorted * (1 - P_sorted))
    
    # Compute the HC+ statistic, avoiding division by zero and ensuring p_{(t)} > 1/n
    # Use np.where to handle p_{(t)} > 1/n condition
    valid_indices = P_sorted > 1/n_sample
    HC_plus_values = np.where(valid_indices, np.sqrt(n_sample) * difference / denominator, 0)
    
    # Compute the max of HC+ values across the valid t values for each simulation
    HC_plus = np.max(HC_plus_values, axis=1)
    
    empirical_dist['HC_plus'] = HC_plus
    
    # Calculate the critical value for the HC+ statistics, set alpha = 0.0001
    alpha = 0.01
    critical_value = np.percentile(HC_plus, (1 - alpha) * 100)
    
    empirical_dist['critical_value'] = critical_value

    pickle.dump(empirical_dist, open(file_path, 'wb'))