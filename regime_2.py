'''
Author: Yangxinyu Xie
Date: 2024-11-07

All rights reserved.
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compare_and_save_statistics(p=0.5, q=0.2, sim=1000, m=10000):
    np.random.seed(42)

    # Sample from uniform distribution for null hypothesis
    P = np.random.uniform(0, 1, (sim, m))
    P.sort(axis=1)  # Sort each row for HC statistics
    Sums = np.sum(P, axis=1)  # Sum for Sum-based statistics

    # Calculate HC+ statistics for null hypothesis
    t = np.arange(1, m + 1).reshape(1, m) / m
    HC_plus = np.sqrt(m) * np.max(((t - P) / np.sqrt(P * (1 - P))) * (P > 1/m), axis=1)

    # Calculate critical values
    alpha = 0.01
    critical_value_HC = np.percentile(HC_plus, (1 - alpha) * 100)
    critical_value_Sum = np.percentile(Sums, alpha * 100)

    # Sample under alternative hypothesis
    P_mixture = np.random.uniform(0, 1, (sim, m))
    eps = m**(-p)
    samples = np.random.rand(sim, m)
    mask = samples < eps
    p_G = 1 - m ** (-q)
    P_mixture[mask] = np.random.uniform(0, p_G, np.count_nonzero(mask))
    P_mixture.sort(axis=1)  # Sort each row for HC statistics
    

    # Calculate HC+ statistics for alternative hypothesis
    HC_plus_alt = np.sqrt(m) * np.max(((t - P_mixture) / np.sqrt(P_mixture * (1 - P_mixture))) * (P_mixture > 1/m), axis=1)
    # Calculate sums for alternative hypothesis
    Sums_alt = np.sum(P_mixture, axis=1)

    # Compute rejection rates
    rejection_rate_HC = np.mean(HC_plus_alt > critical_value_HC)
    rejection_rate_Sum = np.mean(Sums_alt < critical_value_Sum)

    
    # Plotting both statistics in one figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # HC+ Statistics plot
    axs[0].hist(HC_plus, bins=100, density=True, alpha=0.5, label='Null HC+')
    axs[0].hist(HC_plus_alt, bins=100, density=True, alpha=0.5, label='Alternative HC+')
    axs[0].axvline(critical_value_HC, color='r', linestyle='--', label='Critical Value HC+')
    axs[0].legend()
    axs[0].set_title(f'Higher Criticism + Statistics, Rejection Rate: {rejection_rate_HC:.3f}')

    # Sum Statistics plot
    axs[1].hist(Sums, bins=100, density=True, alpha=0.5, label='Null Sum')
    axs[1].hist(Sums_alt, bins=100, density=True, alpha=0.5, label='Alternative Sum')
    axs[1].axvline(critical_value_Sum, color='r', linestyle='--', label='Critical Value Sum')
    axs[1].legend()
    axs[1].set_title(f'Sums Statistics, Rejection Rate: {rejection_rate_Sum:.3f}')

    # Overall figure title
    plt.suptitle(f'Comparison of HC+ and Sum Statistics (p={p}, q={q}, m={m}, eps={eps:.3f}, p(G)={p_G:.3f})')

    # Save the figure
    filename = f'simulation/regime_2/statistics_comparison_p_{p}_q_{q}_m_{m}.png'
    plt.savefig(filename)
    plt.close()
    
    return rejection_rate_HC, rejection_rate_Sum

exp = [2, 3, 4, 5]


for (p,q) in tqdm([(0.1, 0.8), (0.2, 0.5), (0.2, 0.2), (0.2, 0.3), (0.1, 0.4), (0.3, 0.2), (0.2, 0.4), (0.2, 0.6), (0.6, 0.1), (0.3, 0.4), (0.4, 0.4)]):
    # Store the rejection rates for plotting
    rejection_rates_HC = []
    rejection_rates_Sum = []
    for m in tqdm([10**i for i in exp]):
        rr_HC, rr_Sum = compare_and_save_statistics(p=p, q=q, m=m)
        rejection_rates_HC.append(rr_HC)
        rejection_rates_Sum.append(rr_Sum)

    # Plotting the rejection rates
    plt.figure(figsize=(6, 6))
    plt.plot(exp, rejection_rates_HC, label='HC+ Rejection Rate', marker='o')
    plt.plot(exp, rejection_rates_Sum, label='Sum Rejection Rate', marker='s')
    plt.xlabel('Sample Size (log_10(m))')
    # make sure x-axis are all integers
    plt.xticks(exp)
    plt.ylabel('Rejection Rate')
    plt.ylim(0,1.1)
    plt.title(f'Rejection Rates for HC+ and Sum Statistics (p={p}, q={q})')
    plt.legend()
    plt.savefig(f'simulation/regime_2/rejection_rates_p_{p}_q_{q}.png')