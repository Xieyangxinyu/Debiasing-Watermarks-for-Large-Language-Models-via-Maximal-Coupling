import torch
import glob
import os
import numpy as np
from typing import Dict
from collections import defaultdict
import pandas as pd
import json

def check_repetitions(context_history: torch.Tensor) -> torch.Tensor:
    """
    Check for repeated seeds in context history using cumulative sum method.
    
    Args:
        context_history: [batch_size, seq_length] tensor of context seeds
        
    Returns:
        [batch_size, seq_length] boolean tensor where True indicates repeated seeds
    """
    is_repeated = torch.zeros_like(context_history, dtype=torch.bool)
    
    # For each unique seed column, check for repetitions
    for col_idx in range(context_history.shape[1]):
        col = context_history[:, col_idx]
        
        is_repeated_context = (
            context_history[:, :col_idx] == col.unsqueeze(1)
        ).any(dim=1, keepdim=True)

        is_repeated[:, col_idx] = is_repeated_context.squeeze()
    
    return is_repeated

def analyze_repetitions(cache_dir: str) -> Dict[str, float]:
    """
    Analyze repetitions from cache files, efficiently identifying repeated positions.
    """
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "cache_*.pt")))
    
    prompt_repeated_counts = defaultdict(int)
    prompt_total_counts = defaultdict(int)
    
    all_repeated_count = 0
    all_total_count = 0
    
    for cache_idx, cache_file in enumerate(cache_files):
        cache_data = torch.load(cache_file)
        
        non_decoded_mask = cache_data['non_decoded_mask']  # [batch_size, seq_length]
        context_history = cache_data['context_masking_state']  # [batch_size, seq_length]
        
        batch_size, seq_length = context_history.shape
        
        # Create mask for valid positions (not in prompt)
        valid_positions = ~non_decoded_mask
        
        # Check for repetitions in context history
        is_repeated = check_repetitions(context_history)
        
        # Create mask for repeated positions
        repeated_mask = is_repeated & valid_positions
        
        # Process each prompt
        for prompt_idx in range(batch_size):
            prompt_key = f"prompt_{cache_idx}_{prompt_idx}"
            
            # Count repeated and total valid positions
            repeated_count = repeated_mask[prompt_idx].sum().item()
            total_count = valid_positions[prompt_idx].sum().item()
            
            prompt_repeated_counts[prompt_key] += repeated_count
            prompt_total_counts[prompt_key] += total_count
            
            all_repeated_count += repeated_count
            all_total_count += total_count
    
    # Compute statistics
    per_prompt_repetition_rates = []
    
    for prompt_key in prompt_total_counts.keys():
        if prompt_total_counts[prompt_key] > 0:
            repetition_rate = prompt_repeated_counts[prompt_key] / prompt_total_counts[prompt_key]
            per_prompt_repetition_rates.append(repetition_rate)
    
    stats = {
        'overall_repetition_rate': all_repeated_count / all_total_count if all_total_count > 0 else 0,
        'per_prompt_mean_repetition_rate': np.mean(per_prompt_repetition_rates) if per_prompt_repetition_rates else 0,
        'per_prompt_std_repetition_rate': np.std(per_prompt_repetition_rates) if per_prompt_repetition_rates else 0,
        'total_repeated_tokens': all_repeated_count,
        'total_tokens': all_total_count,
        'num_prompts': len(prompt_total_counts),
    }
    
    return stats

def run_analysis(cache_dir: str):
    """Run repetition analysis for a single configuration."""
    # Compute repetition statistics
    stats = analyze_repetitions(cache_dir)
    
    # Collect per-prompt repetition rates
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "cache_*.pt")))
    prompt_repeated_counts = defaultdict(int)
    prompt_total_counts = defaultdict(int)
    
    for cache_idx, cache_file in enumerate(cache_files):
        cache_data = torch.load(cache_file)
        non_decoded_mask = cache_data['non_decoded_mask']
        context_history = cache_data['context_masking_state']
        
        valid_positions = ~non_decoded_mask
        is_repeated = check_repetitions(context_history)
        repeated_mask = is_repeated & valid_positions
        
        batch_size = context_history.shape[0]
        for prompt_idx in range(batch_size):
            prompt_key = f"prompt_{cache_idx}_{prompt_idx}"
            prompt_repeated_counts[prompt_key] += repeated_mask[prompt_idx].sum().item()
            prompt_total_counts[prompt_key] += valid_positions[prompt_idx].sum().item()
    
    repetition_rates = [
        prompt_repeated_counts[key] / prompt_total_counts[key] 
        for key in prompt_total_counts.keys() 
        if prompt_total_counts[key] > 0
    ]
    
    # Print statistics
    with open(f"{cache_dir}/repetition_stats.txt", "w") as f:
        f.write("Repetition Analysis:\n")
        f.write(f"  Overall Repetition Rate: {stats['overall_repetition_rate']:.4f}\n")
        f.write(f"  Per-prompt Mean Repetition Rate: {stats['per_prompt_mean_repetition_rate']:.4f}\n")
        f.write(f"  Per-prompt Std Repetition Rate: {stats['per_prompt_std_repetition_rate']:.4f}\n")
        f.write(f"  Total Repeated Tokens: {stats['total_repeated_tokens']}\n")
        f.write(f"  Total Tokens: {stats['total_tokens']}\n")
        f.write(f"  Number of Prompts: {stats['num_prompts']}\n")
    
    return stats, repetition_rates

if __name__ == "__main__":
    # Define parameter ranges
    ngrams = [2, 4]
    datasets = ["finance_qa", "longform_qa"]
    models = ["phi", "llama"]
    methods = ["coupling", "openai", "maryland", "dipmark"]
    
    # Storage for results
    all_stats = {}
    
    # Iterate through all configurations sequentially
    for ngram in ngrams:
        for data in datasets:
            for model in models:
                for method in methods:
                    cache_dir = f"output/{data}/{model}/{method}/ngram_{ngram}"
                    
                    # Skip if directory doesn't exist
                    if not os.path.exists(cache_dir):
                        print(f"Directory does not exist: {cache_dir}")
                        continue
                    
                    try:
                        
                        stats, _ = run_analysis(
                            cache_dir
                        )
                        
                        # Add configuration information to stats
                        stats['config'] = {
                            'ngram': ngram,
                            'dataset': data,
                            'model': model,
                            'method': method
                        }
                        
                        all_stats[cache_dir] = stats
                        
                        # Print results
                        print(f"\nResults for {cache_dir}:")
                        print(f"  Overall Repetition Rate: {stats['overall_repetition_rate']:.4f}")
                        print(f"  Per-prompt Mean Repetition Rate: {stats['per_prompt_mean_repetition_rate']:.4f}")
                        print(f"  Total Repeated Tokens: {stats['total_repeated_tokens']}")
                        print(f"  Total Tokens: {stats['total_tokens']}")
                        
                    except Exception as e:
                        print(f"Error processing {cache_dir}: {str(e)}")
    
    # Create summary DataFrame
    rows = []
    for cache_dir, stats in all_stats.items():
        config = stats['config']
        row = {
            'ngram': config['ngram'],
            'dataset': config['dataset'],
            'model': config['model'],
            'method': config['method'],
            'overall_repetition_rate': stats['overall_repetition_rate'],
            'per_prompt_mean_repetition_rate': stats['per_prompt_mean_repetition_rate'],
            'per_prompt_std_repetition_rate': stats['per_prompt_std_repetition_rate'],
            'total_repeated_tokens': stats['total_repeated_tokens'],
            'total_tokens': stats['total_tokens'],
            'num_prompts': stats['num_prompts']
        }
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Save results
    os.makedirs("analysis_results", exist_ok=True)
    
    # Save raw stats
    with open("analysis_results/combined_repetition_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    
    # Save summary DataFrame
    summary_df.to_csv("analysis_results/repetition_summary.csv", index=False)