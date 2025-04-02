import torch
import glob
import os
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_cache_files(cache_dir: str) -> Tuple[Dict[str, float], List[float], List[float]]:
    """
    Analyze all cache files to compute rejection statistics per prompt and overall,
    and collect proposal probabilities for accepted and rejected tokens.
    
    Args:
        cache_dir (str): Directory containing cache_*.pt files
        
    Returns:
        Tuple[Dict[str, float], List[float], List[float]]: 
            - Dictionary containing rejection statistics
            - List of proposal probabilities for rejected tokens
            - List of proposal probabilities for accepted tokens
    """
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "cache_*.pt")))
    
    # Store rejections and probabilities per prompt
    prompt_rejections = defaultdict(list)
    prompt_total_tokens = defaultdict(list)
    
    all_rejected_probs = []
    all_accepted_probs = []
    
    for cache_idx, cache_file in enumerate(cache_files):
        cache_data = torch.load(cache_file)
        
        rejection_masks = cache_data['rejection_masks']
        proposal_probs = cache_data['proposal_probs']
        non_decoded_mask = cache_data['non_decoded_mask']
        # context_history = cache_data['context_masking_state']
        
        # Create effective mask (tokens that were actually decoded and not in context)
        effective_mask = ~non_decoded_mask
        
        # Process each prompt (row) separately
        for prompt_idx in range(rejection_masks.shape[0]):
            prompt_key = f"prompt_{cache_idx}_{prompt_idx}"
            
            # Get masks for this prompt
            prompt_effective_mask = effective_mask[prompt_idx]
            prompt_rejection_mask = rejection_masks[prompt_idx]
            prompt_probs = proposal_probs[prompt_idx]
            
            # Count effective tokens and rejections for this prompt
            effective_tokens = prompt_effective_mask.sum().item()
            effective_rejections = (prompt_rejection_mask & prompt_effective_mask).sum().item()
            
            prompt_rejections[prompt_key].append(effective_rejections)
            prompt_total_tokens[prompt_key].append(effective_tokens)
            
            # Collect probabilities for effectively rejected and accepted tokens
            effective_indices = torch.where(prompt_effective_mask)[0]
            for idx in effective_indices:
                prob = prompt_probs[idx].item()
                if prompt_rejection_mask[idx]:
                    all_rejected_probs.append(prob)
                else:
                    all_accepted_probs.append(prob)
    
    # Compute statistics
    per_prompt_rates = []
    for prompt_key in prompt_rejections.keys():
        total_rejections = sum(prompt_rejections[prompt_key])
        total_tokens = sum(prompt_total_tokens[prompt_key])
        if total_tokens > 0:  # Avoid division by zero
            per_prompt_rates.append(total_rejections / total_tokens)
    
    # Calculate overall statistics
    total_rejections = sum(sum(rejects) for rejects in prompt_rejections.values())
    total_tokens = sum(sum(tokens) for tokens in prompt_total_tokens.values())
    overall_rate = total_rejections / total_tokens if total_tokens > 0 else 0
    
    stats = {
        'overall_rejection_rate': overall_rate,
        'per_prompt_mean_rate': np.mean(per_prompt_rates),
        'per_prompt_std_rate': np.std(per_prompt_rates),
        'num_prompts': len(prompt_rejections)
    }
    
    return stats, all_rejected_probs, all_accepted_probs

def main(cache_dir: str):
    # Compute statistics and collect probabilities
    stats, rejected_probs, accepted_probs = analyze_cache_files(cache_dir)
    
    # Print statistics
    with open(os.path.join(cache_dir, "rejection_statistics.txt"), "w") as f:
        f.write("Cache Directory: " + cache_dir + "\n")
        f.write("Rejection Statistics:\n")
        f.write(f"Overall Rejection Rate: {stats['overall_rejection_rate']:.2%}\n")
        f.write(f"Mean Per-prompt Rejection Rate: {stats['per_prompt_mean_rate']:.2%}\n")
        f.write(f"Std Dev of Per-prompt Rejection Rate: {stats['per_prompt_std_rate']:.2%}\n")
        f.write(f"Number of Prompts Analyzed: {stats['num_prompts'] }\n")
    

# Example usage
if __name__ == "__main__":
    for ngram in [2, 4]:
        for data in ["finance_qa", "longform_qa"]:
            for model in ["phi", "llama"]:
                for method in ["openai", "coupling"]:
                    cache_dir = f"speculative/{data}/{model}/{method}/ngram_{ngram}"
                    main(
                        cache_dir,
                        plot_save_path=f'{cache_dir}/probability_distributions.png'
                    )
