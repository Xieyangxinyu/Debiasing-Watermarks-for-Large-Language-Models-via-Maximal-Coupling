'''
Author: Yangxinyu Xie
Date: 2024-11-07

All rights reserved.
'''

from .generator import OpenaiGenerator
from .detector import WmDetector
from typing import List
from scipy.special import gammaln
from scipy.stats import norm
import pickle
import numpy as np
import torch

class CouplingGenerator(OpenaiGenerator):
    """ Generate text using LM and the watemrarking method based on maximal coupling. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def get_green_list(self, vocab_size):
        vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
        green_list = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
        return green_list

    def apply_watermarking(self, probs, ngram_seeds):
        _, vocab_size = probs.shape
        green_mask = torch.zeros_like(probs).to(probs.device)
        zeta = torch.zeros(ngram_seeds.shape[0]).to(probs.device)
        
        for ii in range(ngram_seeds.shape[0]): # batch of texts
            seed = ngram_seeds[ii].item()
            self.rng.manual_seed(seed)
            # get the green list
            green_list = self.get_green_list(vocab_size)
            # generate zeta randomly between [0,1]
            zeta[ii] = torch.rand(1, generator=self.rng)
            # generate green list mask
            mask = torch.zeros(vocab_size).to(probs.device) # n
            mask[green_list] = 1
            green_mask[ii] = mask
        red_mask = 1 - green_mask
        
        green_cum_prob = (probs * green_mask).sum(dim=-1, keepdim=True)
        # Maximal Coupling
        # This is an equivalent implementation of Algorihtm 1 in the paper,
        # when the pseduorandom variable zeta is fixed conditioned on the previous tokens.
        # If zeta is greater than the cumulative probability of green tokens, we choose red tokens.
        # Otherwise, we choose green tokens.

        mask_to_apply = torch.where(zeta.unsqueeze(-1) > green_cum_prob, red_mask, green_mask)
        probs = probs * mask_to_apply
        # normalize probabilities
        probs.div_(probs.sum(dim=-1, keepdim=True))
        return probs

class CouplingGeneratorOneList(CouplingGenerator):
    """ Generate text using LM and the watemrarking method based on maximal coupling. """
    def __init__(self, 
            *args, 
            green_list_key: int = 42,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.green_list_key = green_list_key
        self.green_list = None

    def get_green_list(self, vocab_size):
        if self.green_list is None:
            self.rng.manual_seed(self.green_list_key)
            self.green_list = torch.randperm(vocab_size, generator=self.rng)[:int(self.gamma * vocab_size)]# gamma * n
        return self.green_list

class CouplingMaxDetector(WmDetector):
    def __init__(self, 
            *args, 
            gamma: float = 0.5, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score = zeta if token_id in greenlist else 1 - zeta 
        The last line shifts the scores by token_id. 
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        zeta = torch.rand(1, generator=self.rng)[0]
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)]
        scores = torch.zeros(self.vocab_size)
        scores += 1 - zeta
        scores[greenlist] = zeta
        return scores.roll(-token_id)
                
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        pvalue = score ** ntoks
        return max(pvalue, eps)
    
    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of scores for each token] for each text
        Output:
            pvalues: np array of p-values for each text
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks
        for ss in scores:
            ntoks = np.sum(ss > 0)
            score = ss.max() if ntoks!=0 else 1.
            pvalues.append(self.get_pvalue(score, ntoks, eps=eps))
        return np.asarray(pvalues)
    

class CouplingSumDetector(CouplingMaxDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def irwin_hall_cdf(x, n):
        if n <= 15:
            x = x[0]
            k = np.arange(int(np.floor(x)) + 1)
            log_factorials = gammaln(k + 1) + gammaln(n - k + 1)
            log_power = n * np.log(np.abs(x - k))
            term_log = log_power - log_factorials
            terms = np.exp(term_log) * ((-1)**k)
            cdf = np.sum(terms)
            cdf = np.array([cdf])
        else:
            cdf = norm.cdf(x, loc=n/2, scale=np.sqrt(n/12))
        return cdf

    def get_pvalue(self, score: float, ntoks: int, eps: float):
        pvalue = self.irwin_hall_cdf(score, ntoks)
        return np.maximum(pvalue, eps)
    
    def get_pvalues(self, scores, eps: float=1e-200):
        scores = np.array(scores)  # Ensure scores is a NumPy array
        ntoks = np.sum(scores > 0, axis=1)[0]  # Number of tokens per text
        total_scores = np.sum(scores, axis=1)  # Total score per text
        total_scores[ntoks == 0] = 1.  # Handle case where ntoks == 0
        pvalues = self.get_pvalue(total_scores, ntoks, eps)
        return pvalues

class CouplingSumDetectorOneList(CouplingSumDetector):
    def __init__(self, 
            *args, 
            green_list_key: int = 42,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.rng.manual_seed(green_list_key)
        self.green_list = torch.randperm(self.vocab_size, generator=self.rng)[:int(self.gamma * self.vocab_size)]# gamma * n
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        zeta = torch.rand(1, generator=self.rng)[0]
        greenlist = self.green_list
        scores = torch.zeros(self.vocab_size)
        scores += 1 - zeta
        scores[greenlist] = zeta
        return scores.roll(-token_id)
    

class CouplingHCDetector(CouplingSumDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def HC_test(p):
        """
        Calculate the Higher Criticism (HC) threshold for a given set of p-values and determine
        if the null hypothesis H_0 should be accepted or rejected based on the HC statistic.

        Inputs:
            p (array-like): An n-by-1 array of p-values from the data.

        Outputs:
            H (int): A scalar, "0" or "1", indicating whether H_0 is accepted ("0") or rejected ("1"),
                    based on whether the HC threshold exceeds a critical value.
        """
        n_sample = len(p)
        
        # Sort P along the last axis (samples within each simulation)
        P_sorted = np.sort(p)
        
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
        HC_plus = np.max(HC_plus_values)

        file_path = f'HC_simulate_data/{n_sample}.pkl'
        empirical_dist = pickle.load(open(file_path, 'rb'))

        critical_value = empirical_dist['critical_value']

        return HC_plus > critical_value

    def get_decisions(
            self, 
            scores: List[np.array], 
        ) -> np.array:
        """
        Get test decisions for each text.
        Args:
            score_lists: list of [list of scores for each token] for each text
        Output:
            decisions: list of test decisions
        """
        decisions = []
        scores = np.asarray(scores)
        for ss in scores:
            ntoks = len(ss)
            decisions.append(self.HC_test(ss) if ntoks!=0 else 0.)
        return decisions
    

class CouplingHCDetectorOneList(CouplingHCDetector):
    def __init__(self, 
            *args, 
            green_list_key: int = 42,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.rng.manual_seed(green_list_key)
        self.green_list = torch.randperm(self.vocab_size, generator=self.rng)[:int(self.gamma * self.vocab_size)]# gamma * n
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        zeta = torch.rand(1, generator=self.rng)[0]
        greenlist = self.green_list
        scores = torch.zeros(self.vocab_size)
        scores += 1 - zeta
        scores[greenlist] = zeta
        return scores.roll(-token_id)