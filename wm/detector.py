# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import numpy as np
from scipy import special
from math import factorial
from scipy.stats import norm, beta
import torch
from transformers import LlamaTokenizer
import pickle

class WmDetector():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            vocab_size: int = 50257,
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else 1. for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) if ss.shape[0]!=0 else 1. for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
        Output:
            score_lists: list of [np array of score increments for every token] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt.numpy()[0]
                rts.append(rt)
            score_lists.append(rts)
        return score_lists

    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text
        """
        pvalues = []
        scores = np.asarray(scores)
        for ss in scores:
            ntoks = len(ss)
            score = ss.sum() if ntoks!=0 else 0.
            pvalues.append(self.get_pvalue(score, ntoks, eps=eps))
        return np.asarray(pvalues)

    def get_pvalues_by_t(self, scores: List[float]) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError


class MarylandDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)
    
class OpenaiDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)
    

class ImportanceMaxDetector(WmDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score = xi if token_id in greenlist else 1 - xi 
        The last line shifts the scores by token_id. 
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(1, generator=self.rng)[0]
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)]
        scores = torch.zeros(self.vocab_size)
        scores += 1 - xi
        scores[greenlist] = xi
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
    

class ImportanceSumDetector(ImportanceMaxDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
    
    @staticmethod
    def irwin_hall_cdf(x, n):
        if n <= 15:
            k = np.arange(0, int(np.max(x)) + 1)[:, np.newaxis]
            summands = np.sum((-1) ** k * special.comb(n, k) * (x - k.T) ** n, axis=0)
            cdf = summands / np.math.factorial(n)
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

class ImportanceSumDetectorOneList(ImportanceSumDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            green_list_key: int = 42,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.rng.manual_seed(green_list_key)
        self.green_list = torch.randperm(self.vocab_size, generator=self.rng)[:int(self.gamma * self.vocab_size)]# gamma * n
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(1, generator=self.rng)[0]
        greenlist = self.green_list
        scores = torch.zeros(self.vocab_size)
        scores += 1 - xi
        scores[greenlist] = xi
        return scores.roll(-token_id)
    

class ImportanceHCDetector(ImportanceSumDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        
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
    
class ImportanceHCDetectorOneList(ImportanceHCDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            green_list_key: int = 42,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.rng.manual_seed(green_list_key)
        self.green_list = torch.randperm(self.vocab_size, generator=self.rng)[:int(self.gamma * self.vocab_size)]# gamma * n
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(1, generator=self.rng)[0]
        greenlist = self.green_list
        scores = torch.zeros(self.vocab_size)
        scores += 1 - xi
        scores[greenlist] = xi
        return scores.roll(-token_id)