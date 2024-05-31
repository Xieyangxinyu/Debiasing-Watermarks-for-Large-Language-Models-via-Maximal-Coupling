# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmGenerator():
    def __init__(self, 
            model: AutoModelForCausalLM, 
            tokenizer: AutoTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = 1024
        self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id
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
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/LM/
        """
        
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, off = (cur_pos < start_pos + self.ngram))
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = False # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class OpenaiGenerator(WmGenerator):
    """ Generate text using LM and Aaronson's watermarking method. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = False # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            if off:
                next_token = torch.multinomial(probs_sort, num_samples=1)
            else:
                vocab_size = probs_sort.shape[-1]
                for ii in range(ngram_tokens.shape[0]): # batch of texts
                    # seed with hash of ngram tokens
                    seed = self.get_seed_rng(ngram_tokens[ii])
                    self.rng.manual_seed(seed)
                    # generate rs randomly between [0,1]
                    rs = torch.rand(vocab_size, generator=self.rng) # n
                    rs = torch.Tensor(rs).to(probs_sort.device)
                    rs = rs[probs_idx[ii]] 
                    # compute r^(1/p)
                    probs_sort[ii] = torch.pow(rs, 1/probs_sort[ii])
                # select argmax ( r^(1/p) )
                next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class MarylandGenerator(WmGenerator):
    """ Generate text using LM and Maryland's watemrarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = False # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        """
        if not off:
            logits = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(logits.device) # n
            bias[greenlist] = self.delta
            logits[ii] += bias # add bias to greenlist words
        return logits


class ImportanceGenerator(WmGenerator):
    """ Generate text using LM and the watemrarking method based on Importance Sampling. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = False # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and redlist 
        - choose which list to generate from based on the importance sampling random variable xi
        """
        logits, green_mask, red_mask, xi = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)

            # Top_p sampling
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            if off:
                next_token = torch.multinomial(probs_sort, num_samples=1)
                green_cum_prob = torch.ones_like(xi.unsqueeze(-1)).to(logits.device)
            else:
                # Choose which mask to apply based on xi
                green_mask_sorted = torch.gather(green_mask, dim=-1, index=probs_idx)
                green_cum_prob = (probs_sort * green_mask_sorted).sum(dim=-1, keepdim=True)

                red_mask_sorted = torch.gather(red_mask, dim=-1, index=probs_idx)

                mask_to_apply = torch.where(xi.unsqueeze(-1) > green_cum_prob, red_mask_sorted, green_mask_sorted)
                green_cum_prob = torch.where(xi.unsqueeze(-1) > green_cum_prob, 1 - green_cum_prob, green_cum_prob)
                probs_sort = probs_sort * mask_to_apply
                # Normalize probability
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token, green_cum_prob.squeeze()

    def logits_processor(self, logits, ngram_tokens):
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        green_mask = torch.zeros_like(logits).to(logits.device)
        xi = torch.zeros(ngram_tokens.shape[0]).to(logits.device)
        
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            xi[ii] = torch.rand(1, generator=self.rng)
            mask = torch.zeros(vocab_size).to(logits.device) # n
            mask[greenlist] = 1
            green_mask[ii] = mask
        red_mask = 1 - green_mask
        return logits, green_mask, red_mask, xi
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        green_cum_probs = torch.zeros((bsz, total_len)).to(device).float()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks, green_cum_prob = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, off = (cur_pos < start_pos + self.ngram))
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            green_cum_probs[:, cur_pos] = green_cum_prob
            prev_pos = cur_pos
            

        decoded = []
        green_cum_probs = green_cum_probs.cpu().numpy().tolist()
        green_cum_probs_list = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            green_cum_prob = green_cum_probs[i][: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                index = t.index(self.eos_id)
                t = t[: index]
                green_cum_probs_list.append(green_cum_prob[len(prompt_tokens[i]): index])
            except ValueError:
                green_cum_probs_list.append(green_cum_prob[len(prompt_tokens[i]): ])
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded, green_cum_probs_list


class ImportanceGeneratorOneList(ImportanceGenerator):
    """ Generate text using LM and the watemrarking method based on Importance Sampling. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            green_list_key: int = 42,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.green_list_key = green_list_key
        self.green_list = None

    def logits_processor(self, logits, ngram_tokens):
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        green_mask = torch.zeros_like(logits).to(logits.device)
        xi = torch.zeros(ngram_tokens.shape[0]).to(logits.device)

        if self.green_list is None:
            self.rng.manual_seed(self.green_list_key)
            self.green_list = torch.randperm(vocab_size, generator=self.rng)[:int(self.gamma * vocab_size)]# gamma * n
        
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            greenlist = self.green_list
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            xi[ii] = torch.rand(1, generator=self.rng)
            mask = torch.zeros(vocab_size).to(logits.device) # n
            mask[greenlist] = 1
            green_mask[ii] = mask
        red_mask = 1 - green_mask
        return logits, green_mask, red_mask, xi


class SpeculativeGenerator(ImportanceGenerator):
    """ Generate text using LM and the watemrarking method based on Speculative Sampling. """
    def __init__(self, 
            *args, 
            model_large: AutoModelForCausalLM = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.model_large = model_large

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        logits_large: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = False # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and redlist 
        - choose which list to generate from based on the importance sampling random variable xi
        """
        logits, green_mask, _, xi = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_large = torch.softmax(logits_large / temperature, dim=-1)

            # Top_p sampling
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # sort back to original vocab order
            probs = torch.zeros_like(probs_sort)
            probs.scatter_(dim=-1, index=probs_idx, src=probs_sort)

            probs_sort_large, probs_idx_large = torch.sort(probs_large, dim=-1, descending=True)
            probs_sum_large = torch.cumsum(probs_sort_large, dim=-1)
            mask_large = probs_sum_large - probs_sort_large > top_p
            probs_sort_large[mask_large] = 0.0
            probs_sort_large.div_(probs_sort_large.sum(dim=-1, keepdim=True))
            probs_large = torch.zeros_like(probs_sort_large)
            probs_large.scatter_(dim=-1, index=probs_idx_large, src=probs_sort_large)

            if off:
                next_token = torch.multinomial(probs_large, num_samples=1)
                next_token = next_token.reshape(-1)
            else:
                # add greenlist to probs
                probs = probs * green_mask
                probs.div_(probs.sum(dim=-1, keepdim=True))
                probs[probs != probs] = 1e-10
                probs.div_(probs.sum(dim=-1, keepdim=True))

                # Compute P' = norm(max(0, P - Q))
                probs_fix = torch.max(torch.zeros_like(probs), probs_large - probs)
                probs_fix.div_(probs_fix.sum(dim=-1, keepdim=True))

                probs_fix[probs_fix != probs_fix] = 1e-10
                
                # reject sampling
                next_token = torch.multinomial(probs, num_samples=1) # one hot of next token, ordered by original probs

                Q = torch.gather(probs, -1, next_token).sum(dim=-1)
                P = torch.gather(probs_large, -1, next_token).sum(dim=-1)
                reject_mask = xi * Q > P
                next_token_reject = torch.multinomial(probs_fix, num_samples=1)
                next_token = next_token.reshape(-1)
                next_token_reject = next_token_reject.reshape(-1)
                next_token[reject_mask] = next_token_reject[reject_mask]
        else:
            next_token = torch.argmax(logits_large, dim=-1)
            next_token = next_token.reshape(-1)
        return next_token
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            outputs_large = self.model_large.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs_large.past_key_values if prev_pos > 0 else None
            )

            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], outputs_large.logits[:, -1, :], ngram_tokens, temperature, top_p, off = (cur_pos < start_pos + self.ngram))
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded