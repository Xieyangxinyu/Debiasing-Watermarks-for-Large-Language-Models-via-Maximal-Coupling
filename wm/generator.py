'''
Author: Yangxinyu Xie
Date: 2024-11-07

This code is adapted from https://github.com/facebookresearch/three_bricks/blob/main/wm/generator.py and https://github.com/yihwu/dipmark
'''


from typing import List

import torch
import torch.nn.functional as F

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class ContextMasking():
    def __init__(self, 
            repeated_context_masking: bool = False, 
            batch_size: int = 1,
            context_history_size: int = 1,
            device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        ):
        self.repeated_context_masking = repeated_context_masking
        self.context_history = torch.zeros(
            (batch_size, context_history_size),
            dtype=torch.int64,
            device=device,
        )

    def check_and_update_repeated_context(self, 
            seeds_from_current_context: torch.LongTensor, # size (bsz, 1)
        ) -> torch.BoolTensor:
        '''Check if the current context is repeated. Update the context history with the current context.'''

        is_repeated_context = (
            self.context_history == seeds_from_current_context
        ).any(dim=1, keepdim=True)
        self.context_history = torch.cat(
            (self.context_history[:, 1:], seeds_from_current_context), dim=1
        )
        return is_repeated_context

    def apply_repeated_context_masking(self, 
            probs: torch.FloatTensor, # (bsz, vocab_size): logits/probs for last token
            probs_watermarked: torch.FloatTensor, # (bsz, vocab_size): watermarked logits/probs for last token
            seeds_from_current_context: torch.LongTensor, # (bsz, 1): tokens to consider when seeding
        ) -> torch.FloatTensor:
        '''Set the probabilities of repeated contexts to zero.'''
        if self.repeated_context_masking:
            is_repeated_context = self.check_and_update_repeated_context(seeds_from_current_context)
            probs_watermarked = torch.where(is_repeated_context, probs, probs_watermarked)
        return probs_watermarked

class WmGenerator():
    def __init__(self, 
            model: AutoModelForCausalLM, 
            tokenizer: AutoTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            repeated_context_masking: bool = True,
            context_history_size: int = -1, # -1 means keep all history, else keep the last context_history_size
            save_path: str = None,
        ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device
        self.max_seq_len = 1024
        self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.repeated_context_masking = repeated_context_masking
        self.context_masking = None
        self.context_history_size = context_history_size
        self.save_path = save_path

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
            MAX_INT64 = torch.iinfo(torch.int64).max
            seed = int(self.seed) & MAX_INT64  # Ensure initial seed is within bounds
            
            for i in input_ids:
                # Convert tensors to Python ints and keep everything within bounds
                i_val = int(i.item()) & MAX_INT64
                salt = int(self.salt_key) & MAX_INT64
                
                # Break down the calculation into smaller steps
                # Use modulo at each step to prevent overflow
                temp = (seed % MAX_INT64) * (salt % MAX_INT64)
                temp = temp % MAX_INT64
                temp = (temp + i_val) % MAX_INT64
                seed = temp
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

    def compute_ngram_seeds(self, 
                        ngram_tokens: torch.LongTensor, 
                        input_text_mask: torch.BoolTensor
                        ) -> torch.LongTensor:
        """Compute hash of n-gram tokens."""
        seeds_from_current_context = torch.zeros(ngram_tokens.shape[0], dtype=torch.int64, device=ngram_tokens.device)
        
        for i in range(ngram_tokens.shape[0]):
            seeds_from_current_context[i] = self.get_seed_rng(ngram_tokens[i])
        
        # Apply the mask: invert the Boolean mask to apply to seeds if the token is in the prompt
        seeds_from_current_context *= (~input_text_mask).long()
        
        # check if seeds_from_current_context has changed due to the mask, i.e. if input_text_mask is True, then the seed is 0
        for i in range(ngram_tokens.shape[0]):
            if input_text_mask[i] and seeds_from_current_context[i] != 0:
                print("Error: seeds_from_current_context is not 0 when input_text_mask is True")

        return seeds_from_current_context.unsqueeze(-1)
    
    def generate_init(self, prompts: List[str], max_gen_len: int):
        bsz = len(prompts)

        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        if self.context_history_size == -1:
            self.context_masking = ContextMasking(self.repeated_context_masking, bsz, total_len, self.device)
        else:
            self.context_masking = ContextMasking(self.repeated_context_masking, bsz, self.context_history_size, self.device)

        tokens = torch.full((bsz, total_len), self.pad_id).to(self.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        eos_flags = torch.zeros(tokens.size(0), dtype=torch.bool, device=self.device)  # Track which sentences hit EOS
        return tokens, input_text_mask, start_pos, prev_pos, eos_flags, total_len, prompt_tokens

    def decode(self, tokens, prompt_tokens, max_gen_len):

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            # remove prompt tokens
            t = t[len(prompt_tokens[i]):]
            t = t[:max_gen_len]
            # cut to eos tok if any
            try:
                index = t.index(self.eos_id)
                t = t[: index]
            except ValueError:
                # if pad token is in the middle of the sentence, cut it
                if self.pad_id != self.eos_id and self.pad_id in t:
                    index = t.index(self.pad_id)
                    t = t[: index]
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        
        tokens, input_text_mask, start_pos, prev_pos, eos_flags, total_len, prompt_tokens = self.generate_init(prompts, max_gen_len)

        # Track which sentences have hit eos tok
        eos_flags = torch.zeros(tokens.size(0), dtype=torch.bool, device=self.device)  # Track which sentences hit EOS
        
        for cur_pos in range(start_pos, total_len):
            # Stop if all sentences have hit eos tok
            if eos_flags.all():
                break
            outputs = self.model(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
        
            # set the logits of the pad token to -inf
            if self.pad_id != self.eos_id:
                outputs.logits[:, -1, self.pad_id] = -float('inf')
            
            
            ngram_seeds = self.compute_ngram_seeds(ngram_tokens, input_text_mask[:, cur_pos])
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_seeds, temperature, top_p, off = cur_pos < start_pos + self.ngram)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)

            # Check if we've hit eos tok and this eos tok is not in the prompt
            eos_flags |= (tokens[:, cur_pos] == self.eos_id) & ~input_text_mask[:, cur_pos]
            prev_pos = cur_pos

        decoded = self.decode(tokens, prompt_tokens, max_gen_len)

        # torch save the entropy values, input_text_mask, eos_mask, pad_mask, context_masking_state
        if self.save_path is not None:
            # check if self.save_path/cache_{i}.pt exists and if it does, increment i
            i = 0
            while os.path.exists(f"{self.save_path}/cache_{i}.pt"):
                i += 1
            
            # create eos_id mask and pad_id mask and combine with input_text_mask
            eos_mask = tokens == self.eos_id
            pad_mask = tokens == self.pad_id
            eos_or_pad_mask = eos_mask | pad_mask
            non_decoded_mask = eos_or_pad_mask | input_text_mask
            torch.save({
                'non_decoded_mask': non_decoded_mask,
                'context_masking_state': self.context_masking.context_history,
            }, f"{self.save_path}/cache_{i}.pt")
        
        return decoded
    
    def get_sampling_prob_vector(self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        ngram_seeds: torch.LongTensor = None, # (bsz, ngram): tokens to consider when seeding
        off: bool = True # whether to turn off the watermarking
    ) -> torch.FloatTensor:
        """ Apply temperature and top p sampling. For base class, watermarking is off."""
        probs = torch.softmax(logits / temperature, dim=-1)
        
        if top_p < 1:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # undo the sort
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, probs_idx, probs_sort)

        return probs
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_seeds: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = True # whether to turn off the watermarking
    ) -> torch.LongTensor:
        """Sample next token from logits."""
        if temperature > 0:
            probs = self.get_sampling_prob_vector(logits, temperature, top_p, ngram_seeds, off) # one hot of next token, ordered by original probs
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class OpenaiGenerator(WmGenerator):
    """ Generate text using LM and Aaronson's watermarking method. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def apply_watermarking(self, probs, ngram_seeds):
        vocab_size = probs.shape[-1]
        for ii in range(ngram_seeds.shape[0]): # batch of texts
            # seed with hash of ngram tokens
            seed = ngram_seeds[ii].item()
            self.rng.manual_seed(seed)
            # generate rs randomly between [0,1]
            rs = torch.rand(vocab_size, generator=self.rng).to(self.device) # n
            # compute r^(1/p)
            probs[ii] = torch.pow(rs, 1/probs[ii])
        # change probs to be one hot on argmax ( r^(1/p) )
        probs_watermarked = F.one_hot(torch.argmax(probs, dim=-1), num_classes=vocab_size).float().to(self.device)
        return probs_watermarked
        
    def get_sampling_prob_vector(self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        ngram_seeds: torch.LongTensor = None, # (bsz, ngram): tokens to consider when seeding
        off: bool = False # whether to turn off the watermarking
    ) -> torch.FloatTensor:
        """ Apply temperature and top p sampling. For OpenaiGenerator, watermarking is on."""
        probs = super().get_sampling_prob_vector(logits, temperature, top_p, ngram_seeds, off)
        if not off:
            probs_watermarked = self.apply_watermarking(probs, ngram_seeds)
            probs = self.context_masking.apply_repeated_context_masking(probs, probs_watermarked, ngram_seeds)
            # print whether the watermarking is effectively applied
        return probs

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

    def get_sampling_prob_vector(self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        ngram_seeds: torch.LongTensor = None, # (bsz, ngram): tokens to consider when seeding
        off: bool = False # whether to turn off the watermarking
    ) -> torch.FloatTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and redlist 
        - add delta to greenlist words' logits
        """
        if not off:
            logits_watermarked = self.logitprobsrocessor(logits, ngram_seeds)
            logits = self.context_masking.apply_repeated_context_masking(logits, logits_watermarked, ngram_seeds)
        return super().get_sampling_prob_vector(logits, temperature, top_p, ngram_seeds, off)

    def logitprobsrocessor(self, logits, ngram_seeds):
        """Process logits to mask out words in greenlist."""
        _, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_seeds.shape[0]): # batch of texts
            seed = ngram_seeds[ii].item()
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(self.device) # n
            bias[greenlist] = self.delta
            logits[ii] += bias # add bias to greenlist words
        return logits
    
class DiPMarkGenerator(OpenaiGenerator):
    """ Generate text using LM and DiP-Mark watermarking method. """
    def __init__(self, 
            *args, 
            alpha: float = 0.45,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.alpha = alpha

    def get_shuffle(self, 
                 ngram_seeds: torch.LongTensor, 
                 vocab_size: int
        ) -> torch.LongTensor:
        """Get shuffle for the seeds."""
        # Preallocate a tensor to store all shuffle
        shuffle = torch.empty((ngram_seeds.shape[0], vocab_size), dtype=torch.long, device=self.device)
        
        # Fill the preallocated tensor with shuffled indices
        for ii in range(ngram_seeds.shape[0]):
            seed = ngram_seeds[ii].item()
            self.rng.manual_seed(seed)
            shuffle[ii] = torch.randperm(vocab_size, generator=self.rng).to(self.device)
        return shuffle

    def apply_watermarking(self, probs, ngram_seeds):
        """
        Reweight the probabilities.
        The difference from MarylandGenerator is that the set of green tokens is defined as the latter (gamma*V words) of the permuted vocabulary.
        """
        shuffle = self.get_shuffle(ngram_seeds, probs.shape[-1])
        unshuffle = torch.argsort(shuffle, dim=-1)

        probs = torch.gather(probs, -1, shuffle)
        probs_cumsum = torch.cumsum(probs, dim=-1)

        boundary_1 = torch.argmax((probs_cumsum > self.alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(probs, -1, boundary_1)
        portion_in_right_1 = (torch.gather(probs_cumsum, -1, boundary_1) - self.alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (probs_cumsum > self.alpha).type_as(probs)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((probs_cumsum > (1-self.alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(probs, -1, boundary_2)
        portion_in_right_2 = (torch.gather(probs_cumsum, -1, boundary_2) - (1-self.alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (probs_cumsum > (1-self.alpha)).type_as(probs)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2/2 + s_all_portion_in_right_1/2
        probs = probs * s_all_portion_in_right
        probs.div_(probs.sum(dim=-1, keepdim=True))

        # unshuffle the probabilities
        probs = torch.gather(probs, -1, unshuffle)

        return probs
