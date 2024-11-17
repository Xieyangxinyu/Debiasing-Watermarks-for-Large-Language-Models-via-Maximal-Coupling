'''
Author: Yangxinyu Xie
Date: 2024-11-07

All rights reserved.
'''


from .coupling import CouplingGenerator
from .generator import OpenaiGenerator, WmGenerator, ContextMasking
from transformers import AutoModelForCausalLM
import torch
from typing import List
import os

class SpeculativeGenerator(WmGenerator):
    """ Speculative Sampling as a form of targeted text editing. """
    def __init__(self, 
            *args, 
            model_large: AutoModelForCausalLM = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.model_large = model_large
        self.base_watermarked_generator = WmGenerator(*args, **kwargs)

    def _calculate_fix_probabilities(self, large_probs, base_probs):
        """Calculate probability distribution for rejected samples."""
        fix_probs = torch.max(torch.zeros_like(base_probs), large_probs - base_probs)
        fix_probs.div_(fix_probs.sum(dim=-1, keepdim=True))
        fix_probs[fix_probs != fix_probs] = 1e-10  # Handle NaN values
        return fix_probs
    
    def _handle_rejection_sampling(self, candidate_token, fix_probs, proposal_prob,
                                 acceptance_prob, batch_size):
        """Handle rejection sampling logic and return final token."""
        # Generate random numbers for rejection sampling
        random_values = torch.rand_like(torch.zeros(batch_size)).to(self.device)
        rejection_mask = 0.5 * random_values * proposal_prob > acceptance_prob
        
        # Sample alternative tokens for rejected cases
        alternative_token = torch.multinomial(fix_probs, num_samples=1)
        
        # Reshape tokens to match expected dimensions
        candidate_token = candidate_token.reshape(-1)
        alternative_token = alternative_token.reshape(-1)
        
        # Replace rejected tokens with alternatives
        candidate_token[rejection_mask] = alternative_token[rejection_mask]
        
        return candidate_token, rejection_mask
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        logits_large: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_seeds: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        off: bool = True # whether to turn off the watermarking
    ) -> torch.LongTensor:
        assert temperature > 0, "Temperature must be greater than 0 for this Speculative Sampling Implementation."
    
        base_probs = self.get_sampling_prob_vector(logits, temperature, top_p, ngram_seeds, off=True)
        base_probs_watermarked = self.get_sampling_prob_vector(logits, temperature, top_p, ngram_seeds, off)
        large_probs = self.get_sampling_prob_vector(logits_large, temperature, top_p, ngram_seeds, off=True)
        # Sample initial token
        candidate_token = torch.multinomial(base_probs_watermarked, num_samples=1)
        
        # Calculate acceptance probabilities
        proposal_prob = torch.gather(base_probs, -1, candidate_token).sum(dim=-1)
        proposal_prob_watermarked = torch.gather(base_probs_watermarked, -1, candidate_token).sum(dim=-1)
        acceptance_prob = torch.gather(large_probs, -1, candidate_token).sum(dim=-1)
        
        # Calculate fix probabilities for rejected samples
        fix_probs = self._calculate_fix_probabilities(large_probs, base_probs_watermarked)
        
        # Perform rejection sampling
        next_token, rejection_mask = self._handle_rejection_sampling(
            candidate_token=candidate_token,
            fix_probs=fix_probs,
            proposal_prob=proposal_prob_watermarked,
            acceptance_prob=acceptance_prob,
            batch_size=ngram_seeds.shape[0]
        )
        return next_token, rejection_mask, proposal_prob
    

    def get_sampling_prob_vector(self, logits, temperature, top_p, ngram_tokens, off):
        '''Apply watermarking to the sampling probabilities.'''
        return self.base_watermarked_generator.get_sampling_prob_vector(logits, temperature, top_p, ngram_tokens, off)
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        
        tokens, input_text_mask, start_pos, prev_pos, eos_flags, total_len, prompt_tokens = self.generate_init(prompts, max_gen_len)
        self.base_watermarked_generator.context_masking = self.context_masking

        rejection_masks = torch.zeros_like(tokens, dtype=torch.bool)
        proposal_probs = torch.zeros_like(tokens, dtype=torch.float)
        
        for cur_pos in range(start_pos, total_len):
            # Stop if all sentences have hit eos tok
            if eos_flags.all():
                break
            outputs = self.model(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )

            outputs_large = self.model_large(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs_large.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
        
            # set the logits of the pad token to -inf
            if self.pad_id != self.eos_id:
                outputs.logits[:, -1, self.pad_id] = -float('inf')
            
            ngram_seeds = self.compute_ngram_seeds(ngram_tokens, input_text_mask[:, cur_pos])
            next_toks, rejection_mask, proposal_prob = self.sample_next(outputs.logits[:, -1, :], outputs_large.logits[:, -1, :],
                                          ngram_seeds, temperature, top_p, off = cur_pos < start_pos + self.ngram)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            rejection_masks[:, cur_pos] = rejection_mask
            proposal_probs[:, cur_pos] = proposal_prob
            
            # Check if we've hit eos tok
            eos_flags |= (tokens[:, cur_pos] == self.eos_id) & ~input_text_mask[:, cur_pos]
            prev_pos = cur_pos

        decoded = self.decode(tokens, prompt_tokens, max_gen_len)

        # torch save the rejection_masks, proposal_probs, non_decoded_mask and context_masking_state
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
                'rejection_masks': rejection_masks,
                'proposal_probs': proposal_probs,
                'non_decoded_mask': non_decoded_mask,
                'context_masking_state': self.base_watermarked_generator.context_masking.context_history,
            }, f"{self.save_path}/cache_{i}.pt")
        
        
        return decoded
    

class SpeculativeCouplingGenerator(SpeculativeGenerator):
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        # remove 'model_large' from kwargs
        kwargs.pop('model_large', None)
        self.base_watermarked_generator = CouplingGenerator(
            *args,
            gamma=gamma,
            **kwargs
        )
    

class SpeculativeOpenaiGenerator(SpeculativeGenerator):
    def __init__(self, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        # remove 'model_large' from kwargs
        kwargs.pop('model_large', None)
        self.base_watermarked_generator = OpenaiGenerator(
            *args,
            **kwargs
        )