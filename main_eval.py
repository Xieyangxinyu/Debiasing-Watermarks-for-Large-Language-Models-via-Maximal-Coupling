# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List
import os
import json

import tqdm
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer

from wm import *
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--text_key', type=str, default='text',
                        help='key to access text in json dict')
    parser.add_argument('--tokenizer', type=str, default='opt')

    # watermark parameters
    parser.add_argument('--do_wmeval', type=utils.bool_inst, default=True,
                        help='whether to do watermark evaluation')
    parser.add_argument('--method', type=str, default='none',
                        help='watermark detection method')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')
    parser.add_argument('--one_list', default=False, action='store_true', help="uses only a single green list; only works if detection method is importance-sum")
    parser.add_argument('--alpha', type=float, default=0.0001)
    
    # attack
    parser.add_argument('--attack_name', type=str, default='none',
                        help='attack name to be applied to text before evaluation. Choose between: \
                        none (no attack), tok_substitution (randomly substitute tokens)')
    parser.add_argument('--attack_param', type=float, default=0.1,
                        help='attack parameter. For tok_substitution, it is the probability of substitution')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to evaluate, if None, take all texts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=None,
                        help='split the texts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat texts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat texts as a whole')


    return parser

def load_results(json_path: str, nsamples: int=None, text_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[text_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.tokenizer == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/Sheared-LLaMA-2.7B')
        vocab_size = 32000
    else:
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
        vocab_size = 50272

    # build watermark detector
    if args.method == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, vocab_size = vocab_size)
    elif args.method == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method == "importance-max":
        detector = ImportanceMaxDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method == "importance-sum":
        if args.one_list:
            detector = ImportanceSumDetectorOneList(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
        else:
            detector = ImportanceSumDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method == "importance-HC":
        if args.one_list:
            detector = ImportanceHCDetectorOneList(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
        else:
            detector = ImportanceHCDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)

    # load results and (optional) do splits
    results = load_results(json_path=f'{args.json_path}/results.jsonl', text_key=args.text_key, nsamples=args.nsamples)
    print(f"Loaded {len(results)} results.")
    if args.split is not None:
        nresults = len(results)
        left = nresults * args.split // args.nsplits 
        right = nresults * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nresults
        results = results[left:right]
        print(f"Creating results from {left} to {right}")

    # attack results
    attack_name = args.attack_name
    attack_param = args.attack_param
    if attack_name == 'tok_substitution':
        def attack(text):
            tokens_id = tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens_id:
                if np.random.rand() < attack_param:
                    tokens_id[tokens_id.index(token_id)] = np.random.randint(tokenizer.vocab_size)
            new_text = tokenizer.decode(tokens_id)
            return new_text
        if args.attack_param > 0:
            #print(results[:2])
            results = [attack(text) for text in results]
            print(f"Attacked results with {attack_name}({attack_param})")
            #print(results[:2])

    # evaluate
    log_stats = []
    for ii, text in tqdm.tqdm(enumerate(results), total=len(results)):
        log_stat = {
            'text_index': ii,
        }
        if args.do_wmeval:
            if args.method == "importance-max":
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg, aggregation = 'max')
                pvalues = detector.get_pvalues(scores_no_aggreg)
            elif args.method == "importance-HC":
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                decisions = detector.get_decisions(scores_no_aggreg)
            else:
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                pvalues = detector.get_pvalues(scores_no_aggreg) 

            num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
            log_stat['num_token'] =  num_tokens[0]
            if args.method == "importance-HC":
                log_stat['decision'] = decisions[0]
            else:
                scores = [float(s) for s in scores]
                log_stat['score'] =  scores[0]
                log_stat['pvalue'] =  pvalues[0]
            log_stats.append(log_stat)
    


    df = pd.DataFrame(log_stats)
    if args.method == "importance-HC":
        TPR = sum(df['decision']) / len(df)
    else:
        TPR = sum(df['pvalue'] < args.alpha) / len(df)
    print(f'TPR: {TPR} out of {len(df)} texts.')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
