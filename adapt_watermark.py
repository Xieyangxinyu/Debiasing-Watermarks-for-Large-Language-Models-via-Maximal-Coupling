# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict, List
import os
import time
import json

import tqdm
import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from wm import *
import utils as utils

def binary_entropy(p):
    # Initialize the entropy array with zeros
    p = np.array(p)
    entropy = np.zeros(p.shape)
    
    # Mask for probabilities that are neither 0 nor 1
    valid_mask = (p > 0) & (p < 1)
    
    # Compute binary entropy only for valid probabilities
    entropy[valid_mask] = - (p[valid_mask] * np.log2(p[valid_mask]) + (1 - p[valid_mask]) * np.log2(1 - p[valid_mask]))
    
    return entropy

def calculate_percentage(row):
    count = sum(1 for value in row if value < 0.95)
    total = len(row)
    return (count / total) * 100 if total else 0

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='llama')

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/longform_qa.json")
    parser.add_argument('--prompt_type', type=str, default="alpaca", 
                        help='type of prompt formatting. Choose between: alpaca, oasst, guanaco')
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--one_list', default=False, action='store_true', help="uses only a single green list; only works if detection method is importance-sum")

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose among: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.), importance')
    parser.add_argument('--method_detect', type=str, default='same',
                        help='Statistical test to detect watermark. Choose among: same (same as method), openai, maryland, importance, importance-sum, importance-squared')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=4.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose among: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--do_eval', type=utils.bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)

    return parser


def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    if prompt_type=='alpaca':
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    elif prompt_type=='guanaco':
        PROMPT_DICT = {
            "prompt_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Assistant:"
            )
        }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, prompt_type: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    # new_prompts = [prompt for prompt in prompts if len(prompt["output"].split()) > 5]
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, prompt_type)
    return new_prompts

def load_results(json_path: str, nsamples: int=None, result_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[result_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    if args.model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/Sheared-LLaMA-2.7B')
        model = AutoModelForCausalLM.from_pretrained('princeton-nlp/Sheared-LLaMA-2.7B',device_map="auto").eval()
        vocab_size = 32000
    else:
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
        model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b',device_map="auto").eval()
        vocab_size = 50272


    for param in model.parameters():
        param.requires_grad = False
    
    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    elif args.method == "importance":
        if args.one_list:
            generator = ImportanceGeneratorOneList(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma)
        else:
            generator = ImportanceGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    else:
        prompts = load_prompts(json_path=args.prompt_path, prompt_type=args.prompt_type, nsamples=args.nsamples)

    # do splits
    if args.split is not None:
        nprompts = len(prompts)
        left = nprompts * args.split // args.nsplits 
        right = nprompts * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nprompts
        prompts = prompts[left:right]
        print(f"Creating prompts from {left} to {right}")
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        for ii in range(start_point, len(prompts), args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            if args.method == "importance":
                results, green_cum_probs = generator.generate(
                    prompts[ii:ii+chunk_size], 
                    max_gen_len=args.max_gen_len, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                )
            else:
                results = generator.generate(
                    prompts[ii:ii+chunk_size], 
                    max_gen_len=args.max_gen_len, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            if args.method == "importance":
                for prompt, result, green_cum_prob in zip(prompts[ii:ii+chunk_size], results, green_cum_probs):
                    f.write(json.dumps({
                        "prompt": prompt, 
                        "result": result[len(prompt):],
                        "speed": speed,
                        "eta": eta,
                        "entropy": np.mean(binary_entropy(green_cum_prob)),
                        "percentage_less_than_095": calculate_percentage(green_cum_prob),
                        }) + "\n")
                    f.flush()
            else:
                for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                    f.write(json.dumps({
                        "prompt": prompt, 
                        "result": result[len(prompt):],
                        "speed": speed,
                        "eta": eta}) + "\n")
                    f.flush()

    if args.method_detect == 'same':
        args.method_detect = args.method
    if (not args.do_eval) or (args.method_detect not in ["openai", "maryland", "importance", "importance-sum", "importance-squared"]):
        return
    
    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, vocab_size = vocab_size)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method_detect == "importance":
        detector = ImportanceDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method_detect == "importance-sum":
        if args.one_list:
            detector = ImportanceSumDetectorOneList(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
        else:
            detector = ImportanceSumDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method_detect == "importance-squared":
        detector = ImportanceSquaredDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)

    # build sbert model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="output")
    if args.split is not None:
        results_orig = results_orig[left:right]

    # evaluate
    results = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), nsamples=args.nsamples, result_key="result")
    if "importance" in args.method_detect:
        entropies = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), 
                                       nsamples=args.nsamples, result_key="entropy")
        percents = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), 
                                       nsamples=args.nsamples, result_key="percentage_less_than_095")
    else:
        entropies = np.zeros(len(results))
        percents = np.zeros(len(results))
    log_stats = []
    text_index = left if args.split is not None else 0
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
        for text, text_orig, entropy, percent in tqdm.tqdm(zip(results, results_orig, entropies, percents)):
            # compute watermark score
            if args.method_detect == "openainp":
                scores_no_aggreg, probs = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                pvalues = detector.get_pvalues(scores_no_aggreg, probs)
            elif args.method_detect == "importance":
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg, aggregation = 'max')
                pvalues = detector.get_pvalues(scores_no_aggreg)
            else:
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                pvalues = detector.get_pvalues(scores_no_aggreg) 

            scores = [float(s) for s in scores]
            num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
            # compute sbert score
            xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
            score_sbert = cossim(xs[0], xs[1]).item()
            # log stats and write
            if "importance" in args.method_detect:
                if num_tokens[0] < args.ngram:
                    pvalues[0] = np.nan
                log_stat = {
                    'text_index': text_index,
                    'num_token': num_tokens[0],
                    'score': scores[0],
                    'pvalue': pvalues[0], 
                    'entropy': entropy,
                    'percentage_less_than_095': percent,
                    'score_sbert': score_sbert
                }
            else:
                log_stat = {
                    'text_index': text_index,
                    'num_token': num_tokens[0],
                    'score': scores[0],
                    'pvalue': pvalues[0], 
                    'score_sbert': score_sbert
                }
            log_stats.append(log_stat)
            f.write(json.dumps(log_stat)+'\n')
            text_index += 1
        df = pd.DataFrame(log_stats)
        df['log10_pvalue'] = np.log10(df['pvalue'])
        print(f">>> Scores: \n{df.describe(percentiles=[])}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
