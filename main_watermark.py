import os
from utils import *
import argparse
import time
import json

import tqdm
import pandas as pd
import numpy as np

import torch
from sentence_transformers import SentenceTransformer

from wm import *

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='llama')

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/finance_qa_llama.json")
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_gen_len', type=int, default=512)
    parser.add_argument('--one_list', default=False, action='store_true', help="uses only a single green list; only works if detection method is coupling")

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose among: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.), coupling, dipmark')
    parser.add_argument('--method_detect', type=str, default='same',
                        help='Statistical test to detect watermark. Choose among: openai, maryland, dipmark, coupling-max, coupling, coupling-HC')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='gamma for maryland/coupling: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=1.0, 
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--do_eval', type=bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)

    return parser


    
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    model, tokenizer, vocab_size = load_model(args.model_name)

    for param in model.parameters():
        param.requires_grad = False

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    else:
        prompts = load_prompts(json_path=args.prompt_path, tokenizer=tokenizer, nsamples=args.nsamples)
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, save_path = args.output_dir)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta, save_path = args.output_dir)
    elif args.method == "coupling":
        if args.one_list:
            generator = CouplingGeneratorOneList(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, save_path = args.output_dir)
        else:
            generator = CouplingGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, save_path = args.output_dir)
    elif args.method == "dipmark":
        generator = DiPMarkGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, save_path = args.output_dir)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        for ii in range(start_point, len(prompts), args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
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
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                f.write(json.dumps({
                    "prompt": prompt, 
                    "result": result,
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

    if args.method_detect == 'same':
        args.method_detect = args.method
    if (not args.do_eval) or (args.method_detect not in ["openai", "maryland", "coupling", "dipmark"]):
        print("method_detect not implemented")
        return
    
    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, vocab_size = vocab_size)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method_detect == "coupling":
        if args.one_list:
            detector = CouplingSumDetectorOneList(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
        else:
            detector = CouplingSumDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    elif args.method_detect == "dipmark":
        detector = DiPMarkDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    # build sbert model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="output")

    # evaluate
    results = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), nsamples=args.nsamples, result_key="result")
    log_stats = []
    text_index = 0
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
        for text, text_orig in tqdm.tqdm(zip(results, results_orig)):
            scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
            scores = detector.aggregate_scores(scores_no_aggreg)
            pvalues = detector.get_pvalues(scores_no_aggreg)

            scores = [float(s) for s in scores]
            num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
            # compute sbert score
            xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
            score_sbert = cossim(xs[0], xs[1]).item()
            # log stats and write
            
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
        
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f"{df.describe(percentiles=[])}"+'\n')

        # Compute True Positive Rate
        TPR = sum(df['pvalue'] < 0.01) / len(df)
        f.write(f'TPR: {TPR}'+'\n')
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
