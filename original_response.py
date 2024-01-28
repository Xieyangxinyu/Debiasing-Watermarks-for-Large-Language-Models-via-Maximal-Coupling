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

from wm import WmGenerator
import utils as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='llama')

    # prompts parameters
    parser.add_argument('--data', type=str, default="finance_qa")

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=32)
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


def format_prompts(prompts: List[Dict]) -> List[str]:
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    }
    
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts)
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
    else:
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
        model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b',device_map="auto").eval()

    for param in model.parameters():
        param.requires_grad = False
    
    generator = WmGenerator(model, tokenizer)

    # load prompts
    prompt_path = f"data/{args.data}.json"
    prompts = load_prompts(json_path=prompt_path, nsamples=args.nsamples)

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
    if os.path.exists(os.path.join(args.output_dir, f"{args.data}_{args.model_name}.jsonl")):
        with open(os.path.join(args.output_dir, f"{args.data}_{args.model_name}.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    instructions = load_results(json_path=prompt_path, nsamples=args.nsamples, result_key="instruction")
    inputs = load_results(json_path=prompt_path, nsamples=args.nsamples, result_key="input")
    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"{args.data}_{args.model_name}.jsonl"), "a") as f:
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
            # log chunk
            for instruction, input, prompt, result in zip(instructions[ii:ii+chunk_size], inputs[ii:ii+chunk_size], 
                                                           prompts[ii:ii+chunk_size], results):
                f.write(json.dumps({
                    "instruction": instruction, 
                    "input": input,
                    "output": result[len(prompt):]}) + "\n")
                f.flush()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
