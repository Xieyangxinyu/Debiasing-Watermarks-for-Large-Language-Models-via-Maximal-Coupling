from utils import *
import argparse
import time
import json
import numpy as np

import torch

from wm import WmGenerator

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='llama')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_gen_len', type=int, default=512)

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='data')

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer, _ = load_model(args.model_name)
        
    for param in model.parameters():
        param.requires_grad = False
    
    generator = WmGenerator(model, tokenizer)

    for data in ['finance_qa', 'longform_qa']:
        # load prompts
        prompt_path = f"data/{data}.json"
        prompts = load_prompts(json_path=prompt_path, tokenizer=tokenizer, nsamples=args.nsamples)
        
        # (re)start experiment
        os.makedirs(args.output_dir, exist_ok=True)
        start_point = 0 # if resuming, start from the last line of the file
        if os.path.exists(os.path.join(args.output_dir, f"{data}_{args.model_name}.jsonl")):
            with open(os.path.join(args.output_dir, f"{data}_{args.model_name}.jsonl"), "r") as f:
                for _ in f:
                    start_point += 1
        print(f"Starting from {start_point}")

        instructions = load_results(json_path=prompt_path, nsamples=args.nsamples, result_key="instruction")
        inputs = load_results(json_path=prompt_path, nsamples=args.nsamples, result_key="input")
        # generate
        all_times = []
        with open(os.path.join(args.output_dir, f"{data}_{args.model_name}.jsonl"), "a") as f:
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
                        "output": result}) + "\n")
                    f.flush()


        json_list = []

        # Open the .jsonl file and read line by line
        with open(os.path.join(args.output_dir, f"{data}_{args.model_name}.jsonl"), 'r') as jsonl_file:
            for line in jsonl_file:
                # Parse the JSON object and add it to the list
                json_list.append(json.loads(line))

        # Convert the list to a JSON array
        json_array = json.dumps(json_list, indent=4)

        # Save the JSON array to a new .json file
        with open(os.path.join(args.output_dir, f"{data}_{args.model_name}.json"), 'w') as json_file:
            json_file.write(json_array)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
