from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import json
from huggingface_hub import login

#login(token="your_hf_token")

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def load_prompts(json_path: str, tokenizer, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, tokenizer)
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

def format_prompts(prompts: List[Dict], tokenizer) -> List[str]:
    try:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": example["instruction"]},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for example in prompts
        ]
    except:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": example["instruction"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for example in prompts
        ]
    return prompts


def load_large_model(model_name, device):
    if model_name == 'phi':
        model_large = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-medium-4k-instruct',device_map=device, torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2").eval()
    elif model_name == 'llama':
        model_large = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct',device_map=device, torch_dtype="auto", attn_implementation="flash_attention_2").eval()
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    return model_large

def load_tokenizer(model_name):
    if model_name == 'phi':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)
        vocab_size = 32064
    elif model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
        vocab_size = 128256
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    return tokenizer, vocab_size

def load_model(model_name, large = False):
    tokenizer, vocab_size = load_tokenizer(model_name)
    if model_name == 'phi':
        model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct',device_map="auto", torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2").eval()
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    elif model_name == 'llama':
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct',device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2").eval()
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    else:
        raise NotImplementedError(f"{model_name} not implemented")

    if large:
        model_large = load_large_model(model_name, model.device)
        return model, tokenizer, vocab_size, model_large
    return model, tokenizer, vocab_size
