from tqdm import tqdm
import torch
import random
import json
import argparse
import os
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Ensure WordNet is available
# nltk.download('wordnet')

class ContextAwareSynonymSubstitution():
    def __init__(self, ratio: float, tokenizer, model, device='cuda') -> None:
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def _get_synonyms_from_wordnet(self, word: str):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def _truncate_around_mask(self, words: list, mask_index: int, window_size: int = 128):
        half_window = window_size // 2
        start = max(0, mask_index - half_window)
        end = min(len(words), mask_index + half_window + 1)
        return words[start:end]

    def edit(self, text: str):
        words = text.split()
        num_words = len(words)
        replaceable_indices = [
            i for i, word in enumerate(words)
            if self._get_synonyms_from_wordnet(word)
        ]

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        for i in indices_to_replace:
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)

            max_retries = 5
            for attempt in range(max_retries):
                inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
                mask_positions = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]

                if len(mask_positions) > 0:
                    mask_position = mask_positions[0].item()
                    break
                else:
                    masked_sentence = self._truncate_around_mask(masked_sentence, mask_index=i, window_size=128)
                    masked_text = " ".join(masked_sentence)
            else:
                continue  # skip if failed to find [MASK]

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            replacement = predicted_tokens[0]

            # Adjust casing
            if words[i][0].isupper():
                replacement = replacement.capitalize()
            else:
                replacement = replacement.lower()
            words[i] = replacement

        return ' '.join(words)


def rewrite_results_jsonl(input_path, output_path, editor):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for idx, line in tqdm(enumerate(infile, 1)):
            data = json.loads(line)
            original_text = data.get("result", "")
            paraphrased_text = editor.edit(original_text)
            data["result"] = paraphrased_text
            outfile.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()
    results_path = args.results_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    editor = ContextAwareSynonymSubstitution(ratio=0.3, tokenizer=tokenizer, model=model, device=device)

    for data in ['longform_qa']: #['finance_qa', 'longform_qa']:
        for model_name in ['phi']: #['phi', 'llama']: 
            for method in ['openai']: #['coupling', 'dipmark', 'maryland', 'openai']:
                for ngram in [2, 4]:
                    results_path = f"output/{data}/{model_name}/{method}/ngram_{ngram}/"
                    output_path = f"paraphrase/{data}/{model_name}/{method}/ngram_{ngram}/"
                    input_path = os.path.join(results_path, "results.jsonl")
                    output_path = os.path.join(output_path, "results.jsonl")
                    if os.path.exists(input_path):
                        rewrite_results_jsonl(input_path, output_path, editor)