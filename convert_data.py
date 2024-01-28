import json
import argparse

parser = argparse.ArgumentParser('Args', add_help=False)
parser.add_argument('--prompt_path', type=str, default="data/longform_qa")
args = parser.parse_args()

# Initialize an empty list to hold all JSON objects
json_list = []

# Open the .jsonl file and read line by line
with open(f'{args.prompt_path}.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        # Parse the JSON object and add it to the list
        json_list.append(json.loads(line))

# Convert the list to a JSON array
json_array = json.dumps(json_list, indent=4)

# Save the JSON array to a new .json file
with open(f'{args.prompt_path}.json', 'w') as json_file:
    json_file.write(json_array)
