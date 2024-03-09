import pandas as pd
import argparse

parser = argparse.ArgumentParser('Args', add_help=False)
parser.add_argument('--scores_path', type=str, default="output/importance")
parser.add_argument('--alpha', type=float, default=0.0001)
args = parser.parse_args()

df = pd.read_json(f'{args.scores_path}/scores.jsonl', lines=True)

TPR = sum(df['pvalue'] < args.alpha) / len(df)
print(f'TPR: {TPR}')