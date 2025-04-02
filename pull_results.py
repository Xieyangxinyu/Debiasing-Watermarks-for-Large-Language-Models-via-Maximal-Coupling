import os
import json
import pandas as pd

'''
Please first specify the model and dataset you want to analyze.
The models and datasets are defined in the dictionaries below.
'''

models = {
    #'phi': 'Phi-3-mini-4k-instruct (3.8B)',
    'llama': 'Llama-3.2-1B-Instruct'
}

datasets = {
    'finance_qa': 'FinQA',
    'longform_qa': 'ELI5'
}

def read_summary_file(filepath, method = 'coupling'):
    """Read and parse summary.txt file"""
    results = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Find the 'mean' and '50%' lines for SBERT scores
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == 'mean':
                results['mean_sbert'] = float(parts[-1])  # Last value is score_sbert
                results['num_tokens'] = float(parts[2])
            elif parts and parts[0] == '50%':
                results['median_sbert'] = float(parts[-1])  # Last value is score_sbert
            elif line.startswith('TPR:'):
                results['tpr'] = float(line.split(': ')[1])
            elif line.startswith('{'):
                data = json.loads(line)
                if f'\"{method}\"' in line:
                    results['tpr_aug'] = data['TPR']
                elif f'\"coupling-max\"' in line:
                    if 'tok_substitution' in line:
                        results['tpr_aug_max'] = data['TPR']
                    else:
                        results['tpr_max'] = data['TPR']
                elif f'\"coupling-HC\"' in line:
                    if 'tok_substitution' in line:
                        results['tpr_aug_hc'] = data['TPR']
                    else:
                        results['tpr_hc'] = data['TPR']
    return results

def get_results(base_path, model, dataset, method, ngram):
    """Get results for specific configuration"""
    path = os.path.join(base_path, dataset, model, method, f'ngram_{ngram}', 'summary.txt')
    if os.path.exists(path):
        return read_summary_file(path, method)
    return None

def format_number(num):
    """Format number to 4 decimal places"""
    return f"{num:.4f}"

def generate_main_result_latex_table():
    """Generate LaTeX table comparing results across different watermarking methods"""
    # Method names mapped to their display names
    methods = {
        'openai': 'Gumbel-max',
        'maryland': 'Green/red list',
        'dipmark': 'DiPmark',
        'coupling': 'Ours'
    }
    
    table = []
    table.append(r"\begin{table*}[t]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of the S-BERT similarity scores and the true positive rates (TPR) among three watermarking schemes: (1) the Gumbel-max watermark \citep{aaronson}, (2) the green/red list watermark \citep{kirchenbauer2023watermark}, (3) the DiPmark method \cite{wu2023dipmark} and (4) our proposed one. }")
    table.append(r"\label{tab: S-BERT and TPR, alpha 0.01, main}")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{c c c l l | r | r | r | r}")
    table.append(r"\toprule")
    table.append(r"Model & Data & $k$ & Metric & & Gumbel-max & Green/red list & DiPmark & Ours \\")
    table.append(r"\midrule")

    for model in models:
        model_display = models[model]
        first_model_row = True
        
        for dataset in datasets:
            dataset_display = datasets[dataset]
            first_dataset_row = True
            
            for k in [2, 4]:
                # Get all method results
                method_results = {}
                for method in methods:
                    results = get_results("output", model, dataset, method, k)
                    if results:
                        method_results[method] = results
                
                if not method_results:
                    continue
                
                # Find best TPR and TPR aug values for bold formatting
                tpr_values = {m: method_results[m].get('tpr', 0) for m in method_results}
                tpr_aug_values = {m: method_results[m].get('tpr_aug', 0) for m in method_results}
                
                best_tpr = max(tpr_values.values())
                best_tpr_aug = max(tpr_aug_values.values())
                
                # S-BERT mean row
                row = []
                if first_model_row:
                    # Calculate number of rows for this model (datasets * k values * 4 metrics per combination)
                    total_rows = len(datasets) * 2 * 4  # 2 k values, 4 metrics
                    row.append(fr"\multirow{{{total_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{model_display}}}}}")
                    first_model_row = False
                else:
                    row.append("")
                
                if first_dataset_row:
                    # 8 rows per dataset (2 k values * 4 metrics)
                    row.append(fr"\multirow{{8}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_display}}}}}")
                    first_dataset_row = False
                else:
                    row.append("")
                
                # k value with multirow for 4 metrics
                row.append(fr"\multirow{{4}}{{*}}{{{k}}}")
                
                # S-BERT with multirow for 2 rows (mean and median)
                row.append(r"\multirow{2}{*}{S-BERT}")
                
                # Mean
                row.append("mean")
                
                # Add values for each method
                for method_key in ['openai', 'maryland', 'dipmark', 'coupling']:
                    results = method_results.get(method_key, {})
                    mean_sbert = results.get('mean_sbert', 0)
                    row.append(format_number(mean_sbert))
                
                table.append(" & ".join(row) + r" \\")
                
                # S-BERT median row (no multirow cells)
                row = ["", "", "", "", "median"]
                for method_key in ['openai', 'maryland', 'dipmark', 'coupling']:
                    results = method_results.get(method_key, {})
                    median_sbert = results.get('median_sbert', 0)
                    row.append(format_number(median_sbert))
                
                table.append(" & ".join(row) + r" \\")
                
                # TPR row
                row = ["", "", "", "TPR", ""]
                for method_key in ['openai', 'maryland', 'dipmark', 'coupling']:
                    results = method_results.get(method_key, {})
                    tpr = results.get('tpr', 0)
                    row.append(format_number(tpr))
                
                table.append(" & ".join(row) + r" \\")
                
                # TPR aug row
                row = ["", "", "", "TPR aug.", ""]
                for method_key in ['openai', 'maryland', 'dipmark', 'coupling']:
                    results = method_results.get(method_key, {})
                    tpr_aug = results.get('tpr_aug', 0)
                    row.append(format_number(tpr_aug))
                
                table.append(" & ".join(row) + r" \\")
                
                # Add separator between k values
                if k == 2:
                    table.append(r"\cmidrule{3-9}")
            
            # Add separator between datasets
            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-9}")
    
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\end{table*}")
    
    return "\n".join(table)

def generate_proportion_table():
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv('analysis_results/entropy_summary.csv')
    
    # Map method names to their display names (in desired order)
    methods = {
        'openai': 'Gumbel-max',
        'maryland': 'Green/red list',
        'dipmark': 'DiPMark',
        'coupling': 'Ours'
    }
    
    table = []
    table.append(r"\begin{table}[t]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparative analysis of repeated token proportions across different watermarking methods.}")
    table.append(r"\label{tab:token_stats}")
    table.append(r"\centering")
    table.append(r"\begin{small}")
    
    # Table header
    table.append(r"\begin{tabular}{l l l l | r}")
    table.append(r"\toprule")
    table.append(r"Model & Dataset & k & Method & Repeated (\%) \\")
    table.append(r"\midrule")
    
    for model in models:
        first_model_row = True
        for dataset in datasets:
            first_dataset_row = True
            for k in [2, 4]:
                first_k_row = True
                for method in methods:
                    # Get the corresponding row from DataFrame
                    row_data = df[(df['model'] == model) & 
                                (df['dataset'] == dataset) & 
                                (df['ngram'] == k) & 
                                (df['method'] == method)]
                    
                    if row_data.empty:
                        continue
                        
                    # Build table row
                    table_row = []
                    
                    # Handle model column with rotated text
                    if first_model_row:
                        table_row.append(fr"\multirow{{16}}{{*}}{{\rotatebox[origin=c]{{90}}{{{models[model]}}}}}")
                        first_model_row = False
                    else:
                        table_row.append("")
                    
                    # Handle dataset column with multirow
                    if first_dataset_row:
                        table_row.append(fr"\multirow{{8}}{{*}}{{{datasets[dataset]}}}")
                        first_dataset_row = False
                    else:
                        table_row.append("")
                    
                    # Add k value with multirow
                    if first_k_row:
                        table_row.append(fr"\multirow{{4}}{{*}}{{{k}}}")
                        first_k_row = False
                    else:
                        table_row.append("")
                    
                    # Method name
                    table_row.append(methods[method])
                    
                    # Get proportion of repeated tokens
                    repeated_mean = row_data['proportion_repeated_tokens'].iloc[0]
                    table_row.append(f"{repeated_mean * 100:.2f}\%")
                    
                    table.append(" & ".join(table_row) + r" \\")
                    
                # Add separator between methods except for last group
                if k != 4 or method != list(methods.keys())[-1]:
                    table.append(r"\cmidrule{3-5}")
                if k == 4 and method == list(methods.keys())[-1] and dataset != list(datasets.keys())[-1]:
                    table.append(r"\cmidrule{2-5}")
            
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\end{table}")
    
    return "\n".join(table)

def format_bold_number(num, is_best=False):
    """Format number with optional bold formatting"""
    formatted = f"{num:.4f}".rstrip('0').rstrip('.')
    return f"\\textbf{{{formatted}}}" if is_best else formatted

def generate_stats_table():
    
    table = []
    table.append(r"\begin{table*}[hbt!]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of different test statistics for our watermarking scheme. Num. Tokens denotes the average number of tokens generated by the language model across all samples. The test statistics are the maximum, the sum, and the higher criticism. The sum statistic is more powerful than the maximum statistic, and the higher criticism statistic.}")
    table.append(r"\label{tab:stats}")
    table.append(r"\centering")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{c c c l l | r | r | r}")
    table.append(r"\toprule")
    table.append(r"Model & Data & $k$/Num. Tokens & Metric & Max & Sum & HC$^+$ \\")
    table.append(r"\midrule")

    for model in models:
        first_model_row = True
        for dataset in datasets:
            first_dataset_row = True
            for k in [2, 4]:
                # Get results
                results = get_results("output", model, dataset, "coupling", k)
                if not results:
                    continue

                # First row (TPR)
                row = []
                if first_model_row:
                    row.append(fr"\multirow{{8}}{{*}}{{{models[model]}}}")
                    first_model_row = False
                else:
                    row.append("")

                if first_dataset_row:
                    row.append(fr"\multirow{{4}}{{*}}{{{datasets[dataset]}}}")
                    first_dataset_row = False
                else:
                    row.append("")

                # Format num_tokens
                num_tokens = results.get('num_tokens', 0)
                row.append(fr"\multirow{{2}}{{*}}{{{k}/{num_tokens:.3f}}}")
                
                row.append("TPR")
                
                # Add TPR values with bold formatting for highest
                tpr_values = [
                    results.get('tpr_max', 0),
                    results.get('tpr', 0),
                    results.get('tpr_hc', 0)
                ]
                max_tpr = max(tpr_values)
                row.extend([
                    format_bold_number(val, val == max_tpr)
                    for val in tpr_values
                ])
                
                table.append(" & ".join(row) + r" \\")
                
                # Second row (TPR aug.)
                row = ["", "", "", "TPR aug."]
                tpr_aug_values = [
                    results.get('tpr_aug_max', 0),
                    results.get('tpr_aug', 0),
                    results.get('tpr_aug_hc', 0)
                ]
                max_tpr_aug = max(tpr_aug_values)
                row.extend([
                    format_bold_number(val, val == max_tpr_aug)
                    for val in tpr_aug_values
                ])
                
                table.append(" & ".join(row) + r" \\")
                
                if k != 4:
                    table.append(r"\cmidrule{3-7}")
            
            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-7}")
    
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\vspace{-0.2in}")
    table.append(r"\end{table*}")
    
    return "\n".join(table)

def generate_stats_no_max_table():
    
    table = []
    table.append(r"\begin{table*}[hbt!]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of different test statistics for our watermarking scheme. Num. Tokens denotes the average number of tokens generated by the language model across all samples. The test statistics are the sum and the higher criticism. The sum statistic is more powerful than the higher criticism statistic.}")
    table.append(r"\label{tab:stats}")
    table.append(r"\centering")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{c c c l | r | r}")
    table.append(r"\toprule")
    table.append(r"Model & Data & $k$/Num. Tokens & Metric & Sum & HC$^+$ \\")
    table.append(r"\midrule")

    for model in models:
        first_model_row = True
        for dataset in datasets:
            first_dataset_row = True
            for k in [2, 4]:
                # Get results
                results = get_results("output", model, dataset, "coupling", k)
                if not results:
                    continue

                # First row (TPR)
                row = []
                if first_model_row:
                    row.append(fr"\multirow{{8}}{{*}}{{{models[model]}}}")
                    first_model_row = False
                else:
                    row.append("")

                if first_dataset_row:
                    row.append(fr"\multirow{{4}}{{*}}{{{datasets[dataset]}}}")
                    first_dataset_row = False
                else:
                    row.append("")

                # Format num_tokens
                num_tokens = results.get('num_tokens', 0)
                row.append(fr"\multirow{{2}}{{*}}{{{k}/{num_tokens:.3f}}}")
                
                row.append("TPR")
                
                # Add TPR values with bold formatting for highest
                tpr_values = [
                    results.get('tpr', 0),
                    results.get('tpr_hc', 0)
                ]
                max_tpr = max(tpr_values)
                row.extend([
                    format_bold_number(val, val == max_tpr)
                    for val in tpr_values
                ])
                
                table.append(" & ".join(row) + r" \\")
                
                # Second row (TPR aug.)
                row = ["", "", "", "TPR aug."]
                tpr_aug_values = [
                    results.get('tpr_aug', 0),
                    results.get('tpr_aug_hc', 0)
                ]
                max_tpr_aug = max(tpr_aug_values)
                row.extend([
                    format_bold_number(val, val == max_tpr_aug)
                    for val in tpr_aug_values
                ])
                
                table.append(" & ".join(row) + r" \\")
                
                if k != 4:
                    table.append(r"\cmidrule{3-6}")
            
            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-6}")
    
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\vspace{-0.2in}")
    table.append(r"\end{table*}")
    
    return "\n".join(table)

def read_rejection_stats(filepath):
    """Read rejection statistics from rejection_statistics.txt"""
    results = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Mean Per-prompt Rejection Rate:'):
                    value = line.split(': ')[1].strip().rstrip('%')
                    results['mean_rejection'] = float(value)  # Keep as percentage
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}")
    return results

def generate_speculative_table():
    
    table = []
    table.append(r"\begin{table*}[hbt!]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of the true positive rates (TPR) and rejection rates between the Gumbel-max watermark \citep{aaronson} and our proposed scheme under the speculative decoding setup. By embedding a watermark only on the smaller draft model, we explore the robustness of watermarking techniques when subjected to targeted modifications.}")
    table.append(r"\label{tab:speculative_stats}")
    table.append(r"\centering")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{l l l l | r | r}")
    table.append(r"\toprule")
    table.append(r"Model & Data & $k$ & Metric & Gumbel-max & Ours \\")
    table.append(r"\midrule")

    for model in models:
        first_model_row = True
        for dataset in datasets:
            first_dataset_row = True
            for k in [2, 4]:
                # Get results for both methods from their respective files
                our_rejection = read_rejection_stats(f"speculative/{dataset}/{model}/coupling/ngram_{k}/rejection_statistics.txt")
                our_tpr = get_results("speculative", model, dataset, "coupling", k)
                
                gumbel_rejection = read_rejection_stats(f"speculative/{dataset}/{model}/openai/ngram_{k}/rejection_statistics.txt")
                gumbel_tpr = get_results("speculative", model, dataset, "openai", k)
                
                if not our_rejection or not our_tpr or not gumbel_rejection or not gumbel_tpr:
                    continue

                # First row (Rejection Rate)
                row = []
                if first_model_row:
                    row.append(fr"\multirow{{8}}{{*}}{{{models[model]}}}")
                    first_model_row = False
                else:
                    row.append("")

                if first_dataset_row:
                    row.append(fr"\multirow{{4}}{{*}}{{{datasets[dataset]}}}")
                    first_dataset_row = False
                else:
                    row.append("")

                row.append(fr"\multirow{{2}}{{*}}{{{k}}}")
                row.append("Avg. Rejection Rate")
                row.append(format_number(gumbel_rejection['mean_rejection']) + r"\%")
                row.append(format_number(our_rejection['mean_rejection']) + r"\%")
                
                table.append(" & ".join(row) + r" \\")
                
                # Second row (TPR)
                row = ["", "", "", "TPR"]
                row.append(format_number(gumbel_tpr['tpr']))
                row.append(format_number(our_tpr['tpr']))
                
                table.append(" & ".join(row) + r" \\")
                
                if k != 4:
                    table.append(r"\cmidrule{3-6}")
            
            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-6}")
    
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\vspace{-0.2in}")
    table.append(r"\end{table*}")
    
    return "\n".join(table)

def read_entropy_stats(filepath):
    """Read entropy statistics from file"""
    results = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            in_watermarked = False
            in_repeated = False
            for line in lines:
                if "Watermarked Positions:" in line:
                    in_watermarked = True
                    in_repeated = False
                    continue
                elif "Repeated Positions:" in line:
                    in_watermarked = False
                    in_repeated = True
                    continue
                if line.strip():
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = float(parts[1].strip())
                        if in_watermarked:
                            results[f'watermarked_{key.lower().replace(" ", "_")}'] = value
                        elif in_repeated:
                            results[f'repeated_{key.lower().replace(" ", "_")}'] = value
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}")
    return results

def calculate_repeated_percentage(results):
    """Calculate percentage of tokens that are masked"""
    total_tokens = results.get('watermarked_total_tokens', 0) + results.get('repeated_total_tokens', 0)
    if total_tokens == 0:
        return 0
    return (results.get('repeated_total_tokens', 0) / total_tokens) * 100


def generate_entropy_table():
    methods = {
        'openai': 'Gumbel-max',
        'maryland': 'Green/red list',
        'dipmark': 'DiPMark',
        'coupling': 'Ours'
    }
    
    k_values = [2, 4]
    
    table = []
    table.append(r"\begin{table}[hbt!]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparative analysis of token entropy across different watermarking methods for k=2 and k=4. For each method, we analyze both watermarked and repeated positions, showing mean entropy and token counts.}")
    table.append(r"\label{tab:entropy_analysis}")
    table.append(r"\centering")
    table.append(r"\begin{small}")
    
    # Calculate total rows per model for proper multirow
    rows_per_dataset = len(methods) * 2  # 2 for watermarked and repeated
    rows_per_model = rows_per_dataset * len(datasets)
    total_rows_per_model = rows_per_model * len(k_values)  # Multiply by number of k values
    
    # Table header with adjusted alignment
    table.append(r"\begin{tabular}{@{}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{\hspace{4pt}}l@{\hspace{8pt}}rrr@{}}")
    table.append(r"\toprule")
    table.append(r"\textbf{Model} & \textbf{Data} & \textbf{Method} & \textbf{k} & \textbf{Type} & \textbf{Mean} & \textbf{Total} & \textbf{Repeated} \\")
    table.append(r"& & & & & \textbf{Entropy} & \textbf{Tokens} & \textbf{(\%)} \\")
    table.append(r"\midrule")

    for i, model in enumerate(models):
        for k in k_values:
            for j, dataset in enumerate(datasets):
                # Get all valid results first
                valid_methods = []
                for method in methods:
                    filepath = f"output/{dataset}/{model}/{method}/ngram_{k}/entropy_stats.txt"
                    results = read_entropy_stats(filepath)
                    if results:
                        valid_methods.append((method, results))
                
                if not valid_methods:
                    continue
                    
                rows_this_dataset = len(valid_methods) * 2
                
                for m, (method, results) in enumerate(valid_methods):
                    mask_percentage = calculate_repeated_percentage(results)
                    
                    # First row of the method (Watermarked)
                    row = []
                    
                    # Model column
                    if m == 0 and j == 0 and k == k_values[0]:
                        # rotate model name for first row of each model
                        row.append(fr"\multirow{{{total_rows_per_model}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{models[model]}}}}}")
                    else:
                        row.append("")
                    
                    # Dataset column
                    if m == 0:
                        row.append(fr"\multirow{{{rows_this_dataset}}}{{*}}{{{datasets[dataset]}}}")
                    else:
                        row.append("")
                    
                    # Method column
                    row.append(fr"\multirow{{2}}{{*}}{{{methods[method]}}}")
                    
                    # k column
                    row.append(fr"\multirow{{2}}{{*}}{{{k}}}")
                    
                    # Type column for Watermarked
                    row.append("Watermarked")
                    
                    # Data columns for Watermarked
                    row.append(format_number(results['watermarked_mean_entropy']))
                    row.append(str(int(results['watermarked_total_tokens'])))
                    row.append(f"{mask_percentage:.2f}")  # Add mask percentage
                    table.append(" & ".join(row) + r" \\")
                    
                    # Second row of the method (Repeated)
                    row = ["", "", "", ""]  # Empty for model, dataset, method, and k
                    row.append("Repeated")  # Type column for Repeated
                    row.append(format_number(results['repeated_mean_entropy']))
                    row.append(str(int(results['repeated_total_tokens'])))
                    row.append("")  # Empty cell for mask percentage in repeated row
                    table.append(" & ".join(row) + r" \\")
                    
                    # Add separator between methods
                    if m < len(valid_methods) - 1:
                        table.append(r"\cmidrule{3-8}")
                
                # Add separator between datasets unless it's the last dataset
                if j < len(datasets) - 1 or k < k_values[-1]:
                    table.append(r"\cmidrule{2-8}")
            
        
        # Add separator between models unless it's the last model
        if i < len(models) - 1:
            table.append(r"\midrule")
    
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\end{table}")
    
    return "\n".join(table)

def read_kirchenbauer_file(model, dataset, ngram, delta, gamma):
    """Read summary file for Kirchenbauer method with specific delta and gamma"""
    filepath = os.path.join('more_experiments', dataset, model, 'maryland', f'ngram_{ngram}', 
                           f'delta_{delta}', f'gamma_{gamma}', 
                           'summary.txt')
    # Special case for gamma=0.5, delta=1
    if gamma == 0.5 and delta == 1:
        filepath = os.path.join('output', dataset, model, 'maryland', f'ngram_{ngram}', 'summary.txt')
    
    if os.path.exists(filepath):
        return read_summary_file(filepath, 'maryland')
    return None

def generate_more_experiments_table(model = 'phi'):
    table = []
    table.append(r"\begin{table}")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of the S-BERT similarity scores and the true positive rates (TPR) of the two watermarking schemes: Kirchenbauer \cite{kirchenbauer2023watermark} and our proposed scheme for the " + models[model] + r". The significance level is set to $\alpha = 0.01$. There is a positive correlation between the distortion of the generation and the detection power. When $\gamma = 0.5$, i.e., each green list contains half of the vocabulary, the detection power is competitive when similar generation quality is achieved, as indicated by the S-BERT metric.}")
    table.append(r"\label{table:" + model + r"}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{l l l l | r r | r r | r}")
    table.append(r"\toprule")
    table.append(r"\multirow{3}{*}{\rotatebox[origin=c]{90}{Data}} & \multirow{3}{*}{k} & \multirow{3}{*}{Metric} & & \multicolumn{4}{c}{Kirchenbauer} & \multirow{2}{*}{Our method} \\")
    table.append(r"&&&& \multicolumn{2}{c}{$\delta = 2$} & \multicolumn{2}{c}{$\delta = 1$} & \\")
    table.append(r"&&&& $\gamma = 0.25$ & $\gamma = 0.5$ & $\gamma = 0.25$ & $\gamma = 0.5$ & $\gamma = 0.5$\\")
    table.append(r"\midrule")

    ks = [2, 4]

    for dataset, dataset_name in datasets.items():
        first_dataset = True
        for k in ks:
            # Get results for "Our method"
            our_results = get_results("output", model, dataset, "coupling", k)
            
            # Get results for Kirchenbauer method
            kirchenbauer_results = {
                (2, 0.25): read_kirchenbauer_file(model, dataset, k, 2, 0.25),
                (2, 0.5): read_kirchenbauer_file(model, dataset, k, 2, 0.5),
                (1, 0.25): read_kirchenbauer_file(model, dataset, k, 1, 0.25),
                (1, 0.5): read_kirchenbauer_file(model, dataset, k, 1, 0.5),
            }
            
            # S-BERT mean row
            row = []
            if first_dataset:
                row.append(fr"\multirow{{8}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_name}}}}}")
                first_dataset = False
            else:
                row.append("")
            
            row.append(fr"\multirow{{4}}{{*}}{{{k}}}")
            row.append(r"\multirow{2}{*}{S-BERT}")
            row.append("mean")
            
            # Add Kirchenbauer results
            for delta, gamma in [(2, 0.25), (2, 0.5), (1, 0.25), (1, 0.5)]:
                results = kirchenbauer_results.get((delta, gamma))
                row.append(format_number(results['mean_sbert']) if results else "-")
            
            # Add our method result
            row.append(format_number(our_results['mean_sbert']) if our_results else "-")
            
            table.append(" & ".join(row) + r" \\")
            
            # S-BERT median row
            row = ["", "", "", "median"]
            for delta, gamma in [(2, 0.25), (2, 0.5), (1, 0.25), (1, 0.5)]:
                results = kirchenbauer_results.get((delta, gamma))
                row.append(format_number(results['median_sbert']) if results else "-")
            row.append(format_number(our_results['median_sbert']) if our_results else "-")
            table.append(" & ".join(row) + r" \\")
            
            # TPR row
            row = ["", "", "TPR", ""]
            for delta, gamma in [(2, 0.25), (2, 0.5), (1, 0.25), (1, 0.5)]:
                results = kirchenbauer_results.get((delta, gamma))
                row.append(format_number(results['tpr']) if results else "-")
            row.append(format_number(our_results['tpr']) if our_results else "-")
            table.append(" & ".join(row) + r" \\")
            
            # TPR aug row
            row = ["", "", "TPR aug.", ""]
            for delta, gamma in [(2, 0.25), (2, 0.5), (1, 0.25), (1, 0.5)]:
                results = kirchenbauer_results.get((delta, gamma))
                row.append(format_number(results['tpr_aug']) if results else "-")
            row.append(format_number(our_results['tpr_aug']) if our_results else "-")
            table.append(" & ".join(row) + r" \\")
            
            if k != ks[-1]:
                table.append(r"\cmidrule{3-9}")
        
        if dataset != list(datasets.keys())[-1]:
            table.append(r"\cmidrule{2-9}")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{table}")
    
    return "\n".join(table)


def read_onelist_results(model, dataset, k):
    """Read results from one_list directory"""
    path = os.path.join('one_list', dataset, model, 'coupling', f'ngram_{k}', 'summary.txt')
    if os.path.exists(path):
        return read_summary_file(path)
    return None

def generate_list_table():
    table = []
    table.append(r"\begin{table*}[hbt!]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of the watermarking scheme with a single list and the watermarking scheme with disparate lists. The true positive rate (TPR) and the true positive rate after the substitution attack (TPR aug.) are reported in the form of (Sum statistic/HC$^+$ statistic). Having a single green list does not significantly impact our watermarking scheme when a larger context window is used to generate $\zeta$.}")
    table.append(r"\label{tab: one list}")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    
    # Table header
    table.append(r"\begin{tabular}{l l l l l | r r}")
    table.append(r"\toprule")
    table.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Data} & \multirow{2}{*}{k} & \multirow{2}{*}{Metric} & & \multicolumn{2}{c}{coupling} \\")
    table.append(r"&&&&& single list & disparate lists\\")
    table.append(r"\midrule")

    ks = [2, 4]

    for model, model_name in models.items():
        first_model = True
        for dataset, dataset_name in datasets.items():
            first_dataset = True
            for k in ks:
                # Get results
                onelist_results = read_onelist_results(model, dataset, k)
                disparate_results = get_results("output", model, dataset, "coupling", k)

                # S-BERT mean row
                row = []
                if first_model:
                    row.append(fr"\multirow{{16}}{{*}}{{\rotatebox[origin=c]{{90}}{{{model_name}}}}}")
                    first_model = False
                else:
                    row.append("")
                
                if first_dataset:
                    row.append(fr"\multirow{{8}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_name}}}}}")
                    first_dataset = False
                else:
                    row.append("")
                
                row.append(fr"\multirow{{4}}{{*}}{{{k}}}")
                row.append(r"\multirow{2}{*}{S-BERT}")
                row.append("mean")
                
                # Add results with bold formatting for better value
                mean1 = onelist_results['mean_sbert'] if onelist_results else 0
                mean2 = disparate_results['mean_sbert'] if disparate_results else 0
                row.append(format_bold_number(mean1, mean1 >= mean2))
                row.append(format_bold_number(mean2, mean2 >= mean1))
                
                table.append(" & ".join(row) + r" \\")
                
                # S-BERT median row
                row = ["", "", "", "", "median"]
                median1 = onelist_results['median_sbert'] if onelist_results else 0
                median2 = disparate_results['median_sbert'] if disparate_results else 0
                row.append(format_bold_number(median1, median1 >= median2))
                row.append(format_bold_number(median2, median2 >= median1))
                table.append(" & ".join(row) + r" \\")
                
                # TPR row
                row = ["", "", "", "TPR", ""]
                tpr1 = f"{format_bold_number(onelist_results['tpr'], onelist_results['tpr'] >= disparate_results['tpr'])}/{format_bold_number(onelist_results['tpr_hc'], onelist_results['tpr_hc'] >= disparate_results['tpr_hc'])}" if onelist_results else "-/-"
                          
                tpr2 = f"{format_bold_number(disparate_results['tpr'], disparate_results['tpr'] >= onelist_results['tpr'])}/{format_bold_number(disparate_results['tpr_hc'], disparate_results['tpr_hc'] >= onelist_results['tpr_hc'])}" if disparate_results else "-/-"
                row.append(tpr1)
                row.append(tpr2)
                table.append(" & ".join(row) + r" \\")
                
                # TPR aug row
                row = ["", "", "", "TPR aug.", ""]
                tpr_aug1 = f"{format_bold_number(onelist_results['tpr_aug'], onelist_results['tpr_aug'] >= disparate_results['tpr_aug'])}/{format_bold_number(onelist_results['tpr_aug_hc'], onelist_results['tpr_aug_hc'] >= disparate_results['tpr_aug_hc'])}" if onelist_results else "-/-"
                tpr_aug2 = f"{format_bold_number(disparate_results['tpr_aug'], disparate_results['tpr_aug'] >= onelist_results['tpr_aug'])}/{format_bold_number(disparate_results['tpr_aug_hc'], disparate_results['tpr_aug_hc'] >= onelist_results['tpr_aug_hc'])}" if disparate_results else "-/-"
                              
                row.append(tpr_aug1)
                row.append(tpr_aug2)
                table.append(" & ".join(row) + r" \\")
                
                if k != ks[-1]:
                    table.append(r"\cmidrule{3-7}")
            
            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-7}")
        
        if model != list(models.keys())[-1]:
            table.append(r"\midrule")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\end{table*}")
    
    return "\n".join(table)

def generate_combined_result_latex_table():
    """Generate LaTeX table comparing all metrics including paraphrasing TPR, with best TPR values highlighted (excluding OpenAI)"""
    methods = {
        'openai': 'Gumbel-max',
        'maryland': 'Green/red list',
        'dipmark': 'DiPmark',
        'coupling': 'Ours'
    }

    table = []
    table.append(r"\begin{table*}[t]")
    table.append(r"\captionsetup{font={stretch=1}}")
    table.append(r"\renewcommand{\arraystretch}{0.8}")
    table.append(r"\caption{Comparison of the S-BERT similarity scores and the true positive rates (TPR) among four watermarking schemes: (1) the Gumbel-max watermark \citep{aaronson}, (2) the green/red list watermark \citep{kirchenbauer2023watermark}, (3) the DiPmark method \cite{wu2023dipmark}, and (4) our proposed one.}")
    table.append(r"\label{tab: S-BERT and TPR, alpha 0.01, main}")
    table.append(r"\begin{small}")
    table.append(r"\centering")
    table.append(r"\begin{tabular}{c c c l l | r | r | r | r}")
    table.append(r"\toprule")
    table.append(r"Model & Data & $k$ & Metric & & Gumbel-max & Green/red list & DiPmark & Ours \\")
    table.append(r"\midrule")

    for model in models:
        model_display = models[model]
        first_model_row = True

        for dataset in datasets:
            dataset_display = datasets[dataset]
            first_dataset_row = True

            for k in [2, 4]:
                main_results = {
                    method: get_results("output", model, dataset, method, k)
                    for method in methods
                }
                para_results = {
                    method: get_results("paraphrase", model, dataset, method, k)
                    for method in methods
                }

                if not any(main_results.values()):
                    continue

                # Row: S-BERT mean
                row = []
                if first_model_row:
                    total_rows = len(datasets) * 2 * 5  # 5 rows per k
                    row.append(fr"\multirow{{{total_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{model_display}}}}}")
                    first_model_row = False
                else:
                    row.append("")
                
                if first_dataset_row:
                    row.append(fr"\multirow{{10}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_display}}}}}")
                    first_dataset_row = False
                else:
                    row.append("")

                row.append(fr"\multirow{{5}}{{*}}{{{k}}}")
                row += [r"\multirow{2}{*}{S-BERT}", "mean"]
                for method in methods:
                    value = main_results[method].get('mean_sbert', 0) if main_results[method] else 0
                    row.append(format_number(value))
                table.append(" & ".join(row) + r" \\")

                # Row: S-BERT median
                row = ["", "", "", "", "median"]
                for method in methods:
                    value = main_results[method].get('median_sbert', 0) if main_results[method] else 0
                    row.append(format_number(value))
                table.append(" & ".join(row) + r"\\")

                # Row: TPR (highlight best excluding OpenAI)
                row = ["", "", "", "TPR", ""]
                tpr_vals = {
                    m: main_results[m].get('tpr', 0)
                    for m in methods if m != 'openai' and main_results[m]
                }
                best_tpr = max(tpr_vals.values()) if tpr_vals else 0
                for method in methods:
                    value = main_results[method].get('tpr', 0) if main_results[method] else 0
                    if method != 'openai' and value == best_tpr:
                        row.append(rf"\textbf{{{format_number(value)}}}")
                    else:
                        row.append(format_number(value))
                table.append(" & ".join(row) + r"\\")

                # Row: TPR aug.
                row = ["", "", "", "TPR aug.", ""]
                tpr_aug_vals = {
                    m: main_results[m].get('tpr_aug', 0)
                    for m in methods if m != 'openai' and main_results[m]
                }
                best_tpr_aug = max(tpr_aug_vals.values()) if tpr_aug_vals else 0
                for method in methods:
                    value = main_results[method].get('tpr_aug', 0) if main_results[method] else 0
                    if method != 'openai' and value == best_tpr_aug:
                        row.append(rf"\textbf{{{format_number(value)}}}")
                    else:
                        row.append(format_number(value))
                table.append(" & ".join(row) + r"\\")

                # Row: TPR (para.)
                row = ["", "", "", "TPR para.", ""]
                tpr_para_vals = {
                    m: para_results[m].get('tpr_aug', 0)
                    for m in methods if m != 'openai' and para_results[m]
                }
                best_tpr_para = max(tpr_para_vals.values()) if tpr_para_vals else 0
                for method in methods:
                    value = para_results[method].get('tpr_aug', 0) if para_results[method] else 0
                    if method != 'openai' and value == best_tpr_para:
                        row.append(rf"\textbf{{{format_number(value)}}}")
                    else:
                        row.append(format_number(value))
                table.append(" & ".join(row) + r"\\")

                if k == 2:
                    table.append(r"\cmidrule{3-9}")

            if dataset != list(datasets.keys())[-1]:
                table.append(r"\cmidrule{2-9}")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{small}")
    table.append(r"\end{table*}")

    return "\n".join(table)

# Usage
# If you haven't run `paraphrase_attack.py`, uncomment the following line to generate Table 1 in the paper
# latex_table = generate_main_result_latex_table()
# print(latex_table)

# If you want to include the paraphrase attack results in the table, uncomment the following line
# latex_table = generate_combined_result_latex_table()
# print(latex_table)

# Usage
# uncomment the following line to generate Table 2 in the paper
# make sure you run `analyze_repetition.py` first
# proportion_table = generate_proportion_table()
# print(proportion_table)

# Usage
# uncomment the following line to generate Table 3 in the paper
# make sure you've run `speculative_watermark.py` and `analyze_speculative_cache.py` first
# speculative_table = generate_speculative_table()
# print(speculative_table)

# Usage 
# uncomment the following line to generate Table 4 in the paper
# no_max_table = generate_stats_no_max_table()
# print(no_max_table)

# Usage
# uncomment the following line to generate Table 6 (or 7) in the Appendix
# more_experiments_table = generate_more_experiments_table('phi')
# print(more_experiments_table)

# Usage
# uncomment the following line to generate Table 11 in the Appendix
# list_table = generate_list_table()
# print(list_table)