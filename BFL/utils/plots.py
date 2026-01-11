    
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np 
import pandas as pd
import subprocess 
import ast
from tabulate import tabulate
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns

########## Time Analysis

def get_times(command):
    log_path = subprocess.check_output(command, shell=True).decode('utf-8').strip().split('\n')
    times = []
    with open(log_path[0], 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'second' in line:
                time_line = line.strip()
                time_line = time_line.split('INFO   ')[1]
                time = time_line.split('second')[0]
                time = ast.literal_eval(time)
                times.append(time)
    return times
def get_times_all_seeds(dataset):
    seeds = ['0', '1', '2']
    times_dict_seeds = {seed : {} for seed in seeds}
    for seed in seeds :     
        methods = ['aalv', 'eaa', 'gaa', 'rkl', 'wb_diag']
        nbls = ['1', '2', '3']
        times_dict = {method: {nbl : (0,0)} for method in methods for nbl in nbls}
        for method in methods:
            for nbl in nbls:
                command = f"find '../logs/{dataset}/noniid-labeldir/BFLAVG/{method}/{seed}/{nbl}' -name '*.log'"
                times = get_times(command)
                times_dict[method][nbl] = (np.mean(times), np.std(times))

        command = f"find '../logs/{dataset}/noniid-labeldir/FedAVG/none/{seed}' -name '*.log'"
        times = get_times(command)
        times_dict['FedAVG'] = {'0': (np.mean(times), np.std(times))} 
        times_dict_seeds[seed] = times_dict
    return times_dict_seeds

def average_over_seeds_times(times_dict_seeds):
    # New dictionary to store averaged results
    averaged_results = {}

    # Iterate over each seed
    for seed, methods in times_dict_seeds.items():
        # Iterate over each method
        for method, nbls in methods.items():
            # Ensure the method is initialized in the new dictionary
            if method not in averaged_results:
                averaged_results[method] = {}
            # Iterate over each nbl
            for nbl, times in nbls.items():
                # Ensure the nbl is initialized for the method
                if nbl not in averaged_results[method]:
                    averaged_results[method][nbl] = {'total_avg_time': 0, 'total_std_time': 0, 'count': 0}
                # Accumulate the times and count
                averaged_results[method][nbl]['total_avg_time'] += times[0]
                averaged_results[method][nbl]['total_std_time'] += times[1]
                averaged_results[method][nbl]['count'] += 1

    # Calculate the averages
    for method, nbls in averaged_results.items():
        for nbl, data in nbls.items():
            count = data['count']
            data['avg_time'] = data['total_avg_time'] / count
            data['std_time'] = data['total_std_time'] / count
            # Remove the temporary keys
            del data['total_avg_time'], data['total_std_time'], data['count']

    return averaged_results

def print_times(dataset):
    times_dict_seeds = get_times_all_seeds(dataset)
    averaged_results = average_over_seeds_times(times_dict_seeds)
    # Create a list to store the table data
    table_data = []

    # Iterate over each method
    for method, nbls in averaged_results.items():
        # Iterate over each nbl
        for nbl, data in nbls.items():
            # Append the method, nbl, avg_time, and std_time to the table data
            table_data.append([f"{method}_{nbl}", data['avg_time'], data['std_time']])
            # Sort the table_data based on average time in ascending order
            table_data.sort(key=lambda x: x[1])

    print('Time per Communication Round')
    # Print the sorted table
    print(tabulate(table_data, headers=['Method', 'Avg', 'Std'], tablefmt='grid'))

##### Data Heterogeneity #####
def get_data_distribution(dataset,seed):
    command = f"find '../logs/{dataset}/noniid-labeldir/FedAVG/none/{seed}' -name '*.log'"
    log_path = subprocess.check_output(command, shell=True).decode('utf-8').strip().split('\n')
    #log_path = !find '../logs/{dataset}/noniid-labeldir/FedAVG/none/{seed}' -name "*.log" 
    with open(log_path[0], 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Train data statistics' in line:
                train_data_stats_line = line.strip()
                train_data_stats = ast.literal_eval(train_data_stats_line.split('statistics: ')[1])
    df = pd.DataFrame(train_data_stats)
    df.columns.name = "client"
    df.index.name = "classes"

    df = df.fillna(0)
    df = df.astype(int)
    return df

def plot_data_distribution(df):  # Plotting the heterogeneity between clients in terms of classes
    plt.figure(figsize=(10, 6))
    plt.pcolor(df, cmap='Blues')
    plt.colorbar(label='Number of Samples')
    #plt.title('Classes per clients')
    plt.xlabel('Client')
    plt.ylabel('Class')
    plt.yticks([i + 0.5 for i in range(df.shape[0])], df.index)
    plt.xticks([i + 0.5 for i in range(df.shape[1])], df.columns)
    #plt.savefig('data_distribution.pdf', format='pdf')
    plt.show()


##### Servers comparison #####
def plot_compare_servers(datasets:dict, score:str, dataset_name:str, ax):
    lines = []
    labels = []
    for algorithm, dataset in datasets.items(): 
        line = ax.plot(dataset[score], label=algorithm)
        lines.append(line[0])
        labels.append(algorithm)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(f'{score}')
    return lines, labels 

def plot_compare_servers_all_scores(datasets:dict, dataset_name:str):
    fig , axs = plt.subplots(1,3, figsize=(15,5))
    scores = ['Acc', 'ECE', 'NLL']
    lines = []  # List to store the Line2D objects for the legend
    labels = []  # List to store the labels for the legend

    for i, (ax , score) in enumerate(zip(axs,scores)): 
        line, label = plot_compare_servers(datasets, score, dataset_name, ax)
        lines += line
        if i == 0:  # Only add labels for the first subplot
            labels += label

    # Create a common legend
    fig.legend(lines, labels, loc='upper right')
    fig.suptitle(f'Servers on {dataset_name}')

    fig.tight_layout()



def plot_compare_servers_all_scores_lastround(datasets:dict, dataset_name:str):
    fig , axs = plt.subplots(1,3, figsize=(15,5))
    scores = ['Acc', 'ECE', 'NLL']

    # Generate a list of colors using a colormap
    colors = cm.viridis(np.linspace(0, 1, len(datasets)))

    for i, (ax , score) in enumerate(zip(axs,scores)): 
        results = {dataset: df[score].iloc[-1] for dataset, df in datasets.items()}

        for j, (dataset, result) in enumerate(results.items()):
            color = colors[j]
            bar = ax.bar(dataset, result, color=color, alpha=0.5)
            height = bar[0].get_height()
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

        # Remove x-axis labels
        ax.set_xticks([])

        # Set the title of the subplot at the bottom
        ax.set_xlabel(score)

    # Create a common legend
    legend_patches = [Patch(color=color, label=dataset) for dataset, color in zip(datasets.keys(), colors)]
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))

    # Set the title of the entire figure
    fig.suptitle(f'Servers on {dataset_name}')

    fig.tight_layout()
    plt.show()

def table_compare_servers_all_scores_lastround(datasets:dict, dataset_name:str):
    scores = ['Acc', 'ECE', 'NLL']
    results = {algorithm: {score: df[score].iloc[-1] for score in scores} for algorithm, df in datasets.items()}
    df = pd.DataFrame(results)
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '15pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('font-size', '13pt'), ('text-align', 'center')]}
    ]).background_gradient(cmap='Blues')
    styled_df.set_caption(f'Servers on {dataset_name}')
    plt.show()
    return styled_df

def print_best_n_for_each_metric(datasets: dict, dataset_name: str, n:float):
    scores = ['Acc', 'ECE', 'NLL']
    results = {algorithm: {score: df[score].iloc[-1] for score in scores} for algorithm, df in datasets.items()}
    df = pd.DataFrame(results).T
    for score in scores:
        print(f"Top {n} for {score} on {dataset_name}:")
        if score == 'Acc':
            top_three = df[score].nlargest(n)
        else:  # For 'ECE' and 'NLL', lower is better
            top_three = df[score].nsmallest(n)
        print(top_three)

##### Servers vs Locals comparison #####

def plot_compare_server_locals(df_server, list_datasets:list, score:str, dataset_name:str, algorithm:str, ax): 
    lines = []
    labels = []

    line = ax.plot(df_server[score], label='Server' , linestyle='--')
    lines.append(line[0])
    labels.append('Server')

    for i in range(10): 
        line = ax.plot(list_datasets[i][score], label=f'Client {i}' )
        lines.append(line[0])
        labels.append(f'Client {i}')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel(f'{score}')

    return lines, labels

def plot_compare_server_locals_all_scores(df_server, list_datasets:list, dataset_name:str, algorithm:str):
    fig , axs = plt.subplots(1,3, figsize=(15,5))
    scores = ['Acc', 'ECE', 'NLL']
    lines = []  # List to store the Line2D objects for the legend
    labels = []  # List to store the labels for the legend

    for i, (ax , score) in enumerate(zip(axs,scores)): 
        line, label = plot_compare_server_locals(df_server, list_datasets, score, dataset_name, algorithm, ax)
        lines += line
        if i == 0:  # Only add labels for the first subplot
            labels += label

    # Create a common legend
    fig.legend(lines, labels, loc='upper right')
    fig.suptitle(f'{algorithm} on {dataset_name}')

    fig.tight_layout()

def get_experiments(strings, pattern1, pattern2): 
    experiment_titles = []
    for string in strings:
        start = string.find(pattern1)
        end = string.find(pattern2)
        if start != -1 and end != -1 and start < end:
            extracted_part = string[start + len(pattern1):end]
            experiment_titles.append(extracted_part)

    experiment_titles = sorted(list(set(experiment_titles)))
    experiments = [(title, f'{title}_global')  for title in experiment_titles]
    return experiments

def plot_experiment(results, experiment_name, global_name, dataset, pattern1, length=7):
    experiment_results = {title[length:]: results[title] for title in results if title.startswith(pattern1) and experiment_name in title} 
    experiment_clients = [df for title, df in experiment_results.items() if 'client' in title]
    plot_compare_server_locals_all_scores(experiment_results[global_name], experiment_clients, dataset, experiment_name)



########### Fairness Metrics ############

def extract_accuracies(list_dfs, operation, weights=None):
    accuracies = [list_dfs[i]['Acc'][49] for i in range(10)]
    return operation(accuracies, weights=weights) if weights else operation(accuracies)

def std_variance(list_dfs, weights=None):
    accuracies = [list_dfs[i]['Acc'][49] for i in range(10)]
    mean = np.average(accuracies, weights=weights)
    std = np.sqrt( [(accuracies[i] - mean)**2 for i in range(10)])
    avg_std = np.sum([weights[i] * std[i] for i in range(10)]) 
    return avg_std

def JainFairness(list_dfs, weights=None):
    accuracies = [list_dfs[i]['Acc'][49] for i in range(10)]
    nom = np.sum([weight * acc for weight, acc in zip(weights, accuracies)])**2
    denom = len(accuracies) * np.sum([(weight * acc)**2 for weight, acc in zip(weights, accuracies)])
    return nom/denom 

def entropy(list_dfs):   
    accuracies = [list_dfs[i]['Acc'][49] for i in range(10)]
    sum_acc = np.sum(accuracies)
    return -np.sum([ (acc/sum_acc) * np.log(acc/sum_acc) for acc in accuracies ])


def cosine_similarity(list_dfs, weights=None):
    accuracies = [list_dfs[i]['Acc'][49] for i in range(10)]
    nom = np.average(accuracies, weights=weights)
    denom = np.sqrt(np.sum([ weight * acc**2 for weight,  acc in zip(weights, accuracies)]))
    return nom/denom


def plot_fairness_metrics(locals, weights, dataset_name):
    metrics = {
        'Mean Accuracies': lambda dfs: extract_accuracies(dfs, np.average, weights),
        'Worst 10% Accuracies': lambda dfs: extract_accuracies(dfs, np.min), 
        'Average Std': lambda dfs: std_variance(dfs, weights )
       # "Jain's Fairness Index": lambda dfs: JainFairness(dfs, weights) , 
       # 'Entropy': lambda dfs: entropy(dfs), 
       # 'Cosine Similarity': lambda dfs: cosine_similarity(dfs, weights)
    }
    fig, axs = plt.subplots(1,3, figsize=(15,8))
    # Create a list to store the Patch objects for the legend
    legend_patches = []
    for i, (title, operation) in enumerate(metrics.items()):
        results = {algo: operation(dfs) for algo, dfs in locals.items()}
        # Generate a list of colors using a colormap
        colors = cm.viridis(np.linspace(0, 1, len(results)))
        for j, (algo, result) in enumerate(results.items()):
            color = colors[j]
            bar = axs[i].bar(algo, result, color=color, alpha=0.5)
            height = bar[0].get_height()
            axs[i].text(bar[0].get_x() + bar[0].get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
            # Add a Patch object for this bar to the legend_patches list
            if i == 0:
                legend_patches.append(Patch(color=color, label=algo))
        # Remove x-axis labels
        axs[i].set_xticks([])
        axs[i].set_xlabel(title)
    # Create a common legend
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))
    # Set the title of the entire figure
    fig.suptitle(dataset_name)
    plt.show()


def table_fairness_metrics(locals, weights, dataset_name):
    metrics = {
        'Mean Accuracies': lambda dfs: extract_accuracies(dfs, np.average, weights),
        'Worst 10% Accuracies': lambda dfs: extract_accuracies(dfs, np.min), 
        'Average Std': lambda dfs: std_variance(dfs, weights )
        # "Jain's Fairness Index": lambda dfs: JainFairness(dfs, weights) , 
        # 'Entropy': lambda dfs: entropy(dfs), 
        # 'Cosine Similarity': lambda dfs: cosine_similarity(dfs, weights)
    }
    results = {algo: {title: operation(dfs) for title, operation in metrics.items()} for algo, dfs in locals.items()}
    df = pd.DataFrame(results)
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '15pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('font-size', '13pt'), ('text-align', 'center')]}
    ]).background_gradient(cmap='Blues')
    styled_df.set_caption(dataset_name)
    return styled_df

def print_best_n_fairness_metrics(locals, weights, dataset_name, n):
    metrics = {
        'Mean Accuracies': lambda dfs: extract_accuracies(dfs, np.average, weights),
        'Worst 10% Accuracies': lambda dfs: extract_accuracies(dfs, np.min), 
        'Average Std': lambda dfs: std_variance(dfs, weights )
        # "Jain's Fairness Index": lambda dfs: JainFairness(dfs, weights) , 
        # 'Entropy': lambda dfs: entropy(dfs), 
        # 'Cosine Similarity': lambda dfs: cosine_similarity(dfs, weights)
    }
    results = {algo: {title: operation(dfs) for title, operation in metrics.items()} for algo, dfs in locals.items()}
    df = pd.DataFrame(results).T
    for title in metrics.keys():
        print(f"Top {n} for {title} on {dataset_name}:")
        if title == 'Average Std':
            top_three = df[title].nsmallest(n)
        else:  # For 'Worst 10% Accuracies' and 'Mean Accuracies', higher is better
            top_three = df[title].nlargest(n)
        print(top_three)

#########################################################
def get_gog_results(results_path):
    global_on_global_paths = [path for path in open(results_path).read().split('\n') if 'global_log.csv' in path]
    titles = []
    # Extract the specific parts from each path
    for path in global_on_global_paths:
        parts = path.split('/')
        extracted_parts = [parts[i] for i in [4, 5, 6, 7] ]
        if 'FedAVG' in extracted_parts[0]: 
            extracted_parts[1] = '' 
            extracted_parts[-1] = ''
        titles.append('_'.join(extracted_parts))
        titles = [title[:-1] if title.endswith('_') else title for title in titles]
        titles = [title.replace('__', '_') for title in titles]

    title_path_dict = {title: path for title, path in zip(titles, global_on_global_paths)}
    gog_results = {title: pd.read_csv(path) for title, path in title_path_dict.items()} #golbal on global results
    return gog_results

# group by seed
def group_by_seed(gog_results): 
    grouped_gog_results_by_seeds = {}
    for title, result in gog_results.items():
        if title.split('_')[0] == 'BFLAVG':
            if 'wb_diag' in title: 
                seed = title.split('_')[3]
            else : 
                seed = title.split('_')[2]
        else : 
            seed = title.split('_')[1]
        if seed in grouped_gog_results_by_seeds:
            grouped_gog_results_by_seeds[seed][title] = result
        else:
            grouped_gog_results_by_seeds[seed] = {title: result}
    return grouped_gog_results_by_seeds

# group by seed and nbl
def grouped_gog_results_by_seeds_by_nbl(nbl, grouped_gog_results_by_seeds):
    grouped_gog_results_by_seeds_by_nbl = {}
    for seed, experiments in grouped_gog_results_by_seeds.items():
        grouped_gog_results_by_seeds_by_nbl[seed] = {}
        for title, result in experiments.items():
            if title.startswith('BFLAVG') :
                if nbl in title:
                    grouped_gog_results_by_seeds_by_nbl[seed][title] = result
            else: 
                grouped_gog_results_by_seeds_by_nbl[seed][title] = result
    return grouped_gog_results_by_seeds_by_nbl

def get_short_name(experiment_name):
    first_parts = experiment_name.split('_')[0:3]
    if first_parts[0] == 'FedAVG': 
        short_name = first_parts[0]
    elif first_parts[2] == 'diag':
        short_name = first_parts[1] + '_' + first_parts[2]
    else : 
        short_name = first_parts[1]
    return short_name

def plot_uq_vs_nbl(seed, grouped_gog_results_by_seeds): 
    scores = ['ECE', 'NLL']
    x = [1,2,3] # Number of Bayesian Layers
    fig, axs = plt.subplots(1, len(scores), figsize=(8, 2 * len(scores)))
    for i, score in enumerate(scores):
        # Group experiments by method name for a fixed seed 
        grouped_experiments = {}
        for experiment in grouped_gog_results_by_seeds[seed].keys():
            if 'FedAVG' in experiment:
                fedavg_score = grouped_gog_results_by_seeds[seed][experiment][score][49]
                continue
            method_name = experiment.split('_')[1]
            if method_name in grouped_experiments:
                grouped_experiments[method_name].append(grouped_gog_results_by_seeds[seed][experiment][score][49])
            else:
                grouped_experiments[method_name] = [grouped_gog_results_by_seeds[seed][experiment][score][49]]
        axs[i].plot(0, fedavg_score, 'x', label='FedAVG')
        for method_name, results in grouped_experiments.items():
            axs[i].plot(x, results, 'o-', label=method_name)
        axs[i].set_xticks([0,1,2,3])
        axs[i].set_xlabel('Number of Bayesian Layers') 
        axs[i].set_ylabel(f'{score} Score')

    # Collect labels and handles
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Check to avoid duplicates
                handles.append(handle)
                labels.append(label)

    fig.suptitle('Uncertainty Quantification Scores vs Number of Bayesian Layers')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def get_average_std_results_over_seeds(grouped_gog_results_by_seeds): 
    # Step 1: Transform the original dictionary
    transformed_dict = {}
    for seed, methods in grouped_gog_results_by_seeds.items():
        for method_seed_nbl, table_of_results in methods.items():
            # Assuming method_seed_nbl is a string like "method_seed_nbl"
            # and table_of_results is a list where the last element is the result of the last round
            last_round_result = table_of_results.iloc[-1]  # Assuming the result of the last round is the last element
            if 'FedAVG' in method_seed_nbl:
                method = 'FedAVG'
                nbl = '0'
            else :
                method = method_seed_nbl.split('_')[1]  # Adjust splitting logic as needed 
                nbl = method_seed_nbl.split('_')[-1]  # Adjust splitting logic as needed
            if nbl not in transformed_dict:
                transformed_dict[nbl] = {}
            if method not in transformed_dict[nbl]:
                transformed_dict[nbl][method] = {}
            transformed_dict[nbl][method][seed] = last_round_result

    # Step 2: Calculate average scores and std over seeds
    avg_scores_dict = {}
    std_dict = {}
    for nbl, methods in transformed_dict.items():
        avg_scores_dict[nbl] = {}
        std_dict[nbl] = {}
        for method, seeds_results in methods.items():
            scores = list(seeds_results.values())
            accs = [score['Acc'] for score in scores]
            eces = [score['ECE'] for score in scores]
            nlls = [score['NLL'] for score in scores]
            accs_std , accs_avg = np.std(accs) , np.mean(accs)
            eces_std , eces_avg = np.std(eces) , np.mean(eces)
            nlls_std , nlls_avg = np.std(nlls) , np.mean(nlls)
            std_dict[nbl][method] = {'Acc': accs_std, 'ECE': eces_std, 'NLL': nlls_std}
            avg_scores_dict[nbl][method] = {'Acc': accs_avg, 'ECE': eces_avg, 'NLL': nlls_avg}

    return avg_scores_dict, std_dict

def get_average_std_results_over_seeds_by_nbl(grouped_gog_results_by_seeds, nbl): 
    average_scores, std = get_average_std_results_over_seeds(grouped_gog_results_by_seeds)
    return average_scores[nbl], std[nbl]

def plot_avg_std_uq_vs_nbl( grouped_gog_results_by_seeds): 
    # Assuming get_average_std_results_over_seeds function is available and imported
    average_scores, std_scores = get_average_std_results_over_seeds(grouped_gog_results_by_seeds)

    # Ensure '0' is included in nbls and is the first element
    nbls = sorted(average_scores.keys(), key=lambda x: int(x))

    methods = set()
    for nbl in nbls:
        methods.update(average_scores[nbl].keys())

    # Prepare data for plotting (std)
    stds = { 'ECE': {}, 'NLL': {}}
    for method in methods:
        for score_type in stds:
            stds[score_type][method] = []

    # Initialize data structures for plotting
    scores = { 'ECE': {}, 'NLL': {}}
    for method in methods:
        for score_type in scores:
            scores[score_type][method] = []

    # Populate the data structures
    for nbl in nbls:
        for method in methods:
            if method != 'FedAVG' and nbl == '0':
                continue  # Skip '0' for non-'FedAVG' methods
            for score_type in scores:
                scores[score_type][method].append(average_scores.get(nbl, {}).get(method, {}).get(score_type, 0))
                stds[score_type][method].append(std_scores.get(nbl, {}).get(method, {}).get(score_type, 0))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Adjust size as needed
    for idx, (score_type, method_scores) in enumerate(scores.items()):
        for method, values in method_scores.items():
            std_values = stds[score_type][method]
            if method == 'FedAVG':
                # Plot 'FedAVG' for nbl == '0' as a point
                axs[idx].errorbar([0], [values[0]], yerr=[std_values[0]], label=method, marker='o', capsize=5)
            else:
                # Plot the mean value line for other methods, skipping the first '0'
                axs[idx].plot(range(1, len(values)+1), values, label=method, marker='o')
                # Fill the area between the upper and lower bounds for std error
                upper_bound = [v + s for v, s in zip(values, std_values)]
                lower_bound = [v - s for v, s in zip(values, std_values)]
                axs[idx].fill_between(range(1, len(values)+1), lower_bound, upper_bound, alpha=0.1)
        
        axs[idx].set_xlabel('Nbl')
        axs[idx].set_ylabel(score_type)
        

        axs[idx].set_xticks(range(len([0, 1, 2, 3])))
        axs[idx].set_xticklabels([0, 1, 2, 3])

    # Collect labels and handles
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Check to avoid duplicates
                handles.append(handle)
                labels.append(label)

    plt.subplots_adjust(top=0.85)  # Adjust this value to control the space above the subplots
    plt.tight_layout()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    #fig.suptitle('Evolution of Scores by nbl', fontsize=14)

    plt.show()

def print_scores(dataset,grouped_gog_results_by_seeds):
    nbls = [ '1' , '2', '3', '0'] 
    for nbl in nbls : 
        average_scores, std = get_average_std_results_over_seeds_by_nbl(grouped_gog_results_by_seeds, nbl)
        table_data = [[name, f"{average_scores[name]['Acc']:.2f} ± {std[name]['Acc']:.2f}", f"{average_scores[name]['ECE']:.2f} ± {std[name]['ECE']:.2f}", f"{average_scores[name]['NLL']:.2f} ± {std[name]['NLL']:.2f}"] for name in average_scores.keys()]
        #sorted_table_data = sorted(table_data, key=lambda x: float(x[1].split(' ± ')[0]), reverse=True)
        headers = ['Experiment', 'Acc', 'ECE', 'NLL']
        print(f"Experiments on {dataset} with {nbl} Bayesian Layers")
        print(tabulate(table_data, headers=headers))
        print("--------------------------------------------------------------")

def plot_scores(dataset,grouped_gog_results_by_seeds):
    scores = ['Acc', 'ECE', 'NLL']
    nbls = ['1','2','3']
    for nbl in nbls : 
        avg_sc , std = get_average_std_results_over_seeds_by_nbl(grouped_gog_results_by_seeds, nbl)
        # Define the number of subplots
        num_subplots = len(scores)
        # Create a figure with subplots
        fig, axs = plt.subplots(1, num_subplots, figsize=(6*num_subplots,8))
        # Iterate over the scores and create subplots
        for i, score in enumerate(scores):
            mean_scores = [value[score] for value in avg_sc.values()]
            std_devs = [value[score] for value in std.values()]
            labels = list(avg_sc.keys())
            x_pos = range(len(mean_scores))
            
            # Create the bar plot in the corresponding subplot
            axs[i].bar(x_pos, mean_scores, yerr=std_devs, capsize=5, alpha=0.75, color='skyblue')
            axs[i].set_xlabel('Methods')
            axs[i].set_ylabel(score)
            axs[i].set_xticks(x_pos)
            axs[i].set_xticklabels(labels, rotation=45)
            axs[i].yaxis.grid(True)
            
            # Add text annotations for the difference between methods
            for j in range(len(mean_scores)):
                axs[i].text(x_pos[j], mean_scores[j] + std_devs[j] , f'{std_devs[j]:.2f}', ha='center')
        # Adjust the layout and display the figure
        # Adjust the layout and display the figure
        fig.suptitle(f'Results on {dataset} with BFLAVG with {nbl} Bayesian Layer(s)', ha='center')
        plt.tight_layout()
        plt.show()

####################



def get_weights(dataset,seed):

    df = get_data_distribution(dataset,seed)
    total = df.sum().sum()

    weights = []
    for i in range(10): 
        weights.append( df[i].sum() / total)
    return weights

def get_locog_results(results_path):
    local_on_global_paths = [path for path in open(results_path).read().split('\n') if 'local_results' in path and 'on_global_data' in path]
    titles = []
    # Extract the specific parts from each path
    for path in local_on_global_paths:
        parts = path.split('/')
        extracted_parts = [parts[i] for i in [4, 5, 6, 7, -1] ]
        extracted_parts[-1] = extracted_parts[-1].split('_')[2] # save only the client id 
        if 'FedAVG' in extracted_parts[0]: 
            extracted_parts[1] = '' 
            extracted_parts[-2] = ''
        titles.append('_'.join(extracted_parts))
        titles = [title[:-1] if title.endswith('_') else title for title in titles]
        titles = [title.replace('__', '_') for title in titles]

    title_path_dict = {title: path for title, path in zip(titles, local_on_global_paths)}
    locog_results = {title: pd.read_csv(path) for title, path in title_path_dict.items()} #local on global results
    return locog_results

def group_by_experiment_locog(locog_results):
    grouped_locog_results = {}
    for k in locog_results.keys():
        experiment_name = '_'.join(k.split('_')[:-1])
        grouped_locog_results[experiment_name] = [value for key, value in locog_results.items() if experiment_name in key]
    return grouped_locog_results


def get_fairness_scores(dataset, seed, grouped_by_seed_grouped_locog_results):
    weights = get_weights(dataset,seed)
    metrics = {
    'Mean Accuracies': lambda dfs: extract_accuracies(dfs, np.average, weights),
    'Worst 10% Accuracies': lambda dfs: extract_accuracies(dfs, np.min), 
    'Average Std': lambda dfs: std_variance(dfs, weights )
    }
    results = {algo: {title: operation(dfs) for title, operation in metrics.items()} for algo, dfs in grouped_by_seed_grouped_locog_results[seed].items()}
    return results

def plot_fairness_vs_nbl(dataset,seed,grouped_by_seed_grouped_locog_results):
    scores = ['Mean Accuracies', 'Worst 10% Accuracies', 'Average Std']
    results = get_fairness_scores(dataset,seed,grouped_by_seed_grouped_locog_results)

    x = [1,2,3] # Number of Bayesian Layers
    fig, axs = plt.subplots(1, len(scores), figsize=(18, 2*len(scores)))
    # Group experiments by method name for a fixed seed
    for i, score in enumerate(scores):
        # Group experiments by method name for a fixed seed 
        grouped_experiments = {}
        for experiment in results.keys():
            if 'FedAVG' in experiment:
                fedavg_score = results[experiment][score]
                continue
            method_name = experiment.split('_')[1]
            if method_name in grouped_experiments:
                grouped_experiments[method_name].append(results[experiment][score])
            else:
                grouped_experiments[method_name] = [results[experiment][score]]
        axs[i].plot(0, fedavg_score, 'x', label='FedAVG')
        for method_name, s in grouped_experiments.items():
            axs[i].plot(x, s, 'o-', label=method_name)
        axs[i].set_xticks([0,1,2,3])
        axs[i].set_xlabel('Number of Bayesian Layers') 
        axs[i].set_ylabel(f'{score} Score')

    # Collect labels and handles
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Check to avoid duplicates
                handles.append(handle)
                labels.append(label)

    fig.suptitle('Fairness Scores vs Number of Bayesian Layers')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def get_average_std_fscores_over_seeds(fscores): 
    # Step 1: Transform the original dictionary
    transformed_dict = {}
    for seed, methods in fscores.items():
        for method_seed_nbl, table_of_results in methods.items():
            # Assuming method_seed_nbl is a string like "method_seed_nbl"
            # and table_of_results is a list where the last element is the result of the last round
            if 'FedAVG' in method_seed_nbl:
                method = 'FedAVG'
                nbl = '0'
            else :
                method = method_seed_nbl.split('_')[1]  # Adjust splitting logic as needed 
                nbl = method_seed_nbl.split('_')[-1]  # Adjust splitting logic as needed
            if nbl not in transformed_dict:
                transformed_dict[nbl] = {}
            if method not in transformed_dict[nbl]:
                transformed_dict[nbl][method] = {}
            transformed_dict[nbl][method][seed] = table_of_results

    # Step 2: Calculate average scores and std over seeds
    avg_scores_dict = {}
    std_dict = {}
    for nbl, methods in transformed_dict.items():
        avg_scores_dict[nbl] = {}
        std_dict[nbl] = {}
        for method, seeds_results in methods.items():
            scores = list(seeds_results.values())
            mean_accs_avg = np.mean([score['Mean Accuracies'] for score in scores])
            mean_accs_std = np.std([score['Mean Accuracies'] for score in scores])
            worst_acc_avg = np.mean([score['Worst 10% Accuracies'] for score in scores])
            worst_acc_std = np.std([score['Worst 10% Accuracies'] for score in scores])
            av_avg = np.mean([score['Average Std'] for score in scores])
            av_std = np.std([score['Average Std'] for score in scores])
            avg_scores_dict[nbl][method] = {'Mean Accuracies': mean_accs_avg, 'Worst 10% Accuracies': worst_acc_avg, 'Average Std': av_avg}
            std_dict[nbl][method] = {'Mean Accuracies': mean_accs_std, 'Worst 10% Accuracies': worst_acc_std, 'Average Std': av_std}

    return avg_scores_dict, std_dict

def print_fscores(dataset, fscores):
    nbls = [ '1' , '2', '3', '0'] 
    for nbl in nbls : 
        average_scores, std = get_average_std_fscores_over_seeds(fscores)
        average_scores, std = average_scores[nbl], std[nbl]
        table_data = [[name, f"{average_scores[name]['Mean Accuracies']:.2f} ± {std[name]['Mean Accuracies']:.2f}", f"{average_scores[name]['Worst 10% Accuracies']:.2f} ± {std[name]['Worst 10% Accuracies']:.2f}", f"{average_scores[name]['Average Std']:.2f} ± {std[name]['Average Std']:.2f}"] for name in average_scores.keys()]
        #sorted_table_data = sorted(table_data, key=lambda x: float(x[1].split(' ± ')[0]), reverse=True)
        headers = ['Experiment', 'Mean Accuracies', 'Worst 10% Accuracies', 'Average Std']
        print(f"Experiments on {dataset} with {nbl} Bayesian Layers")
        print(tabulate(table_data, headers=headers))
        print("--------------------------------------------------------------")
    
def plot_fscores(dataset,fscores):
    scores = ['Mean Accuracies', 'Worst 10% Accuracies', 'Average Std']
    nbls = ['1','2','3']
    for nbl in nbls : 
        avg_sc , std = get_average_std_fscores_over_seeds(fscores)
        avg_sc , std = avg_sc[nbl], std[nbl]
        # Define the number of subplots
        num_subplots = len(scores)
        # Create a figure with subplots
        fig, axs = plt.subplots(1, num_subplots, figsize=(6*num_subplots,8))
        # Iterate over the scores and create subplots
        for i, score in enumerate(scores):
            mean_scores = [value[score] for value in avg_sc.values()]
            std_devs = [value[score] for value in std.values()]
            labels = list(avg_sc.keys())
            x_pos = range(len(mean_scores))
            
            # Create the bar plot in the corresponding subplot
            axs[i].bar(x_pos, mean_scores, yerr=std_devs, capsize=5, alpha=0.75, color='skyblue')
            axs[i].set_xlabel('Methods')
            axs[i].set_ylabel(score)
            axs[i].set_xticks(x_pos)
            axs[i].set_xticklabels(labels, rotation=45)
            axs[i].yaxis.grid(True)
            
            # Add text annotations for the difference between methods
            for j in range(len(mean_scores)):
                axs[i].text(x_pos[j], mean_scores[j] + std_devs[j] , f'{std_devs[j]:.2f}', ha='center')
        # Adjust the layout and display the figure
        # Adjust the layout and display the figure
        fig.suptitle(f'Results on {dataset} with BFLAVG with {nbl} Bayesian Layer(s)', ha='center')
        plt.tight_layout()
        plt.show()


def plot_avg_std_fairness_vs_nbl(dataset, grouped_by_seed_grouped_locog_results): 
    seeds = ['0', '1', '2']    
    fscores = {}
    for seed in seeds : 
        fscores[seed] = get_fairness_scores(dataset, seed , grouped_by_seed_grouped_locog_results)
    # Assuming get_average_std_results_over_seeds function is available and imported
    average_scores, std_scores = get_average_std_fscores_over_seeds(fscores)

    # Ensure '0' is included in nbls and is the first element
    nbls = sorted(average_scores.keys(), key=lambda x: int(x))

    methods = set()
    for nbl in nbls:
        methods.update(average_scores[nbl].keys())

    # Prepare data for plotting (std)
    stds = { 'Mean Accuracies': {}, 'Worst 10% Accuracies': {}}
    for method in methods:
        for score_type in stds:
            stds[score_type][method] = []

    # Initialize data structures for plotting
    scores = { 'Mean Accuracies': {}, 'Worst 10% Accuracies': {}}
    for method in methods:
        for score_type in scores:
            scores[score_type][method] = []

    # Populate the data structures
    for nbl in nbls:
        for method in methods:
            if method != 'FedAVG' and nbl == '0':
                continue  # Skip '0' for non-'FedAVG' methods
            for score_type in scores:
                scores[score_type][method].append(average_scores.get(nbl, {}).get(method, {}).get(score_type, 0))
                stds[score_type][method].append(std_scores.get(nbl, {}).get(method, {}).get(score_type, 0))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Adjust size as needed
    for idx, (score_type, method_scores) in enumerate(scores.items()):
        for method, values in method_scores.items():
            std_values = stds[score_type][method]
            if method == 'FedAVG':
                # Plot 'FedAVG' for nbl == '0' as a point
                axs[idx].errorbar([0], [values[0]], yerr=[std_values[0]], label=method, marker='o', capsize=5)
            else:
                # Plot the mean value line for other methods, skipping the first '0'
                axs[idx].plot(range(1, len(values)+1), values, label=method, marker='o')
                # Fill the area between the upper and lower bounds for std error
                upper_bound = [v + s for v, s in zip(values, std_values)]
                lower_bound = [v - s for v, s in zip(values, std_values)]
                axs[idx].fill_between(range(1, len(values)+1), lower_bound, upper_bound, alpha=0.1)
        
        axs[idx].set_xlabel('Nbl')
        axs[idx].set_ylabel(score_type)
        

        axs[idx].set_xticks(range(len([0, 1, 2, 3])))
        axs[idx].set_xticklabels([0, 1, 2, 3])

    # Collect labels and handles
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Check to avoid duplicates
                handles.append(handle)
                labels.append(label)

    plt.subplots_adjust(top=0.85)  # Adjust this value to control the space above the subplots
    plt.tight_layout()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    #fig.suptitle('Evolution of Scores by nbl', fontsize=14)

    plt.show()

################### For statistical tests ####################


def organize_results_as_dataframe(grouped_gog_results_by_seeds, dataset_name="unknown"):
    records = []
    
    for seed, methods in grouped_gog_results_by_seeds.items():
        for method_seed_nbl, table_of_results in methods.items():
            last_round_result = table_of_results.iloc[-1]  # Get the final round's results
            
            if 'FedAVG' in method_seed_nbl:
                method = 'FedAVG'
                nbl = 0
            else:
                parts = method_seed_nbl.split('_')
                method = parts[1]  # assuming pattern like gog_method_nbl
                nbl = int(parts[-1])

            records.append({
                "aggregation_method": method,
                "nbl": nbl,
                "seed": int(seed),
                "dataset": dataset_name,
                "Acc": last_round_result['Acc'],
                "ECE": last_round_result['ECE'],
                "NLL": last_round_result['NLL']
            })

    df = pd.DataFrame(records)
    return df


def get_significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def plot_wilcoxon_heatmap(data, dataset, nbl, metric):
    filtered = data[
        (data['Dataset'] == dataset) &
        (data['nbl'] == nbl) &
        (data['Metric'] == metric)
    ]

    if filtered.empty:
        print(f"No data found for {dataset}, nbl={nbl}, metric={metric}")
        return

    methods = sorted(set(filtered['Method1']).union(filtered['Method2']))
    
    # Sort methods using hierarchical clustering
    method_scores = pd.DataFrame(index=methods, columns=methods)
    for _, row in filtered.iterrows():
        method_scores.loc[row['Method1'], row['Method2']] = row['p_value']
        method_scores.loc[row['Method2'], row['Method1']] = row['p_value']
    method_scores.fillna(1.0, inplace=True)
    
    # Get clustering order
    linkage_matrix = linkage(method_scores.values, method='average')
    order = leaves_list(linkage_matrix)
    sorted_methods = [methods[i] for i in order]
    
    # Create pivot table of p-values
    p_matrix = pd.DataFrame(index=sorted_methods, columns=sorted_methods, data=1.0)
    annot_matrix = pd.DataFrame(index=sorted_methods, columns=sorted_methods, data='')

    for _, row in filtered.iterrows():
        m1, m2, pval = row['Method1'], row['Method2'], row['p_value']
        p_matrix.loc[m1, m2] = pval
        p_matrix.loc[m2, m1] = pval
        star = get_significance_stars(pval)
        annot_matrix.loc[m1, m2] = star
        annot_matrix.loc[m2, m1] = star

    # Mask upper triangle
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        p_matrix.astype(float), 
        annot=annot_matrix, 
        fmt='', 
        cmap='coolwarm', 
        mask=mask, 
        cbar_kws={'label': 'p-value'},
        vmin=0, vmax=1, 
        linewidths=0.5, 
        square=True
    )
    plt.title(f"Wilcoxon p-values for {dataset}, nbl={nbl}, metric={metric}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_wilcoxon_grid(wilcoxon_df, metric='Acc'):
    datasets = sorted(wilcoxon_df['Dataset'].unique())
    pretty_dataset_names = {
    'cifar10': 'CIFAR-10',
    'fmnist': 'FashionMNIST',
    'svhn': 'SVHN'
    }
    pretty_method_names = {
    "eaa": "EAA",
    "wb": "WB",
    "rkl": "RKLB",
    "gaa": "GAA",
    "aalv": "AALV"
    }
    # Define your fixed method order (raw names)
    raw_method_order = ["eaa", "aalv", "gaa", "wb", "rkl"]
    pretty_method_names = {
        "eaa": "EAA",
        "wb": "WB",
        "rkl": "RKLB",
        "gaa": "GAA",
        "aalv": "AALV"
    }
    pretty_method_order = [pretty_method_names[m] for m in raw_method_order]

    nbls = sorted(wilcoxon_df['nbl'].unique())

    fig, axes = plt.subplots(len(datasets), len(nbls), figsize=(18, 12), squeeze=False)
    
    # Create a dummy colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height]

    for i, dataset in enumerate(datasets):
        for j, nbl in enumerate(nbls):
            ax = axes[i][j]
            filtered = wilcoxon_df[
                (wilcoxon_df['Dataset'] == dataset) &
                (wilcoxon_df['nbl'] == nbl) &
                (wilcoxon_df['Metric'] == metric)
            ]

            if filtered.empty:
                ax.set_visible(False)
                continue

            #methods = sorted(set(filtered['Method1']).union(filtered['Method2']))
            methods_raw = sorted(set(filtered['Method1']).union(filtered['Method2']))
            methods = [pretty_method_names.get(m, m) for m in methods_raw]

            #method_scores = pd.DataFrame(index=methods, columns=methods)
            method_scores = pd.DataFrame(index=methods_raw, columns=methods_raw)
            for _, row in filtered.iterrows():
                method_scores.loc[row['Method1'], row['Method2']] = row['p_value']
                method_scores.loc[row['Method2'], row['Method1']] = row['p_value']
            method_scores.fillna(1.0, inplace=True)

            # Clustering
            # linkage_matrix = linkage(method_scores.values, method='average')
            # order = leaves_list(linkage_matrix)
            # #sorted_methods = [methods[k] for k in order]
            # sorted_methods_raw = [methods_raw[k] for k in order]
            # sorted_methods = [pretty_method_names.get(m, m) for m in sorted_methods_raw]
            # Skip clustering, use fixed order
            sorted_methods_raw = raw_method_order
            sorted_methods = pretty_method_order



            p_matrix = pd.DataFrame(index=sorted_methods, columns=sorted_methods, data=1.0)
            annot_matrix = pd.DataFrame(index=sorted_methods, columns=sorted_methods, data='')

            # for _, row in filtered.iterrows():
            #     m1, m2, pval = row['Method1'], row['Method2'], row['p_value']
            #     #p_matrix.loc[m1, m2] = pval
            #     #p_matrix.loc[m2, m1] = pval
            #     p_matrix.loc[pretty_method_names.get(m1, m1), pretty_method_names.get(m2, m2)] = pval
            #     annot_matrix.loc[pretty_method_names.get(m1, m1), pretty_method_names.get(m2, m2)] = star
            #     p_matrix.loc[pretty_method_names.get(m2, m2), pretty_method_names.get(m1, m1)] = pval
            #     annot_matrix.loc[pretty_method_names.get(m2, m2), pretty_method_names.get(m1, m1)] = star

            #     star = get_significance_stars(pval)
            #     #annot_matrix.loc[m1, m2] = star
            #     #annot_matrix.loc[m2, m1] = star
            for _, row in filtered.iterrows():
                m1 = pretty_method_names.get(row['Method1'], row['Method1'])
                m2 = pretty_method_names.get(row['Method2'], row['Method2'])
                pval = row['p_value']
                star = get_significance_stars(pval)
                p_matrix.loc[m1, m2] = pval
                p_matrix.loc[m2, m1] = pval
                annot_matrix.loc[m1, m2] = star
                annot_matrix.loc[m2, m1] = star


            mask = np.triu(np.ones_like(p_matrix.values, dtype=bool))

            sns.heatmap(
                p_matrix.astype(float),
                annot=annot_matrix,
                fmt='',
                cmap='coolwarm',
                mask=mask,
                ax=ax,
                vmin=0,
                vmax=1,
                square=True,
                linewidths=0.3,
                xticklabels=True,
                yticklabels=True,
                cbar=(i == 0 and j == len(nbls) - 1),  # Add cbar only on last top row cell
                cbar_ax=cbar_ax if (i == 0 and j == len(nbls) - 1) else None
            )

            if i == 0:
                ax.set_title(f"Nbl = {nbl}")
            if j == 0:
                ax.set_ylabel(pretty_dataset_names.get(dataset, dataset))
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    #plt.suptitle(f"Wilcoxon p-values (Metric: {metric})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(f"images/wilcoxon_grid_{metric}.pdf", bbox_inches='tight')
    plt.show()
