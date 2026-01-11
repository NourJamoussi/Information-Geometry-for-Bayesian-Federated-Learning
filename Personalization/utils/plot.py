import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import art3d
import numpy as np
from matplotlib.patches import FancyArrowPatch

class Args:
    def __init__(self, dataset, init_seed, alg, update_method, nbl, lagrangian_parameters, perso_method) -> None:
        self.dataset = dataset
        self.init_seed = init_seed
        self.alg = alg
        self.update_method = update_method
        self.nbl = nbl
        self.lagrangian_parameters = lagrangian_parameters
        self.perso_method = perso_method


def merge_results(args):
    # Define directories
    dir = f'logs/{args.dataset}/{args.init_seed}'
    experiment_dir = f'{dir}/{args.alg}/{args.update_method}/{args.nbl}'
    results_dir = f'{experiment_dir}/results'

    # Initialize an empty DataFrame to store all results
    merged_data = pd.DataFrame()

    # Helper function to read and append CSV files
    def read_and_append_csv(folder, label):
        nonlocal merged_data
        folder_path = os.path.join(results_dir, folder)
        if os.path.isfile(folder_path + '.csv'):
            # If the path is a file, read it directly
            df = pd.read_csv(folder_path + '.csv')
            df['source'] = label
            df['file'] = folder + '.csv'
            df['file_number'] = 0  # Assign a default number for single files
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        else:
            # If the path is a directory, read all CSV files in it
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                    df['source'] = label
                    df['file'] = file_name
                    # Extract the number from the file name
                    file_number = int(''.join(filter(str.isdigit, file_name)))
                    df['file_number'] = file_number
                    merged_data = pd.concat([merged_data, df], ignore_index=True)

    # Read global model results
    read_and_append_csv('global_on_local', 'global_on_local')
    read_and_append_csv('local_on_global', 'local_on_global')
    read_and_append_csv('global_on_global', 'global_on_global')
    read_and_append_csv('local_on_local', 'local_on_local')

    # Read personalized model results
    for lam in args.lagrangian_parameters:
        read_and_append_csv(f'personalized/{args.perso_method}/{lam}/perso_on_local', f'perso_on_local_lambda_{lam}')
        read_and_append_csv(f'personalized/{args.perso_method}/{lam}/perso_on_global', f'perso_on_global_lambda_{lam}')

    # Sort the merged data by source and file_number
    merged_data.sort_values(by=['source', 'file_number'], inplace=True)

    # Drop the file_number column as it's no longer needed
    merged_data.drop(columns=['file_number'], inplace=True)

    # Save the merged data to a new CSV file
    merged_data.to_csv(os.path.join(f'{results_dir}/personalized/{args.perso_method}/', 'merged_results.csv'), index=False)


def extract_client_id(file_name):
    client_match = re.search(r'client_(\d+)', file_name)
    data_match = re.search(r'data_(\d+)', file_name)
    perso_math = re.search(r'perso_(\d+)', file_name)
    if client_match:
        return client_match.group(1)
    elif data_match:
        return data_match.group(1)
    elif perso_math:
        return perso_math.group(1)
    else:
        return None

def plot_results_on_local_data(args):
    # Define the path to the CSV file
    def get_path(init_seed):
        dir = f'logs/{args.dataset}/{init_seed}'
        experiment_dir = f'{dir}/{args.alg}/{args.update_method}/{args.nbl}'
        results_dir = f'{experiment_dir}/results' 
        return f'{results_dir}/personalized/{args.perso_method}/merged_results.csv'

    # Step 1: Read the CSV files for init_seed 0 to 2 and concatenate them
    dfs = []
    for seed in range(3):
        path = get_path(seed)
        df = pd.read_csv(path)
        df['init_seed'] = seed
        dfs.append(df)

    df = pd.concat(dfs)

    # Step 2: Filter the DataFrame to include only the relevant rows for the plot
    df = df[df['source'].str.contains(f'local_on_local|global_on_local|perso_on_local_lambda_0.5|perso_on_local_lambda_1|perso_on_local_lambda_2')]

    # Step 3: Extract the client ID from the 'file' column
    df['client_id'] = df['file'].apply(extract_client_id)
    df = df.dropna(subset=['client_id'])


    # Step 4: Group the data by client and method for Acc, ECE, and NLL
    grouped_acc = df.groupby(['client_id', 'source'])['Acc'].agg(['mean', 'std']).unstack()
    grouped_ece = df.groupby(['client_id', 'source'])['ECE'].agg(['mean', 'std']).unstack()
    grouped_nll = df.groupby(['client_id', 'source'])['NLL'].agg(['mean', 'std']).unstack()


    # Step 5: Reorder the columns as specified
    order = [f'global_on_local', f'perso_on_local_lambda_0.5', f'perso_on_local_lambda_1', f'perso_on_local_lambda_2', f'local_on_local']
    grouped_acc = grouped_acc.reindex(order, axis=1, level=1)
    grouped_ece = grouped_ece.reindex(order, axis=1, level=1)
    grouped_nll = grouped_nll.reindex(order, axis=1, level=1)


    # Step 6: Rename the columns for better legend names
    rename_columns = {
        f'global_on_local': 'Global Model',
        f'perso_on_local_lambda_0.5': 'Personalized Model (λ=0.5)',
        f'perso_on_local_lambda_1': 'Personalized Model (λ=1)',
        f'perso_on_local_lambda_2': 'Personalized Model (λ=2)',
        f'local_on_local': 'Local Model'
    }
    grouped_acc = grouped_acc.rename(columns=rename_columns, level=1)
    grouped_ece = grouped_ece.rename(columns=rename_columns, level=1)
    grouped_nll = grouped_nll.rename(columns=rename_columns, level=1)

    # Step 7: Create subplots for Acc, ECE, and NLL
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Accuracy
    grouped_acc['mean'].plot(kind='bar', yerr=grouped_acc['std'], ax=axes[0], width=0.8, capsize=4)
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(title=f'On local data:')
    axes[0].set_title('Accuracy')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees

    # Plot ECE
    grouped_ece['mean'].plot(kind='bar', yerr=grouped_ece['std'], ax=axes[1], width=0.8, capsize=4)
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('ECE')
    axes[1].legend(title=f'On local data:')
    axes[1].set_title('Expected Calibration Error (ECE)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees

    # Plot NLL
    grouped_nll['mean'].plot(kind='bar', yerr=grouped_nll['std'], ax=axes[2], width=0.8, capsize=4)
    axes[2].set_xlabel('Client ID')
    axes[2].set_ylabel('NLL')
    axes[2].legend(title=f'On local data:')
    axes[2].set_title('Negative Log-Likelihood (NLL)')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees


    # Customize the plot
    #plt.tight_layout()
    plt.suptitle(f'Dataset: {args.dataset}, Update Method: {args.update_method}, Personalization Method: {args.perso_method}, Nbl: {args.nbl}', y=1.02)
    os.makedirs(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}', exist_ok=True)
    plt.savefig(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}/results_local_data.pdf')
    plt.savefig(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}/results_local_data.png')
    plt.close()
    #plt.show()



def plot_results_on_global_data(args):

    # Define the seeds
    seeds = ['0', '1', '2']

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each seed and read the corresponding CSV file
    for seed in seeds:
        dir = f'logs/{args.dataset}/{seed}'
        experiment_dir = f'{dir}/{args.alg}/{args.update_method}/{args.nbl}'
        results_dir = f'{experiment_dir}/results' 
        path = f'{results_dir}/personalized/{args.perso_method}/merged_results.csv'
        
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Add a column for the seed
        df['seed'] = seed
        
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames
    df = pd.concat(dfs)

    # Filter the DataFrame to include only the relevant rows for the plot
    df = df[df['source'].str.contains('local_on_global|global_on_global|perso_on_global_lambda_0.5|perso_on_global_lambda_1|perso_on_global_lambda_2')]

    # Extract the client ID from the 'file' column
    df['client_id'] = df['file'].apply(extract_client_id)

    # Handle global_on_global separately
    global_on_global = df[df['source'] == 'global_on_global'].copy()

    # Group the data by client and method for Acc, ECE, and NLL
    grouped_acc = df.groupby(['client_id', 'source'])['Acc'].agg(['mean', 'std']).unstack()
    grouped_ece = df.groupby(['client_id', 'source'])['ECE'].agg(['mean', 'std']).unstack()
    grouped_nll = df.groupby(['client_id', 'source'])['NLL'].agg(['mean', 'std']).unstack()

    # Duplicate global_on_global results for each client
    global_mean_acc = global_on_global['Acc'].mean()
    global_std_acc = global_on_global['Acc'].std()
    global_mean_ece = global_on_global['ECE'].mean()
    global_std_ece = global_on_global['ECE'].std()
    global_mean_nll = global_on_global['NLL'].mean()
    global_std_nll = global_on_global['NLL'].std()

    for client_id in grouped_acc.index:
        grouped_acc.loc[client_id, ('mean', 'global_on_global')] = global_mean_acc
        grouped_acc.loc[client_id, ('std', 'global_on_global')] = global_std_acc
        grouped_ece.loc[client_id, ('mean', 'global_on_global')] = global_mean_ece
        grouped_ece.loc[client_id, ('std', 'global_on_global')] = global_std_ece
        grouped_nll.loc[client_id, ('mean', 'global_on_global')] = global_mean_nll
        grouped_nll.loc[client_id, ('std', 'global_on_global')] = global_std_nll

    # Reorder the columns as specified
    order = ['global_on_global', 'perso_on_global_lambda_0.5', 'perso_on_global_lambda_1', 'perso_on_global_lambda_2', 'local_on_global']
    grouped_acc_mean = grouped_acc['mean'][order]
    grouped_acc_std = grouped_acc['std'][order]
    grouped_ece_mean = grouped_ece['mean'][order]
    grouped_ece_std = grouped_ece['std'][order]
    grouped_nll_mean = grouped_nll['mean'][order]
    grouped_nll_std = grouped_nll['std'][order]

    # Rename the columns for better legend names
    rename_columns = {
        'global_on_global': 'Global Model',
        'perso_on_global_lambda_0.5': 'Personalized Model (λ=0.5)',
        'perso_on_global_lambda_1': 'Personalized Model (λ=1)',
        'perso_on_global_lambda_2': 'Personalized Model (λ=2)',
        'local_on_global': 'Local Model'
    }

    # Rename columns
    grouped_acc_mean = grouped_acc_mean.rename(columns=rename_columns)
    grouped_acc_std = grouped_acc_std.rename(columns=rename_columns)
    grouped_ece_mean = grouped_ece_mean.rename(columns=rename_columns)
    grouped_ece_std = grouped_ece_std.rename(columns=rename_columns)
    grouped_nll_mean = grouped_nll_mean.rename(columns=rename_columns)
    grouped_nll_std = grouped_nll_std.rename(columns=rename_columns)

    # Create subplots for Acc, ECE, and NLL
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Accuracy
    grouped_acc_mean.plot(kind='bar', yerr=grouped_acc_std, ax=axes[0], width=0.8, capsize=4)
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(title='On global data:')
    axes[0].set_title('Accuracy')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees

    # Plot ECE
    grouped_ece_mean.plot(kind='bar', yerr=grouped_ece_std, ax=axes[1], width=0.8, capsize=4)
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('ECE')
    axes[1].legend(title='On global data:')
    axes[1].set_title('Expected Calibration Error (ECE)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees

    # Plot NLL
    grouped_nll_mean.plot(kind='bar', yerr=grouped_nll_std, ax=axes[2], width=0.8, capsize=4)
    axes[2].set_xlabel('Client ID')
    axes[2].set_ylabel('NLL')
    axes[2].legend(title='On global data:')
    axes[2].set_title('Negative Log-Likelihood (NLL)')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)  # Set rotation to 0 degrees

    # Customize the plot
    #plt.tight_layout()
    plt.suptitle(f'Dataset: {args.dataset}, Update Method: {args.update_method}, Personalization Method: {args.perso_method}, Nbl: {args.nbl}', y=1.02)
    os.makedirs(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}', exist_ok=True)
    plt.savefig(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}/results_global_data.pdf')
    plt.savefig(f'figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}/results_global_data.png')
    plt.close()
    #plt.show()

    

def create_html(dataset, alg, update_methods, nbls, lagrangian_parameters, personalization_methods, on_data='local', init_seed=0): 

    html_content = f'<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head><body><h1>{dataset} - experiments on {on_data} data </h1><div style="display: flex; flex-wrap: wrap;">'
    for update_method in update_methods:
        for perso_method in personalization_methods:
            for nbl in nbls:
                args = Args(dataset, init_seed, alg, update_method, nbl, lagrangian_parameters, perso_method)
                
                # Create a title for each plot
                title = f'Update Method : {args.update_method}, Personalization Method: {args.perso_method}, Nbl: {args.nbl}'
                
                # Add the title and image to the HTML content
                html_content += f'<div style="margin: 10px; text-align: center;">'
                html_content += f'<h3>{title}</h3>'
                html_content += f'<img src="../figs/{args.dataset}/{args.alg}/{args.update_method}/{args.nbl}/{args.perso_method}/results_{on_data}_data.png" style="max-width: 100%; height: auto;">'
                html_content += '</div>'

    html_content += '</div></body></html>'

    with open(f'htmls/{dataset}_{on_data}_plots.html', 'w') as f:
        f.write(html_content)


def plot_sphere():
    plt.rcParams['text.usetex'] = True
    os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
    # Increase the fontsize
    plt.rcParams['font.size'] = 18

    # Plot the center
    # Define the center and radius of the sphere
    P_k = [0, 0, 0]  # Center coordinates (x, y, z)
    r_k = 1  # Radius

    # Generate the sphere data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_k * np.outer(np.cos(u), np.sin(v)) + P_k[0]
    y = r_k * np.outer(np.sin(u), np.sin(v)) + P_k[1]
    z = r_k * np.outer(np.ones(np.size(u)), np.cos(v)) + P_k[2]

    # Create a 3D plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    ax.plot_surface(x, y, z, color='r', alpha=0.4)
    ax.scatter(P_k[0], P_k[1], P_k[2], color='b', marker='o')

    # Define the coordinates of point P_g
    P_g = [1.5, 1.5, 1.5]

    # Plot the point P_g
    ax.scatter(P_g[0], P_g[1], P_g[2], color='g', marker='o')

    # Calculate the projection of P_g onto the sphere surface
    direction = np.array(P_g) - np.array(P_k)
    direction = direction / np.linalg.norm(direction)  # Normalize the direction
    P_proj = np.array(P_k) + direction * r_k  # Projected point on the sphere surface
    P_sym = -  P_proj 


    # Extend the arrow slightly inside the sphere's surface
    penetration_depth = -0.12  # Negative value to go inside the sphere
    P_inside = P_proj + direction * penetration_depth

    # Create a curved arrow from P_g to inside the sphere
    t = np.linspace(0, 1, 100)
    curve_x = (1 - t) * P_g[0] + t * P_inside[0] + 0.2 * np.sin(2*np.pi * t)
    curve_y = (1 - t) * P_g[1] + t * P_inside[1] + 0.2 * np.sin(0.5*np.pi * t)
    curve_z = (1 - t) * P_g[2] + t * P_inside[2] + 0.2 * np.sin(np.pi * t)

    ax.plot3D(curve_x, curve_y, curve_z, 'k')

    # Add the arrow head at the end of the curve
    arrow = FancyArrowPatch((curve_x[-2], curve_y[-2], curve_z[-2]), 
                            (curve_x[-1], curve_y[-1], curve_z[-1]), 
                            mutation_scale=500, color='k', arrowstyle='-|>')
    ax.add_patch(arrow)
    art3d.pathpatch_2d_to_3d(arrow, z=curve_z[-1], zdir="z")


    ax.plot([P_k[0], P_sym[0]], [P_k[1], P_sym[1]], [P_k[2], P_sym[2]], color='b', linestyle='--')
    # Add the radius label
    ax.text(P_k[0] -0.5 , P_k[1] -0.4 , P_k[2] -0.2, r'$r_k$', color='b')
    # Add the sphere label 
    ax.text(P_k[0] -0.5 , P_k[1] -0.4 , P_k[2] + 1.3, r'$\mathcal{S}_k$', color='r')

    # Add labels to the points with some offset
    offset = 0.1
    ax.text(P_k[0] + offset, P_k[1] + offset, P_k[2] + offset, r'$p_k$', color='b')
    ax.text(P_g[0] + offset, P_g[1] + offset, P_g[2] + offset, r'$p_g$', color='g')

    # Set plot limits and labels
    ax.set_xlim([-r_k + P_k[0], P_g[0] + r_k])
    ax.set_ylim([-r_k + P_k[1], P_g[1] + r_k])
    ax.set_zlim([-r_k + P_k[2], P_g[2] + r_k])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove the grid
    ax.grid(False)

    # Show the plot
    plt.tight_layout()
    plt.savefig('sphere_projection.pdf', bbox_inches='tight')
    plt.show()