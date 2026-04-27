# %% [markdown]
# # Training Classification Models
# 
# This file is responsible for training the network traffic classification models to be used for performing the attacks.
# Make sure to add the dataset files in the `dataset` folder.

# %% [markdown]
# ## Parameters

# %%
from utils.parameters import *
n_copies = 3

# %% [markdown]
# # Load datasets

# %%
from utils.helpers.load_datasets import load_all_datasets

datasets = load_all_datasets(copies=n_copies, random_state=random_state, frac=0.3)

# %% [markdown]
# # Creating the neural networks

# %%
from utils.helpers.load_networks import load_networks, network_types

for dataset_name, dataset in datasets.items():
    load_networks(
        dataset=dataset,
        dataset_name=dataset_name,
        batch_size=batch_size,
        n_hidden_layers=n_hidden_layers,
        n_neurons=n_neurons,
        dropout_prob_fnn=dropout_prob_fnn,
        dropout_prob_snn=dropout_prob_snn,
        learning_rate_fnn=learning_rate_fnn,
        learning_rate_snn=learning_rate_snn,
        random_state=random_state
    )

# %% [markdown]
# # Training the neural networks

# %%
for dataset_name, dataset in datasets.items():
    print(f"Training models for dataset: {dataset_name}")
    if dataset["fnn_trainer"].epoch < training_epochs:
        dataset["fnn_trainer"].train_network(epochs=training_epochs - dataset["fnn_trainer"].epoch)
    if dataset["snn_trainer"].epoch < training_epochs:
        dataset["snn_trainer"].train_network(epochs=training_epochs - dataset["snn_trainer"].epoch)

# %% [markdown]
# ## Calculate metrics

# %%
import pandas as pd
import json

metrics = {}

for dataset_name, dataset in datasets.items():
    base_dataset_name = dataset_name.split('_copy_')[0]
    
    if base_dataset_name not in metrics:
        metrics[base_dataset_name] = {}
        for network in network_types:
            metrics[base_dataset_name][network] = []
    
    for network in network_types:        
        network_report = dataset[f"{network}_trainer"].get_classification_report(dataset["dataPreparer"], output_dict=True)
        metrics[base_dataset_name][network].append(network_report)
    
with open(classification_results_filename, "w") as f:
    json.dump(metrics, f, indent=4)
    
# %% [markdown]
# ## Attacks
# ### Prepare the attack data

# %% [markdown]
# ### Conduct attacks

# %%
from utils.helpers.conduct_attacks import get_attack_datasets, create_and_train_idsgan, evaluate_idsgan, create_and_train_genaal, evaluate_genaal, create_and_train_polytope, evaluate_polytope
from utils.helpers.load_networks import network_types

def conduct_attacks(dataset: dict, attack_trials: dict):
    n_trial = max(list(attack_trials.values())) if len(attack_trials) > 0 else 0
    
    results_partial = {}
    
    # Prepare attack datasets
    attack_train_df, attack_test_df = get_attack_datasets(dataset, frac=dataset["attack_frac"], train_size=attack_train_size, random_state=random_state + n_trial)

    for network_name in network_types:
        #
        # IDSGAN
        #
        
        attack_name = f"idsgan_{network_name}"        
        if attack_name not in attack_trials or attack_trials[attack_name] < n_trials['idsgan']:
            print(f"Conducting IDSGAN attack on {network_name.upper()} (Trial {attack_trials.get(attack_name, 0) + 1}/{n_trials['idsgan']})")
            
            idsgan = create_and_train_idsgan(
                train_df=attack_train_df,
                test_df=attack_test_df,
                dataset=dataset,
                dataset_name=dataset_name,
                network_name=network_name,
                functional_features=dataset["functional_features"],
                noise_dim = noise_dim,
                n_hidden_layers_generator = n_hidden_layers_generator,
                n_neurons_generator = n_neurons_generator,
                learning_rate_generator = learning_rate_generator,
                steps_generator = steps_generator,
                n_hidden_layers_discriminator = n_hidden_layers_discriminator,
                n_neurons_discriminator = n_neurons_discriminator,
                learning_rate_discriminator = learning_rate_discriminator,
                steps_discriminator = steps_discriminator,
                weight_clip = weight_clip,
                batch_size = batch_size,
                idsgan_epochs = idsgan_epochs
            )

            results_run = evaluate_idsgan(
                idsgan,
                attack_test_df,
                dataset,
                network_name,
                dataset["functional_features"]
            )

            print(f"IDSGAN {network_name.upper()} Trial Results: Accuracy: {results_run['accuracy']:.4f}, Average Distance: {results_run['average_distance']:.4f}")
            results_partial[attack_name] = results_run

        #
        # GenAAL
        #
         
        attack_name = f"genaal_{network_name}"
        if attack_name not in attack_trials or attack_trials[attack_name] < n_trials['gen_aal']:
            print(f"Conducting GenAAL attack on {network_name.upper()} (Trial {attack_trials.get(attack_name, 0) + 1}/{n_trials['gen_aal']})")   
            
            genaal = create_and_train_genaal(
                train_df=attack_train_df,
                test_df=attack_test_df,
                dataset=dataset,
                dataset_name=dataset_name,
                network_name=network_name,
                latent_dim = latent_dim,
                vae_hidden = vae_hidden,
                sids_hidden = sids_hidden,
                lambda_kl = lambda_kl,
                lambda_recon = lambda_recon,
                lambda_l2 = lambda_l2,
                lambda_label = lambda_label,
                gen_lr = gen_lr,
                sid_lr = sid_lr,
                batch_size = batch_size,
                max_iterations = max_iterations,
                candidate_pool_k = candidate_pool_k,
                nquery = nquery,
                pretrain_epochs = pretrain_epochs,
                sids_epochs = sids_epochs,
                gen_epochs = gen_epochs,
                label_query = label_query
            )

            results_run = evaluate_genaal(
                genaal,
                attack_test_df,
                dataset,
                network_name
            )

            print(f"GenAAL {network_name.upper()} Trial Results: Accuracy: {results_run['accuracy']:.4f}, Average Distance: {results_run['average_distance']:.4f}")
            results_partial[attack_name] = results_run

        #
        # Polytope Attack
        #
        
        attack_name = f"polytope_{network_name}"
        
        if attack_name not in attack_trials or attack_trials[attack_name] < n_trials['polytope']:
            print(f"Conducting Polytope attack on {network_name.upper()} (Trial {attack_trials.get(attack_name, 0) + 1}/{n_trials['polytope']})")
            
            polytope_attack = create_and_train_polytope(
                train_df=attack_train_df,
                test_df=attack_test_df,
                dataset=dataset,
                network_name=network_name,
                batch_size=batch_size,
                random_state=random_state + n_trial,
                n_rays=n_rays,
                step_size=step_size)

            results_run = evaluate_polytope(
                polytope_attack,
                attack_test_df,
                dataset,
                network_name,
                functional_features=dataset["functional_features"],
                move_inside=0
            )

            print(f"Polytope {network_name.upper()} Trial Results: Accuracy: {results_run['accuracy']:.4f}, Average Distance: {results_run['average_distance']:.4f}")
            results_partial[attack_name] = results_run
        
    return results_partial

# %%
import json
import numpy as np

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    else:
        return obj
    
def load_results_from_json(filename="attack_results.json"):
    try:
        with open(filename, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
        
    return results
    
def save_results_to_json(results, filename="attack_results.json"):
    results_old = load_results_from_json(filename)
        
    # Merge old results with new results
    for dataset_name in results:
        if dataset_name not in results_old:
            results_old[dataset_name] = results[dataset_name]
        else:
            results_old[dataset_name].extend(results[dataset_name])
            # Remove duplicates
            runs_unique = []
            
            for run in results_old[dataset_name]:
                if run not in runs_unique:
                    runs_unique.append(run)
            results_old[dataset_name] = runs_unique

    results = results_old
    with open(filename, "w") as f:
        json.dump(numpy_to_python(results), f)

# %%
results = load_results_from_json(filename=attack_results_filename)

for dataset_name in datasets:    
    if dataset_name not in results:
        results[dataset_name] = []
        
    while True:
        attack_trials = {}
        for trial in results[dataset_name]:
            for attack_name in trial.keys():
                if attack_name not in attack_trials:
                    attack_trials[attack_name] = 1
                attack_trials[attack_name] += 1
        
        results_trial = conduct_attacks(datasets[dataset_name], attack_trials)
        if len(results_trial) == 0:
            break
    
        results[dataset_name].append(results_trial)
        save_results_to_json(results, filename=attack_results_filename)

# %% [markdown]
# # Plot Results
print("Plotting results...")
        
# %% [markdown]
# ## Plot Classification Results

# %%
import json
import pandas as pd

with open(classification_results_filename, "r") as f:
    results = json.load(f)
    
classification_results = {}

for dataset_name, reports in results.items():
    fnn_df = pd.DataFrame([metric['weighted avg'] for metric in reports["fnn"]])
    snn_df = pd.DataFrame([metric['weighted avg'] for metric in reports["snn"]])
    
    # Calculate average and std for each metric
    fnn_df = fnn_df.agg(['mean', 'std'])
    snn_df = snn_df.agg(['mean', 'std'])
    
    reports["avg_fnn"] = fnn_df
    reports["avg_snn"] = snn_df
    
    classification_results[dataset_name] = {
        "fnn": {
            "accuracy": [metric['accuracy'] for metric in reports["fnn"]],
            "f1-score": [metric['weighted avg']['f1-score'] for metric in reports["fnn"]]
        },
        "snn": {
            "accuracy": [metric['accuracy'] for metric in reports["snn"]],
            "f1-score": [metric['weighted avg']['f1-score'] for metric in reports["snn"]]
        }
    }

# %% [markdown]
# ### Classification models accuracy and F-1 score

# %%
import matplotlib.pyplot as plt

n_datasets = len(classification_results)
n_rows = (n_datasets + 1) // 2

fig, axes = plt.subplots(
    n_rows,
    1,
    figsize=(5, 4.5 * n_rows),
    squeeze=False
)

if n_datasets == 1:
    axes = np.array([axes[0, 0]])

for idx, (dataset_name, metrics) in enumerate(classification_results.items()):
    ax = axes[idx]

    data = [
        metrics["fnn"]["accuracy"],
        metrics["snn"]["accuracy"],
        metrics["fnn"]["f1-score"],
        metrics["snn"]["f1-score"],
    ]

    ax.boxplot(
        data,
        tick_labels=['FNN\nAcurácia', 'SNN\nAcurácia', 'FNN\nF1-score', 'SNN\nF1-score'],
        widths=0.6,
        showfliers=True,
        patch_artist=False,
    )
    
    for label in ax.get_xticklabels():
        label.set_fontsize(14)

    # Axis formatting
    ax.set_ylabel('Valor', fontsize=14)
    ax.set_ylim(0.9, 1.0)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)

    ax.grid(
        True,
        axis='y',
        linestyle='--',
        linewidth=0.6,
        alpha=0.6
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(dataset_names[dataset_name], fontsize=16)

plt.tight_layout()

# Save figure as pdf and png
fig.savefig("results/classification_results.pdf", format='pdf')
fig.savefig("results/classification_results.png", format='png', dpi=300)

# %% [markdown]
# ## Plot Attack Results

# %%
import json

with open(attack_results_filename, "r") as f:
    results = json.load(f)
    
results_datasets = {}

for dataset_name, dataset_results in results.items():
    dataset_basename = dataset_name.split('_copy')[0]
    if dataset_basename not in results_datasets:
        results_datasets[dataset_basename] = {}
        
    for trial in dataset_results:
        for attack_network, res in trial.items():
            splits = attack_network.split('_')
            attack_type, network_type = splits[0], splits[1]
            
            if len(splits) > 2:
                attack_type += '_' + "_".join(splits[2:])
            
            if network_type not in results_datasets[dataset_basename]:
                results_datasets[dataset_basename][network_type] = {}
            if attack_type not in results_datasets[dataset_basename][network_type]:
                results_datasets[dataset_basename][network_type][attack_type] = []
                
            results_datasets[dataset_basename][network_type][attack_type].append(res)

# %% [markdown]
# ### Scatter plot of accuracy X distance

# %%
import matplotlib.pyplot as plt

lim_pad = 0.05

for dataset_name, dataset_results in results_datasets.items():
    n_network_types = len(dataset_results)
    
    fig, axes = plt.subplots(1, n_network_types, figsize=(8 * n_network_types, 5), squeeze=False)
    title = f'Resultados de evasão - {dataset_names.get(dataset_name, dataset_name.upper())}'
    # fig.suptitle(title, fontsize=16, y=0.92)
    
    for network_type, network_results in dataset_results.items():
        network_name = network_type.upper()
        
        # Accuracy vs Average Distance
        ax1 = axes[0, list(dataset_results.keys()).index(network_type)]

        for attack_type, attack_results in network_results.items():
            attack_name = attack_types.get(attack_type, attack_type.upper())

            accuracies = [res['accuracy'] * 100 for res in attack_results]
            distances = [res['average_distance'] for res in attack_results]

            # scatter plot with color mapping
            ax1.scatter(
                distances,
                accuracies,
                label=attack_name,
                marker=marker_map.get(attack_type, 'o'),
                color=color_map.get(attack_type, 'black'),
                s=40,                    
                edgecolors='black',
                linewidths=0.6,
                alpha=0.85
            )

        # Axis labels
        ax1.set_xlabel('Distância Média (escala logarítmica)')
        ax1.set_ylabel('Taxa de Sucesso de Evasão (%)')
        ax1.xaxis.label.set_size(15)
        ax1.yaxis.label.set_size(15)

        # Scaling and limits
        ax1.set_xscale('log')
        ax1.set_ylim(-lim_pad * 100, 100 + lim_pad * 100)
        
        for label in ax1.get_yticklabels():
            label.set_fontsize(12)
            
        for label in ax1.get_xticklabels():
            label.set_fontsize(12)

        ax1.grid(
            True,
            which='major',
            axis='y',
            linestyle='--',
            linewidth=0.6,
            alpha=0.6
        )

        ax1.legend(
            title='Tipo de Ataque',
            frameon=False,
            fontsize='large',
            title_fontsize='large',
            loc='center right'
        )

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
    fig.savefig(f"results/attack_results_{dataset_name}.pdf", format='pdf')
    fig.savefig(f"results/attack_results_{dataset_name}.png", format='png', dpi=300)

# %% [markdown]
# ### Table with metrics from 3 best runs for each attack type

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

results_best = {}
n_best = 3

# For each dataset
for dataset_name, dataset_results in results_datasets.items():
    # For each network type
    results_best[dataset_name] = {}
    for network_type, network_results in dataset_results.items():
        # For each attack type
        best_metrics = {}
        for attack_type, attack_results in network_results.items():
            # Find the best n results based on accuracy, using the average distance as tiebreaker
            sorted_results = sorted(
                attack_results,
                key=lambda x: (x['accuracy'], -x['average_distance']),
                reverse=True
            )
            best_n = sorted_results[:n_best]
            accuracies = [res['accuracy'] for res in best_n]
            distances = [res['average_distance'] for res in best_n]
            best_metrics[attack_type] = {
                'accuracies': accuracies,
                'distances': distances
            }
        results_best[dataset_name][network_type] = best_metrics

# Convert to dataframe table
table_data = []
for dataset_name, dataset_results in results_best.items():
    for network_type, network_results in dataset_results.items():
        for attack_type, metrics in network_results.items():
            for i in range(n_best):
                table_data.append({
                    'Dataset': dataset_names.get(dataset_name, dataset_name.upper()),
                    'Network Type': network_type.upper(),
                    'Attack Type': attack_types.get(attack_type, attack_type.upper()),
                    'Accuracy': metrics['accuracies'][i],
                    'Average Distance': metrics['distances'][i]
                })
                
df = pd.DataFrame(table_data)

# Group results
grouped_df = df.groupby(['Network Type', 'Attack Type']).agg({
    'Accuracy': ['mean', 'std'],
    'Average Distance': ['mean', 'std']
}).reset_index()

# Flatten MultiIndex columns
grouped_df.columns = ['_'.join(col).strip('_') for col in grouped_df.columns.values]

# Change accuracy and distance to +- std format
grouped_df['Sucesso (%)'] = grouped_df.apply(
    lambda row: f"{row['Accuracy_mean']*100:.2f}% ± {row['Accuracy_std']*100:.2f}%",
    axis=1
)

grouped_df['Distância Média'] = grouped_df.apply(
    lambda row: f"{row['Average Distance_mean']:.4f} ± {row['Average Distance_std']:.4f}",
    axis=1
)

# Save as LaTeX table, grouped by Attack Type and Network Type
latex_table = grouped_df.pivot(
    index='Attack Type',
    columns='Network Type',
    values=['Sucesso (%)', 'Distância Média']
).to_latex(multicolumn=True, multirow=True, float_format=".4f")

with open("results/best_attack_results_table.tex", "w") as f:
    f.write(latex_table)

grouped_df

# %% [markdown]
# ### Boxplot showing number of requests and number of malicious requests

# %%
query_data = []

for dataset_name, dataset_results in results_datasets.items():
    for network_type, network_results in dataset_results.items():
        for attack_type, metrics in network_results.items():            
            for metric in metrics:
                m = {
                    "Attack Type": attack_types.get(attack_type, attack_type.upper()),
                    "Requisições": metric['query_stats']['n_queries'],
                    "Requisições Maliciosas": metric['query_stats']['malicious_queries'],
                }
                query_data.append(m)
            
# Create a pd table
df = pd.DataFrame(query_data)

# Plot boxplot of number of queries and number of malicious queries per attack type
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
attack_types_list = df['Attack Type'].unique()
for i, col in enumerate(['Requisições', 'Requisições Maliciosas']):
    ax = axes[i]
    data = [df[df['Attack Type'] == at][col] for at in attack_types_list]
    ax.boxplot(
        data,
        tick_labels=attack_types_list,
        widths=0.6,
        showfliers=True,
        patch_artist=False,
    )
    ax.set_ylabel(col, fontsize=15)
    ax.set_yscale('log')
    # ax.set_title(f'Distribuição de {col} por Tipo de Ataque', fontsize=16)
    ax.grid(
        True,
        axis='y',
        linestyle='--',
        linewidth=0.6,
        alpha=0.6
    )
    
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        
plt.tight_layout()

fig.savefig("results/query_stats_boxplots.pdf", format='pdf')
fig.savefig("results/query_stats_boxplots.png", format='png', dpi=300)

# %% [markdown]
# ### Boxplot showing evasion success rate and average distance

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

n_datasets = len(results_datasets)
n_rows = (n_datasets + 1) // 2

fig, axes = plt.subplots(
    n_rows,
    1,
    figsize=(8, 4.5 * n_rows),
    squeeze=False
)

if n_datasets == 1:
    axes = np.array([axes[0, 0]])

for idx, (dataset_name, metrics) in enumerate(results_datasets.items()):
    ax = axes[idx]
    
    network_types = list(metrics.keys())
    attacks = list(metrics[network_types[0]].keys())
    
    data_accuracy = []
    data_distance = []
    
    for attack in attacks:
        data_attack = {
            "accuracy": [],
            "distance": []
        }

        for network_type in network_types:
            data_attack["accuracy"].extend([ metric['accuracy'] * 100 for metric in metrics[network_type][attack]])
            data_attack["distance"].extend([ metric['average_distance'] for metric in metrics[network_type][attack]])
            
        data_accuracy.append(data_attack["accuracy"])
        data_distance.append(data_attack["distance"])
        
    x = np.arange(1, len(attacks) + 1)
    offset = 0.18
            
    bp_acc = ax.boxplot(
        data_accuracy,
        tick_labels=[attack_types.get(attack, attack.upper()) for attack in attacks],
        positions=x - offset,
        widths=0.3,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor='tab:blue', alpha=0.6),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='tab:blue', alpha=0.5),
    )
    
    ax2 = ax.twinx()
    
    bp_dist = ax2.boxplot(
        data_distance,
        tick_labels=[attack_types.get(attack, attack.upper()) for attack in attacks],
        positions=x + offset,
        widths=0.3,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor='tab:orange', alpha=0.6),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='tab:orange', alpha=0.5),
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(
        [attack_types.get(attack, attack.upper()) for attack in attacks],
        fontsize=14
    )
    
    ax.set_ylabel('Taxa de Sucesso de Evasão (%)', fontsize=14)
    ax2.set_ylabel('Distância Média', fontsize=14)

    # Axis formatting
    ax.set_ylim(0.0, 100.0 + 5)
    ax2.set_yscale('log')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)

    ax.grid(
        True,
        axis='y',
        linestyle='--',
        linewidth=0.6,
        alpha=0.6
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(dataset_names[dataset_name], fontsize=16)
    
    legend_elements = [
        Patch(facecolor='tab:blue', label='Sucesso (%)', alpha=0.6),
        Patch(facecolor='tab:orange', label='Distância', alpha=0.6),
    ]

    ax.legend(
        handles=legend_elements,
        loc='best',
        fontsize=10
    )
    
plt.tight_layout()

# Save figure as pdf
fig.savefig("results/attack_results.pdf", format='pdf')
fig.savefig("results/attack_results.png", format='png', dpi=300)