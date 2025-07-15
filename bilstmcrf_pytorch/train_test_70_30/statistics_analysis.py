"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy.stats import friedmanchisquare, wilcoxon
import pandas as pd

def read_json_file(file_name):
    with open(file_name, 'r') as file:
        list_f1_score = json.load(file)
    return list_f1_score

list_f1_score_split_1_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_1\macro_f1_scores_70_30.json"
list_f1_score_split_2_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_2\macro_f1_scores_70_30.json"
list_f1_score_split_3_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_3\macro_f1_scores_70_30.json"
list_f1_score_split_4_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_4\macro_f1_scores_70_30.json"
list_f1_score_split_5_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_5\macro_f1_scores_70_30.json"
list_f1_score_split_6_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_6\macro_f1_scores_70_30.json"
list_f1_score_split_7_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_7\macro_f1_scores_70_30.json"
list_f1_score_split_8_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_8\macro_f1_scores_70_30.json"
list_f1_score_split_9_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_9\macro_f1_scores_70_30.json"
list_f1_score_split_10_path = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_10\macro_f1_scores_70_30.json"

# Open and read the JSON file
list_f1_score_split_1 = read_json_file(list_f1_score_split_1_path)
list_f1_score_split_2 = read_json_file(list_f1_score_split_2_path)
list_f1_score_split_3 = read_json_file(list_f1_score_split_3_path)
list_f1_score_split_4 = read_json_file(list_f1_score_split_4_path)
list_f1_score_split_5 = read_json_file(list_f1_score_split_5_path)
list_f1_score_split_6 = read_json_file(list_f1_score_split_6_path)
list_f1_score_split_7 = read_json_file(list_f1_score_split_7_path)
list_f1_score_split_8 = read_json_file(list_f1_score_split_8_path)
list_f1_score_split_9 = read_json_file(list_f1_score_split_9_path)
list_f1_score_split_10 = read_json_file(list_f1_score_split_10_path)

## Combine data for easier handling
data = {
    "split_1": list_f1_score_split_1,
    "split_2": list_f1_score_split_2,
    "split_3": list_f1_score_split_3,
    "split_4": list_f1_score_split_4,
    "split_5": list_f1_score_split_5,
    "split_6": list_f1_score_split_6,
    "split_7": list_f1_score_split_7,
    "split_8": list_f1_score_split_8,
    "split_9": list_f1_score_split_9,
    "split_10": list_f1_score_split_10
}

df = pd.DataFrame.from_dict(data)
df = df.T
print(df)
df.to_csv(r"bilstmcrf_pytorch\train_test_70_30\metrics\metrics_27_models.csv")
# Reorganizar os dados por modelo
num_models = len(data['split_1'])
models_data = [[] for _ in range(num_models)]

for split_scores in data.values():
    for model_idx, score in enumerate(split_scores):
        models_data[model_idx].append(score)

# Converter para array numpy
models_array = np.array(models_data)

print(models_array)
# 1. Teste de Friedman
friedman_stat, friedman_p = friedmanchisquare(*models_data)

print(f'Teste de Friedman:')
print(f'Estatística = {friedman_stat:.4f}, p-valor = {friedman_p:.4f}')

if friedman_p < 0.05:
    print('\nHá diferenças significativas entre os modelos (p < 0.05)')
    
    # 2. Testes Post-Hoc com Wilcoxon e correção de Holm
    pairs = list(combinations(range(num_models), 2))
    p_values = []
    
    for i, j in pairs:
        _, p = wilcoxon(models_array[i], models_array[j])
        p_values.append(p)
    
    # Aplicar correção de Holm
    reject, p_corrected, _, _ = multipletests(p_values, method='holm')
    
    # Coletar pares significativos
    significant_pairs = []
    for idx, (i, j) in enumerate(pairs):
        if reject[idx]:
            significant_pairs.append((
                (i, j),
                p_corrected[idx]
            ))
    
    print(f'\nNúmero de comparações significativas: {len(significant_pairs)}')
    print('\nPares significativos (pós-correção):')
    for pair, p in sorted(significant_pairs, key=lambda x: x[1]):
        print(f'Modelo {pair[0]+1} vs {pair[1]+1}: p = {p:.4f}')

else:
    print('\nNão há diferenças significativas entre os modelos (p ≥ 0.05)')
"""
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('bilstmcrf_pytorch/train_test_70_30/metrics/metrics_27_models.csv', index_col=0)
df.columns = [f'model_{i+1}' for i in range(27)]  # Renomear colunas para model_0 a model_26

# Preparar dados em formato longo
data_long = df.reset_index().melt(id_vars='index', var_name='Model', value_name='F1')
data_long = data_long.rename(columns={'index': 'Split'})

# Teste de Friedman
data_matrix = [df[model].values for model in df.columns]
friedman_stat, friedman_p = friedmanchisquare(*data_matrix)

print(f"Teste de Friedman:\nEstatística = {friedman_stat:.4f}, p-valor = {friedman_p:.4f}")

if friedman_p < 0.05:
    print("\nHá diferenças significativas entre os modelos. Aplicando post-hoc Nemenyi...")
    # Post-hoc Nemenyi
    nemenyi_results = posthoc_nemenyi_friedman(df)
    # Visualizar matriz de p-values
    plt.figure(figsize=(15, 10))
    sns.heatmap(nemenyi_results < 0.05, cmap='Blues', annot=False)
    plt.title("Significant differences (p < 0.05) between models (Nemenyi)")
    #plt.savefig(r'bilstmcrf_pytorch\train_test_70_30\metrics\nemenyi_results.jpg')
    plt.show()
else:
    print("\nNão há diferenças significativas entre os modelos.")

# Estatísticas descritivas
mean_f1 = df.mean().sort_values(ascending=False)
std_f1 = df.std()

print("\nMédias de F1 por modelo (ordenadas):")
print(mean_f1.to_string(float_format='%.4f'))
# Gráfico de caixa
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("F1-score distribution by Model")
plt.ylabel("F1-Score")
#plt.savefig(r'bilstmcrf_pytorch\train_test_70_30\metrics\boxplot.jpg')
plt.show()
"""
# Step 1: Descriptive Statistics
def descriptive_stats(data):
    print("=== Descriptive Statistics ===")
    for split, scores in data.items():
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)
        var = np.var(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        print(f"{split}: Mean={mean:.4f}, Median={median:.4f}, Std={std:.4f}, Var={var:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")

descriptive_stats(data)

# Step 2: Normality Test (Shapiro-Wilk)
print("\n=== Normality Test (Shapiro-Wilk) ===")
for split, scores in data.items():
    stat, p = stats.shapiro(scores)
    print(f"{split}: Statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print(f"  {split} appears to be normally distributed.")
    else:
        print(f"  {split} does not appear to be normally distributed.")



# Converter dicionário para array
f1_array = np.array(list(data.values()))  # Shape (n_splits, n_models)

# Teste de Friedman
t_stat, p_value_friedman = stats.friedmanchisquare(*f1_array.T)
print(f"Friedman test statistic: {t_stat:.4f}, p-value: {p_value_friedman:.4f}")

# Se o p-valor do Friedman for significativo, aplica Wilcoxon entre pares
if p_value_friedman < 0.05:
    print("P-valor significativo para Friedman, aplicando Wilcoxon post-hoc...")
    models = list(data.keys())
    pairs = list(combinations(range(len(models)), 2))  # Todas as combinações de pares de modelos
    p_values = []
    pair_names = []
    
    for i, j in pairs:
        stat, p = stats.wilcoxon(f1_array[:, i], f1_array[:, j])
        p_values.append(p)
        pair_names.append(f"{models[i]} vs {models[j]}")
    
    # Correção de múltiplas comparações (FDR - Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Exibir resultados
    for (pair, p, p_corr) in zip(pair_names, p_values, p_adj):
        print(f"{pair}: p-value = {p:.4f}, adjusted p-value = {p_corr:.4f}")
# Step 8: Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([data["split_1"], data["split_2"], data["split_3"], data["split_4"], data["split_5"]
             ,data["split_6"], data["split_7"], data["split_8"], data["split_9"], data["split_10"]], 
            labels=["split_1", "split_2", "split_3", "split_4", "split_5", 
                    "split_6", "split_7", "split_8", "split_9", "split_10"])

plt.title("Boxplot of Macro F1-Scores Across Splits")
plt.ylabel("F1-Score")
plt.show()

"""

