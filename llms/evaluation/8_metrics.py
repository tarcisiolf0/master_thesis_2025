import pandas as pd
import csv

def compare_answers(answer_train, answer_test):
    if answer_train == True and answer_test == True:
        return 1, 0, 0, 0  # TP, TN, FN, FP
    elif answer_train == False and answer_test == False:
        return 0, 1, 0, 0  # TP, TN, FN, FP
    elif answer_train == True and answer_test == False:
        return 0, 0, 1, 0  # TP, TN, FN, FP
    elif answer_train == False and answer_test == True:
        return 0, 0, 0, 1  # TP, TN, FN, FP
    elif(answer_test == answer_train):
        return 1, 0, 0, 0  # TP, TN, FN, FP
    elif(answer_test != answer_train):
        return 0, 0, 0, 1  # TP, TN, FN, FP
    return 0, 0, 0, 0

def calculate_metrics(df_train, df_test, columns):
    metrics = {col: {'TP': 0, 'TN': 0, 'FN': 0, 'FP': 0} for col in columns}

    for i in range(len(df_train)):
        for col in columns:
            answer_train = df_train[col[0]].iloc[i]
            answer_test = df_test[col[1]].iloc[i]

            TP, TN, FN, FP = compare_answers(answer_train, answer_test)
            metrics[col]['TP'] += TP
            metrics[col]['TN'] += TN
            metrics[col]['FN'] += FN
            metrics[col]['FP'] += FP

    return metrics

def calculate_precision_recall_f1(metrics):
    results = {}
    for col, counts in metrics.items():
        TP = counts['TP']
        TN = counts['TN']
        FN = counts['FN']
        FP = counts['FP']

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

        results[col] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
    
    return results

def save_metrics(csv_filename, results):
    with open(csv_filename, encoding='utf-8', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Coluna", "Precision", "Recall", "F1 Score"])
        for col, result in results.items():
            writer.writerow([col, result['precision'], result['recall'], result['f1_score']])

if __name__=='__main__':

    columns = [
    ("O nódulo é sólido ou em partes moles?", "O nódulo é sólido ou em partes moles?"),
    ("O nódulo tem densidade semissólida ou parcialmente sólida?", "O nódulo tem densidade semissólida ou parcialmente sólida?"),
    ("O nódulo tem densidade em vidro fosco?", "O nódulo é em vidro fosco?"),
    ("O nódulo tem borda espiculada e/ou mal definida?", "O nódulo é espiculado, irregular ou mal definido?"),
    ("O nódulo é calcificado?", "O nódulo é calcificado?"),
    ("Localização do nódulo", "Localização do nódulo"),
    ("Tamanho do nódulo (mm)", "Tamanho do nódulo (mm)")
]
    df_test = pd.read_csv("llms/doc_similarity/data/test_post_processed.csv")

    # Zero shot
    #df_result = pd.read_csv("data/one_lung_nodule/zero_shot/results_gemini/results_prompt_2_structured_post_processing_v2.csv")
    #csv_filename = "data/one_lung_nodule/zero_shot/results_gemini/metrics_prompt_2_structured_v2.csv"

    # Few shot
    df_result = pd.read_csv("llms/few_shot/data/llama_results/results_prompt_1_ten_ex_structured_post_processed.csv")
    csv_filename = "llms/few_shot/data/llama_results/metrics_results_prompt_1_ten_ex_structured_post_processed.csv"

    metrics = calculate_metrics(df_test, df_result, columns)
    results = calculate_precision_recall_f1(metrics)
    save_metrics(csv_filename, results)

    for col, result in results.items():
        print(f"{col}: Precision: {result['precision']}, Recall: {result['recall']}, F1 Score: {result['f1_score']}")

    print("\n\n")