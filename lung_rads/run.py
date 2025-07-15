from classification import classify_nodules
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # Carregar os DataFrames
    df_results_bilstmcrf = pd.read_csv(r"bilstmcrf_pytorch\lung_rads_data\lung_rads_test_predicted_post_processed.csv")
    df_results_biobertpt = pd.read_csv(r"biobertpt\lung_rads_data\lung_rads_test_predicted_post_processed.csv")
    df_results_gemini1_5 = pd.read_csv(r"llms\few_shot\data\gemini_results\results_prompt_2_ten_ex_structured_post_processed.csv")
    df_results_gpt4 = pd.read_csv(r"llms\few_shot\data\gpt_results\results_prompt_2_ten_ex_structured_post_processed.csv")
    df_results_llama3 = pd.read_csv(r"llms\few_shot\data\llama_results\results_prompt_2_ten_ex_structured_post_processed.csv")


    df_structured_data = pd.read_excel('llms\doc_similarity\data\structured_data_for_lung_rads.ods')
    #lung_rads_column = df_structured_data[df_structured_data['Lung-RADS'].notna()]
    df_structured_data_filtered = df_structured_data[~df_structured_data["Laudo"].astype(str).str.contains(r'\.(1|2|3)$', regex=True)]
    lung_rads_column = df_structured_data_filtered['Lung-RADS'].dropna()
    list_lung_rads = lung_rads_column.to_list()
    

    df_results_bilstmcrf.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_biobertpt.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_gemini1_5.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_gpt4.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_llama3.insert(8 ,"Lung-RADS", list_lung_rads)

    # Classificar os nódulos nos DataFrames
    df_results_bilstmcrf['Lung-RADS-Pred'] = classify_nodules(df_results_bilstmcrf)
    df_results_biobertpt['Lung-RADS-Pred'] = classify_nodules(df_results_biobertpt)
    df_results_gemini1_5['Lung-RADS-Pred'] = classify_nodules(df_results_gemini1_5)
    df_results_gpt4['Lung-RADS-Pred'] = classify_nodules(df_results_gpt4)
    df_results_llama3['Lung-RADS-Pred'] = classify_nodules(df_results_llama3)

    # Exibir o DataFrame com as classificações
    
    
    y_true = list_lung_rads
    y_pred_bilstmcrf = df_results_bilstmcrf['Lung-RADS-Pred'].tolist()
    y_pred_biobertpt = df_results_biobertpt['Lung-RADS-Pred'].tolist()
    y_pred_gemini1_5 = df_results_gemini1_5['Lung-RADS-Pred'].tolist()
    y_pred_gpt4o = df_results_gpt4['Lung-RADS-Pred'].tolist()
    y_pred_llama3 = df_results_llama3['Lung-RADS-Pred'].tolist()

    y_true_str = [str(x) for x in y_true]
    y_pred_bilstmcrf = [str(item) for item in y_pred_bilstmcrf]
    y_pred_biobertpt = [str(item) for item in y_pred_biobertpt]
    y_pred_gemini1_5 = [str(item) for item in y_pred_gemini1_5]
    y_pred_gpt4o = [str(item) for item in y_pred_gpt4o]
    y_pred_llama3 = [str(item) for item in y_pred_llama3]


    metrics_bilstmcrf = classification_report(y_true_str, y_pred_bilstmcrf, zero_division=0.0)
    print(metrics_bilstmcrf)
    metrics_biobertpt = classification_report(y_true_str, y_pred_biobertpt, zero_division=0.0)
    print(metrics_biobertpt)
    metrics_gemini1_5 = classification_report(y_true_str, y_pred_gemini1_5, zero_division=0.0)
    print(metrics_gemini1_5)
    metrics_gpt4o = classification_report(y_true_str, y_pred_gpt4o, zero_division=0.0)
    print(metrics_gpt4o)
    metrics_llama3 = classification_report(y_true_str, y_pred_llama3, zero_division=0.0)
    print(metrics_llama3)

    df_results_bilstmcrf.to_csv('lung_rads\data\lung_rads_predictions_bilstmcrf.csv', index=False)
    df_results_biobertpt.to_csv('lung_rads\data\lung_rads_predictions_biobertpt.csv', index=False)
    df_results_gemini1_5.to_csv('lung_rads\data\lung_rads_predictions_gemini1_5.csv', index=False)
    df_results_gpt4.to_csv('lung_rads\data\lung_rads_predictions_gpt4o.csv', index=False)
    df_results_llama3.to_csv('lung_rads\data\lung_rads_predictions_llama3.csv', index=False)
