import pandas as pd

input_file_name = 'bilstmcrf_pytorch/df_tokens_labeled_iob.csv'
input_aux_file_name = 'llms/doc_similarity/data/test.csv'
output_file_name = 'bilstmcrf_pytorch/lung_rads_data/lung_rads_test.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file_name)
df_llms = pd.read_csv(input_aux_file_name)

list_of_index = df_llms.Laudo.to_list()
list_of_index = [int(x) for x in list_of_index]

# Filter the DataFrame based on the list of indices
df_filtered = df[df['report_index'].isin(list_of_index)]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file_name, index=False)

print(f"Filtered data saved to {output_file_name}")