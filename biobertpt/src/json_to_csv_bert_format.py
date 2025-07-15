import json
import pandas as pd

def process_reports(json_file_path, csv_file_path):
    """
    Reads a JSON file, processes the data, and saves it to a CSV file.

    Args:
        json_file_path (str): The path to the JSON file.
        csv_file_path (str): The path to the output CSV file.
    """

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_rows = []

    for report_data in data:
        report_index = report_data['report_index']
        sentence_data = report_data['sentence']

        # Split the sentence data into four parts
        num_parts = 4
        words_per_part = len(sentence_data) // num_parts

        for i in range(num_parts):
            start = i * words_per_part
            end = (i + 1) * words_per_part if i < num_parts - 1 else len(sentence_data)
            part_data = sentence_data[start:end]

            # Extract text and labels
            text = ' '.join([str(item[0]) for item in part_data])
            labels = ' '.join([str(item[1]) for item in part_data])

            all_rows.append([report_index, text, labels])

    df = pd.DataFrame(all_rows, columns=['report', 'text', 'iob_labels'])
    df.to_csv(csv_file_path, index=False)

# Example usage
#input_file_name = "/home/tarcisiolf/Documents/Mestrado/Lung_RADS_Automation/1_bilstmcrf_pytorch/train_test_70_30/data/train_0/train_data.json"
#output_file_name = "/home/tarcisiolf/Documents/Mestrado/biobertpt_modulirize/data/split_0/df_train_tokens_labeled_iob_bert_format.csv"
input_file_name = "bilstmcrf_pytorch/lung_rads_data/test_data.json"
output_file_name = "biobertpt/lung_rads/df_test_tokens_labeled_iob_bert_format.csv"
process_reports(input_file_name, output_file_name)