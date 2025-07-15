import pandas as pd
import json
import csv

def jsonl_to_csv(jsonl_file, csv_file):
    """
    Reads a JSONL file and converts it to a CSV file.

    Args:
        jsonl_file (str): Path to the input JSONL file.
        csv_file (str): Path to the output CSV file.
    """
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            first_line = True
            header = None  # Initialize header to None.

            for line in infile:
                try:
                    data = json.loads(line)
                    first_json_element = data[0]
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in line: {line}. Error: {e}")
                    continue #skip the bad line.

                if first_line:
                    #header = list(data.keys())
                    header = list(first_json_element.keys())  # Use keys from first JSON object in the file as header.
                    writer.writerow(header)
                    first_line = False

                #row = [data.get(key, '') for key in header]  # Use .get() to handle missing keys
                row = [first_json_element.get(key, '') for key in header]  # Use keys from first JSON object in the file as header.
                writer.writerow(row)

        print(f"Successfully converted {jsonl_file} to {csv_file}")

    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':

    # Zero Shot
    #input_filename = "llms/zero_shot/data/gpt_results/results_prompt_1.jsonl"
    #output_filename = "llms/zero_shot/data/gpt_results/results_prompt_1_structured.csv"

    # Few Shot
    input_filename = "llms/few_shot/data/gemini_results/results_prompt_1_five_ex.jsonl"
    output_filename = "llms/few_shot/data/gemini_results/results_prompt_1_five_ex_structured.csv"
    jsonl_to_csv(input_filename, output_filename)
