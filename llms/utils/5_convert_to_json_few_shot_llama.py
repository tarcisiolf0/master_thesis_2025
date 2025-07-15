import json

def convert_to_jsonl(input_file_path, output_file_path):
    """
    Converts a text file with multiple JSON-like dictionaries into a JSONL file.

    Args:
        input_file_path (str): The path to the input text file.
        output_file_path (str): The path to the output JSONL file.
    """
    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        lines = input_file.readlines()

        # Group dictionaries with the same 'Id do laudo'
        groups = {}
        for line in lines:
            line = line.strip()
            if line:
                try:
                    dictionary = eval(line)  # Evaluate the string as a dictionary
                    report_id = dictionary['Id do laudo']
                    if report_id not in groups:
                        groups[report_id] = []
                    groups[report_id].append(dictionary)
                except (SyntaxError, NameError, TypeError) as e:
                    print(f"Error processing line: {line}. Error: {e}")
                    continue

        # Write each group as a JSONL line
        for group in groups.values():
            json.dump(group, output_file, ensure_ascii=False)
            output_file.write('\n')

# Replace 'your_file.txt' and 'your_file.jsonl' with the actual file paths
convert_to_jsonl(r'3_llms\few_shot\data\llama_results\results_prompt_2_five_ex_post_processed.txt', r'3_llms\few_shot\data\llama_results\results_prompt_2_five_ex.jsonl')
