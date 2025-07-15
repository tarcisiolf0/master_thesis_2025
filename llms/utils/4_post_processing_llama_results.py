import re

def extract_json_tables(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match JSON-like tables inside { }
    json_matches = re.findall(r'\{[^{}]*\}', content, re.DOTALL)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for match in json_matches:
            out_file.write(match + '\n\n')

# Example usage
extract_json_tables("llms/zero_shot/data/llama_results/results_prompt_2.txt", "llms/zero_shot/data/llama_results/results_prompt_2_post_processed.txt")