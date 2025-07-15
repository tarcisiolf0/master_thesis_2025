import json
import os
from pathlib import Path

def extract_macro_f1_scores(directory_path):
    """
    Extract macro average f1-scores from all JSON files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
        
    Returns:
        dict: Dictionary with filenames as keys and macro f1-scores as values
    """
def extract_macro_f1_scores(directory_path):
    """
    Extract macro average f1-scores from all JSON files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
        
    Returns:
        list: List of macro f1-scores
    """

    scores = []
    
    # Create Path object
    dir_path = Path(directory_path)
    
    # Iterate through all JSON files in the directory
    for json_file in dir_path.glob('*.json'):
        try:
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract macro average f1-score
            if 'macro avg' in data and 'f1-score' in data['macro avg']:
                macro_f1 = data['macro avg']['f1-score']
                scores.append(macro_f1)
            else:
                print(f"Warning: Required fields not found in {json_file.name}")
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {json_file.name}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
    
    return scores

def save_results(results, output_file):
    """
    Save results to a JSON file.
    
    Args:
        results (dict): Dictionary containing the results
        output_file (str): Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def main():
    # Directory containing the JSON files
    input_directory = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_10"  # Replace with your directory path
    
    # Output file path
    output_file = r"bilstmcrf_pytorch\train_test_70_30\metrics\train_10\macro_f1_scores_70_30.json"
    
    # Extract scores
    results = extract_macro_f1_scores(input_directory)
    
    # Save results
    save_results(results, output_file)
    
    print(f"Processed {len(results)} files. Results saved to {output_file}")

if __name__ == "__main__":
    main()