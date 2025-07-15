import json
from collections import defaultdict

def evaluate_consistency(filepath):
    """
    Evaluates the consistency of LLM responses from a JSONL-formatted file.

    Args:
        filepath: Path to the JSONL file containing the LLM responses.

    Returns:
        A dictionary containing:
            - Overall consistency, inconsistency, and skip percentages.
            - Per-question consistency analysis.
            - Breakdown of consistency per "Id do laudo".
    """
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]  # Each line is a list of 3 JSON objects
    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Invalid JSON format in file."

    num_questions = 0
    num_consistent = 0
    num_inconsistent = 0
    num_skips = 0

    consistency_details = {}  # Store details for each "Id do laudo"
    question_stats = defaultdict(lambda: {"consistent": 0, "inconsistent": 0, "skipped": 0})  # Track question stats

    i = 0
    for line in data:  # Each `line` is a list of 3 JSON objects
        
        if len(line) != 3:  # Ensure each line contains exactly 3 objects
            return "Invalid data format: Each line should contain exactly 3 JSON objects."

        id_laudo = line[0].get('Id do laudo', 'Unknown')  # Use first object's ID
        consistency_details[id_laudo] = {}

        # Get all question keys (excluding 'Id do laudo')
        questions = [key for key in line[0].keys() if key != 'Id do laudo']

        for question in questions:
            #print(i)
            #print(question)
            responses = [entry[question] for entry in line]  # Collect responses from all 3 objects
            num_questions += 1

            if all(r == responses[0] for r in responses):  # Check for consistency
                num_consistent += 1
                consistency_details[id_laudo][question] = "Consistent"
                question_stats[question]["consistent"] += 1
            elif any(r.lower() == "skip" for r in responses):  # Check for "skip"
                num_skips += 1
                consistency_details[id_laudo][question] = "Skip"
                question_stats[question]["skipped"] += 1
            else:
                num_inconsistent += 1
                consistency_details[id_laudo][question] = "Inconsistent"
                question_stats[question]["inconsistent"] += 1
        i = i+1
    if num_questions == 0:  # Avoid division by zero
        return {"error": "No questions found in the data."}

    # Compute overall percentages
    consistency_percentage = (num_consistent / num_questions) * 100
    inconsistency_percentage = (num_inconsistent / num_questions) * 100
    skip_percentage = (num_skips / num_questions) * 100

    # Compute per-question percentages
    per_question_analysis = {
        question: {
            "consistency_percentage": (stats["consistent"] / sum(stats.values())) * 100,
            "inconsistency_percentage": (stats["inconsistent"] / sum(stats.values())) * 100,
            "skip_percentage": (stats["skipped"] / sum(stats.values())) * 100,
            "total_occurrences": sum(stats.values())
        }
        for question, stats in question_stats.items()
    }

    results = {
        "overall_consistency_percentage": consistency_percentage,
        "overall_inconsistency_percentage": inconsistency_percentage,
        "overall_skip_percentage": skip_percentage,
        "per_question_analysis": per_question_analysis,
        "consistency_details_by_laudo": consistency_details
    }

    return results


# Example usage:
input_file_name = "llms/zero_shot/data/gemini_results/results_prompt_1.jsonl"
output_file_name = "llms/zero_shot/data/gemini_results/response_consistency_prompt_1.jsonl"
results = evaluate_consistency(input_file_name)

#if isinstance(results, dict):
#    print(json.dumps(results, indent=4, ensure_ascii=False))  # Print results in formatted JSON
#elif isinstance(results, str):  # Handle errors
#    print(results)

#json.dump(results, open(output_file_name, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=False, indent = 2)