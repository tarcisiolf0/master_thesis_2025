import google.generativeai as genai
import os
import time
import process_files as pf

def gemini_req(inputs, prompts):
    results = []
    total_exec_time = 0

    genai.configure(api_key="")

    for i in range(len(inputs)):     
        input = inputs[i]
        input_results = []

        prompt = prompts[i]

        for j in range(3):
            model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", "temperature" : 0})

            start_time = time.time()   

            final_prompt = ''
            final_prompt = input+"\n\n"+prompt

            response = model.generate_content(final_prompt)

            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}:\n {response.text}") # Indicate which request it is
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.text)
        
            # Sleep to avoid hitting the rate limit
            time.sleep(5)
        
        results.append(input_results)
        
    return results


if __name__ == '__main__':
    
    # Prompt 1 - 5 Examples
    inputs = pf.read_input_file("few_shot/data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    prompts_one_five_ex = pf.read_input_file("few_shot/data/prompt_1_five_ex.txt")
    prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    results_one_five_ex = gemini_req(inputs, prompts_one_five_ex)
    pf.write_output_file("few_shot/data/gemini_results/results_prompt_1_five_ex.txt", results_one_five_ex)

    # Prompt 1 - 10 Examples
    prompts_one_ten_ex = pf.read_input_file("few_shot/data/prompt_1_ten_ex.txt")
    prompts_one_ten_ex = pf.pre_process_input_file(prompts_one_ten_ex)

    results_one_ten_ex = gemini_req(inputs, prompts_one_ten_ex)
    pf.write_output_file("few_shot/data/gemini_results/results_prompt_1_ten_ex.txt", results_one_ten_ex)

    # Prompt 2 - 5 Examples
    prompts_two_five_ex = pf.read_input_file("few_shot/data/prompt_2_five_ex.txt")
    prompts_two_five_ex = pf.pre_process_input_file(prompts_two_five_ex)

    results_two_five_ex = gemini_req(inputs, prompts_two_five_ex)
    pf.write_output_file("few_shot/data/gemini_results/results_prompt_2_five_ex.txt", results_two_five_ex)

    # Prompt 2 - 10 Examples
    prompts_two_ten_ex = pf.read_input_file("few_shot/data/prompt_2_ten_ex.txt")
    prompts_two_ten_ex = pf.pre_process_input_file(prompts_two_ten_ex)

    results_two_ten_ex = gemini_req(inputs, prompts_two_ten_ex)
    pf.write_output_file("few_shot/data/gemini_results/results_prompt_2_ten_ex.txt", results_two_ten_ex)

