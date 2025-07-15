import google.generativeai as genai
import time
import process_files as pf

def gemini_req(inputs, prompt):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):     
        input = inputs[i]      
        input_results = []

        for j in range(3): 
            genai.configure(api_key="")
            model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", "temperature" : 0})
            start_time = time.time()      
            
            final_prompt = ''
            final_prompt = input+"\n\n"+prompt

            response = model.generate_content(final_prompt)
            #print(response.text)
            
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}: {response.text}") # Indicate which request it is
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.text)
            
            # Sleep to avoid hitting the rate limit
            time.sleep(5)

        results.append(input_results)
    return results


if __name__ == '__main__':
    
    # Prompt 1
    inputs = pf.read_input_file("zero_shot/data/inputs.txt")
    prompt_1 = pf.read_prompt_file("zero_shot/data/prompt_1.txt")

    inputs = pf.pre_process_input_file(inputs)

    results_1 = gemini_req(inputs, prompt_1)
    pf.write_output_file("zero_shot/data/gemini_results/results_prompt_1.txt", results_1)

    # Prompt 2
    prompt_2 = pf.read_prompt_file("zero_shot/data/prompt_2.txt")
    results_2 = gemini_req(inputs, prompt_2)
    pf.write_output_file("zero_shot/data/gemini_results/results_prompt_2.txt", results_2)

    
