import time
from openai import OpenAI
import process_files as pf

def gpt_req(inputs, prompts):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):
        
        prompt = prompts[i]
        input = inputs[i]
        input_results = []

        for j in range(3):
            client = OpenAI(api_key="")

            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input}
                ],
                seed=42,
                temperature=0
            )

            text_response = response.choices[0].message.content
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}:\n {text_response}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)
            
        results.append(input_results)

    return results
    

if __name__ == '__main__':
    # Prompt 1 - 5 Examples
    inputs = pf.read_input_file("data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    prompts_one_five_ex = pf.read_input_file("data/prompt_1_five_ex.txt")
    prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    results_one_five_ex = gpt_req(inputs, prompts_one_five_ex)
    pf.write_output_file("data/gpt_results/results_prompt_1_five_ex.txt", results_one_five_ex)
   
    # Prompt 1 - 10 Examples
    prompts_one_ten_ex = pf.read_input_file("data/prompt_1_ten_ex.txt")
    prompts_one_ten_ex = pf.pre_process_input_file(prompts_one_ten_ex)

    results_one_ten_ex = gpt_req(inputs, prompts_one_ten_ex)
    pf.write_output_file("data/gpt_results/results_prompt_1_ten_ex.txt", results_one_ten_ex)

    # Prompt 2 - 5 Examples
    prompts_two_five_ex = pf.read_input_file("data/prompt_2_five_ex.txt")
    prompts_two_five_ex = pf.pre_process_input_file(prompts_two_five_ex)

    results_two_five_ex = gpt_req(inputs, prompts_two_five_ex)
    pf.write_output_file("data/gpt_results/results_prompt_2_five_ex.txt", results_two_five_ex)

    # Prompt 2 - 10 Examples
    prompts_two_ten_ex = pf.read_input_file("data/prompt_2_ten_ex.txt")
    prompts_two_ten_ex = pf.pre_process_input_file(prompts_two_ten_ex)

    results_two_ten_ex = gpt_req(inputs, prompts_two_ten_ex)
    pf.write_output_file("data/gpt_results/results_prompt_2_ten_ex.txt", results_two_ten_ex)