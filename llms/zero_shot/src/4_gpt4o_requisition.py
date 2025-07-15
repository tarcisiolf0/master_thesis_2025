import time
from openai import OpenAI

import process_files as pf

def gpt_req(inputs, prompt):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):
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
            
            print(f"Request {j+1} for input {i+1}: {text_response}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)

        results.append(input_results)
    return results
    

if __name__ == '__main__':
    # Prompt 1
    inputs = pf.read_input_file("llms/zero_shot/data/inputs.txt")
    prompt_1 = pf.read_prompt_file("llms/zero_shot/data/prompt_1.txt")

    inputs = pf.pre_process_input_file(inputs)

    results_1 = gpt_req(inputs, prompt_1)
    pf.write_output_file("llms/zero_shot/data/gpt_results/results_prompt_1.txt", results_1)

    # Prompt 2
    prompt_2 = pf.read_prompt_file("llms/zero_shot/data/prompt_2.txt")
    results_2 = gpt_req(inputs, prompt_2)
    pf.write_output_file("llms/zero_shot/data/gpt_results/results_prompt_2.txt", results_2)