import time
from together import Together
import process_files as pf


def llama_req(inputs, prompt):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):
        input = inputs[i]      
        input_results = []

        for j in range(3):
            client = Together(api_key="")

            start_time = time.time()

            #model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input}
                ],
                temperature=0
            )

            text_response = response.choices[0].message.content
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            #print(f"Request {j+1} for input {i+1}: {text_response}")
            print(f"Request {j+1} for input {i+1}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)

        print(f"{text_response}")
        results.append(input_results)

    return results
    

if __name__ == '__main__':
   # Prompt 1
    inputs = pf.read_input_file("llms/zero_shot/data/inputs.txt")
    prompt_1 = pf.read_prompt_file("llms/zero_shot/data/prompt_1.txt")

    inputs = pf.pre_process_input_file(inputs)

    results_1 = llama_req(inputs, prompt_1)
    pf.write_output_file("llms/zero_shot/data/llama_results/results_prompt_1.txt", results_1)

    #Prompt 2
    prompt_2 = pf.read_prompt_file("llms/zero_shot/data/prompt_2.txt")
    results_2 = llama_req(inputs, prompt_2)
    pf.write_output_file("llms/zero_shot/data/llama_results/results_prompt_2.txt", results_2)