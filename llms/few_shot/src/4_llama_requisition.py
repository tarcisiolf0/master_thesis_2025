import time
from together import Together
import process_files as pf
import tiktoken  # You'll need to install this: pip install tiktoken

def truncate_and_count_tokens(prompt, input_text, max_new_tokens_target, model_name="cl100k_base"):  # Default encoding
    """
    Truncates the prompt and input to fit within Together's token limits,
    calculates token counts, and returns the adjusted prompt, input, and
    max_new_tokens.

    Args:
        prompt: The system prompt.
        input_text: The user input.
        max_new_tokens_target: The desired number of new tokens.
        model_name: The name of the tiktoken encoding to use.  "cl100k_base" is a good default.
            You might need to adjust this depending on the model you are using.
            See tiktoken's documentation for more details.

    Returns:
        A tuple containing:
            - truncated_prompt: The potentially truncated system prompt.
            - truncated_input: The potentially truncated user input.
            - adjusted_max_new_tokens: The adjusted max_new_tokens.
            - prompt_token_count: The token count of the (truncated) prompt.
            - input_token_count: The token count of the (truncated) input.
            - total_tokens: The sum of prompt_token_count, input_token_count and adjusted_max_new_tokens
    """
    #encoding = tiktoken.encoding_for_model(model_name)  # Get the encoding
    encoding = tiktoken.get_encoding("cl100k_base")

    prompt_tokens = encoding.encode(prompt)
    input_tokens = encoding.encode(input_text)

    prompt_token_count = len(prompt_tokens)
    input_token_count = len(input_tokens)

    MAX_TOTAL_TOKENS = 8193  # Together's limit

    total_tokens_needed = prompt_token_count + input_token_count + max_new_tokens_target

    """
    
    print(prompt_token_count)
    print(input_token_count)
    print(total_tokens_needed)
    """

    if total_tokens_needed > MAX_TOTAL_TOKENS:
        # Truncate the input first, then the prompt if necessary.
        tokens_to_remove = total_tokens_needed - MAX_TOTAL_TOKENS

        # Truncate the prompt first
        if prompt_token_count > tokens_to_remove:
            prompt_tokens = prompt_tokens[:-tokens_to_remove]
            prompt_token_count = len(prompt_tokens)
            adjusted_max_new_tokens = max_new_tokens_target
        else:
            # If the prompt is smaller than the number of tokens to remove,
            # truncate the input as well
            tokens_to_remove -= prompt_token_count
            prompt_tokens = []
            prompt_token_count = 0

            if input_token_count > tokens_to_remove:
                input_tokens = input_tokens[:-tokens_to_remove]
                input_token_count = len(input_tokens)
                adjusted_max_new_tokens = max_new_tokens_target
            else:
                input_tokens = []
                input_token_count = 0
                adjusted_max_new_tokens = MAX_TOTAL_TOKENS - prompt_token_count - input_token_count # In this very unlikely case, where both prompt and input are smaller than needed to remove.


        truncated_prompt = encoding.decode(prompt_tokens)
        truncated_input = encoding.decode(input_tokens)
        total_tokens = prompt_token_count + input_token_count + adjusted_max_new_tokens

        """
        print("Truncated_prompt\n", truncated_prompt)
        print()
        print(len(truncated_prompt))
        print()
        print("Truncated_input\n", truncated_input)
        print()
        print("Adjusted max new tokens\n", adjusted_max_new_tokens)
        print("Prompt count\n",prompt_token_count)
        print("input token count\n",input_token_count)
        print("total tokens\n", total_tokens)
        """
    else:
        truncated_prompt = prompt
        truncated_input = input_text
        adjusted_max_new_tokens = max_new_tokens_target
        total_tokens = total_tokens_needed

    return truncated_prompt, truncated_input, adjusted_max_new_tokens, prompt_token_count, input_token_count, total_tokens



def llama_req(inputs, prompts):
    results = []
    total_exec_time = 0
    max_new_tokens = 2500

    for i in range(len(inputs)):
        input_results = []
        input = inputs[i]      
        prompt = prompts[i]

        truncated_prompt, truncated_input, adjusted_max_new_tokens, prompt_token_count, input_token_count, total_tokens = truncate_and_count_tokens(prompt, input, max_new_tokens)
    
        for j in range(3):
            client = Together(api_key="")

            start_time = time.time()

            """
            
            #model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input}
                ],
                temperature=0
            )
            """

            response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": truncated_prompt},
                {"role": "user", "content": truncated_input}
            ],
            max_new_tokens=adjusted_max_new_tokens,
            temperature=0
            )

            text_response = response.choices[0].message.content
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)

            
        print(f"{text_response}")
        results.append(input_results)
        
    return results
    

if __name__ == '__main__':
    
    inputs = pf.read_input_file("few_shot/data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    """
    # Prompt 1 - 5 Examples
    prompts_one_five_ex = pf.read_input_file("few_shot/data/prompt_1_five_ex.txt")
    prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    results_one_five_ex = llama_req(inputs, prompts_one_five_ex)
    pf.write_output_file("few_shot/data/llama_results/results_prompt_1_five_ex.txt", results_one_five_ex)

    # Prompt 2 - 5 Examples
    prompts_two_five_ex = pf.read_input_file("few_shot/data/prompt_2_five_ex.txt")
    prompts_two_five_ex = pf.pre_process_input_file(prompts_two_five_ex)

    results_two_five_ex = llama_req(inputs, prompts_two_five_ex)
    pf.write_output_file("few_shot/data/llama_results/results_prompt_2_five_ex.txt", results_two_five_ex)
    
    # Prompt 1 - 10 Examples
    prompts_one_ten_ex = pf.read_input_file("few_shot/data/prompt_1_ten_ex.txt")
    prompts_one_ten_ex = pf.pre_process_input_file(prompts_one_ten_ex)

    results_one_ten_ex = llama_req(inputs, prompts_one_ten_ex)
    pf.write_output_file("few_shot/data/llama_results/results_prompt_1_ten_ex.txt", results_one_ten_ex)

    # Prompt 2 - 10 Examples
    prompts_two_ten_ex = pf.read_input_file("few_shot/data/prompt_2_ten_ex.txt")
    prompts_two_ten_ex = pf.pre_process_input_file(prompts_two_ten_ex)

    results_two_ten_ex = llama_req(inputs, prompts_two_ten_ex)
    pf.write_output_file("few_shot/data/llama_results/results_prompt_2_ten_ex.txt", results_two_ten_ex)
    """



