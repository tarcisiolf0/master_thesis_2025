def read_input_file(filename):
    with open(filename, encoding="utf-8", mode='r') as file:
    # Leia todas as linhas do arquivo
        lines = file.read()

    # Divida o texto em exemplos individuais
    input_file = lines.split("\n\n\n")

    return input_file

def read_input_file_as_list(filename):
    with open(filename, encoding='utf-8', mode='r') as file:
    # Leia todas as linhas do arquivo
        lines = file.read()

    results = lines.split("\n\n")

    return results

def read_prompt_file(filename):
    with open(filename, encoding="utf-8", mode='r') as file:
    # Leia todas as linhas do arquivo
        prompt_file = file.read()
    return prompt_file

def write_output_file(filename, results):
    with open(filename, encoding="utf-8", mode="w") as txt_file:
        for line in results:
            txt_file.write("".join(line) + "\n\n") # works with any number of elements in a line

def pre_process_input_file(file):
    file.pop(-1)
    return file

def print_execution_stats(i, execution_time, total_execution_time):
    print("Index_Report {} || Exec_Time {} || Total_Time {}\n\n".format(i, execution_time, total_execution_time))