import json
from tqdm import tqdm
import csv

def read_data(file_name):
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(file_name):
    print("reading ...")
    example_idx = []
    with open(file_name, "r") as file:
        for line in file:
            example_idx.append(json.loads(line.strip()))
    return example_idx

def read_similarity_results(file_name):
    similarity_results = {}
    with open(file_name, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            report_id = int(row['report'])
            similar_5 = json.loads(row['5_similares'])
            similar_10 = json.loads(row['10_similares'])
            similarity_results[report_id] = {'5_similares': similar_5, '10_similares': similar_10}
    return similarity_results


def construct_prompt(train_data, train_tables, test_data, similarity_results, example_num):
    print("prompt ...")

    id_to_list_idx = {item['id']: i for i, item in enumerate(train_data)}

    def get_example(index):
        example_prompt = ""
        for idx_ in similarity_results[index]['10_similares'][:example_num]:
            idx_ = int(idx_)
            #print(idx_)
            real_idx = id_to_list_idx.get(idx_)
            #print(read_idx)
            id = int(train_data[real_idx]["id"])
            text = train_data[real_idx]["text"]
            table = train_tables[real_idx]

            # Ad ID
            example_prompt += f"O laudo exemplo: Id ({id}) {text}\n"
            example_prompt += f"O laudo exemplo com a tabela preenchida: {table}\n"
        return example_prompt

    results = []
    inputs = []

    for item_idx in tqdm(range(len(test_data))):
        
        item_ = test_data[item_idx]
        id = item_['id']
        text = item_['text']

        """
        # PROMPT 1
        prompt = Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"Id do laudo": "",
"O nódulo é sólido ou em partes moles?" : "",
"O nódulo tem densidade semissólida ou parcialmente sólida?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.

Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

"""
        
        # PROMPT 2
        prompt = """Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"Id do laudo" : "",
"O nódulo é sólido ou em partes moles?" : "",
"O nódulo tem densidade semissólida ou parcialmente sólida?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

A seguir são descritos alguns requisitos para extração:
1. Por favor extraia informações estruturadas para o nódulo pulmonar mencionado no laudo para preencher a tabela. Nesse processo você deve desconsiderar todos os achados descritos no laudo exceto: nódulos, imagem ovalar hiperdensa ou imagem hiperatenuante ovalar.
2. Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. 
3. A pergunta do tamanho do nódulo deve ser respondida apenas com números e a unidade de medida.

Aqui são descritos alguns pontos de conhecimento médico prévio para sua referência
1. Imagem ovalar hiperdensa deve ser considerada como nódulo pulmonar calcificado.
2. Sólido, partes moles e vidro fosco, são mutuamente exclusivas. Apenas uma das três perguntas pode ser "Sim", e a 
opacidade mista em vidro fosco significa que o tumor tem componentes de opacidade sólidos e em vidro fosco.
3. Micronódulo é um nódulo no pulmão com menos de 3 milímetros (mm) de diâmetro. Nesse contexto devido as suas pequenas dimensões não estamos interessados em extrair suas características. Portanto, não deve ser extraída as suas características.

Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

"""
        
        
        

        prompt += get_example(index=item_['id']) # Usando o ID do item atual para buscar exemplos similares
        prompt += '\n'

        input = f"Dado o laudo: Id ({id}) {text}\nRetornar a tabela do laudo preenchida no formato JSON:\n\n"

        inputs.append(input)
        results.append(prompt)
        
    return inputs, results


if __name__ == '__main__':
    train_samples = read_data("doc_similarity/data/train_samples.json")
    test_samples = read_data("doc_similarity/data/test_samples.json")
    train_tables = read_data("doc_similarity/data/train_tabels.json")
    similarity_results = read_similarity_results("doc_similarity/data/similarity_results.csv") # Lendo os resultados de similaridade

    inputs, prompts = construct_prompt(train_data=train_samples, train_tables=train_tables, test_data=test_samples, 
                                       similarity_results=similarity_results, example_num=10)

    """
    with open("few_shot/data/inputs.txt", encoding="utf-8", mode="w") as txt_file:
        for line in inputs:
            txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    """

    """
    with open("few_shot/data/prompt_2_ten_ex.txt", encoding="utf-8", mode="w") as txt_file:
        for line in prompts:
            txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    """