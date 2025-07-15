import json
from tqdm import tqdm

def read_data(file_name):
    return json.load(open(file_name, encoding="utf-8"))


def read_idx(file_name):
    print("reading ...")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx


def construct_prompt(train_data, train_tables, test_data, example_idx=None, example_num=1):
    print("prompt ...")

    def get_example(index):
        exampel_prompt = ""
        for idx_ in example_idx[index][:example_num]:
            id = int(train_data[idx_]["id"])
            text = train_data[idx_]["text"]
            table = train_tables[idx_]

            # Ad ID
            exampel_prompt += f"O laudo exemplo: Id ({id}) {text}\n"
            exampel_prompt += f"O laudo exemplo com a tabela preenchida: {table}\n"
        return exampel_prompt
        
    results = []
    inputs = []

    for item_idx in tqdm(range(len(test_data))):

        item_ = test_data[item_idx]
        id = item_["id"]
        text = item_["text"]

        # PROMPT 1
        """
        prompt = Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"Id do laudo" : "",
"O nódulo é sólido?" : "",
"O nódulo é em partes moles, semissólido ou subsólido?" : "",
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
"O nódulo é sólido?" : "",
"O nódulo é em partes moles, semissólido ou subsólido?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

A seguir são descritos alguns requisitos para extração:
1. Por favor extraia informações estruturadas para o nódulo pulmonar mencionado no laudo para preencher a tabela. Nesse processo você deve desconsiderar todos os achados descritos no laudo exceto: nódulos, imagem ovalar hiperdensa ou imagem hiperatenuante ovalar.
2. Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. 
3. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.
4. Se o laudo contiver mais de um nódulo descrito crie a quantidade de tabelas necessárias para armazenar as informações relevantes de todos os nódulos pulmonares.

Aqui são descritos alguns pontos de conhecimento médico prévio para sua referência
1. Imagem ovalar hiperdensa deve ser considerada como nódulo pulmonar calcificado.
2. Sólido, partes moles e vidro fosco, são mutuamente exclusivas. Apenas uma das três perguntas pode ser "Sim", e a 
opacidade mista em vidro fosco significa que o tumor tem componentes de opacidade sólidos e em vidro fosco.
3. Micronódulo é um nódulo no pulmão com menos de 3 milímetros (mm) de diâmetro. Nesse contexto devido as suas pequenas dimensões não estamos interessados em extrair suas características. Portanto, não deve ser extraída as suas características.

Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

"""
        
        prompt += get_example(index=item_idx)
        prompt += '\n'

        input = f"Dado o laudo: Id ({id}) {text}\nRetornar a tabela do laudo preenchida no formato JSON:\n\n"

        inputs.append(input)
        results.append(prompt)
    
    return inputs, results


if __name__ == '__main__':
    train_samples = read_data("doc_similarity/data/train_samples.json")
    test_samples = read_data("doc_similarity/data/test_samples.json")
    train_tables = read_data("doc_similarity/data/train_tabels.json")

    example_idx = read_idx(r"data\all_lung_nodule\few_shot\test.100.simcse.dev.5.knn.jsonl")
    inputs, prompts = construct_prompt(train_data=train_samples, train_tables=train_tables, test_data=test_samples, example_idx=example_idx, example_num=5)

    #with open(r"data\all_lung_nodule\inputs.txt", encoding="utf-8", mode="w") as txt_file:
    #    for line in inputs:
    #        txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    
    #with open(r"data\all_lung_nodule\few_shot\prompt_2_five_ex.txt", encoding="utf-8", mode="w") as txt_file:
    #    for line in prompts:
    #        txt_file.write("".join(line) + "\n") # works with any number of elements in a line