import pandas as pd

def create_inputs_txt(df, file_name):
    """
    Cria um arquivo inputs.txt com o texto dos laudos do DataFrame.

    Args:
        df: DataFrame do pandas contendo os dados dos laudos.
        file_name: Nome do arquivo para salvar os dados.
    """

    with open(file_name, "w", encoding="utf-8") as arquivo:
        for _, row in df.iterrows():
            text= row["Texto"]

            # Escreve no arquivo no formato desejado
            arquivo.write(f"Dado o laudo: {text}\n")
            arquivo.write("Retornar a tabela do laudo preenchida no formato JSON:\n\n\n")

if __name__ == "__main__":
    # Ler um DataFrame do CSV
    df = pd.read_csv("doc_similarity/data/test.csv")
    output_txt_file_name = "zero_shot/data/inputs.txt"
    create_inputs_txt(df, output_txt_file_name)