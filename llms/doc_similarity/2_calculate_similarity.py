import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util


# Download necessary NLTK data (do this once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Text preprocessing function (Portuguese)
def preprocess_text_pt(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(words)


if __name__ == '__main__':
    # Load your data (replace with your actual file paths)
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Apply preprocessing
    train_df['Texto'] = train_df['Texto'].apply(preprocess_text_pt)
    test_df['Texto'] = test_df['Texto'].apply(preprocess_text_pt)

    # Carregue o modelo pré-treinado em português
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Supondo que seus dataframes sejam chamados 'df_train' e 'test_df'
    # Gere os embeddings para os textos de treino
    embeddings_train = model.encode(train_df['Texto'].tolist())

    # Gere os embeddings para os textos de teste
    embeddings_test = model.encode(test_df['Texto'].tolist())

    # Crie um dicionário para armazenar os resultados
    resultados = {
        'N_laudo_teste': [],
        '1_similar': [],
        '5_similares': [],
        '10_similares': []
    }

    # Para cada laudo de teste, encontre os documentos mais similares
    for i, embedding_test in enumerate(embeddings_test):
        similarities = util.cos_sim(embedding_test, embeddings_train)
        
        # Ordene os resultados por similaridade (do maior para o menor)
        results = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i], reverse=True)
        
        # Adicione os resultados ao dicionário
        resultados['report'].append(test_df['Laudo'][i])
        #resultados['1_similar'].append(train_df['Laudo'][results[0]])
        resultados['5_similares'].append(train_df['Laudo'][results[:5]].tolist())
        resultados['10_similares'].append(train_df['Laudo'][results[:10]].tolist())
    
    # Crie um DataFrame a partir do dicionário
    df_resultados = pd.DataFrame(resultados)

    # Salve o DataFrame em um arquivo CSV (opcional)
    df_resultados.to_csv('similarity_results.csv', index=False)

    # Exiba o DataFrame de resultados
    print(df_resultados)