import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
import random

def preprocess_ner_data(csv_file, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_file, encoding='utf-8')
    grouped = df.groupby('report_index')
    sentences = []
    #for _, group in grouped:
    for report_index, group in grouped:
        sentence = list(zip(group['token'], group['iob_tag']))
        #sentences.append(sentence)
        sentences.append({"report_index": report_index, "sentence": sentence})  # Include report_index


    #unique_tags = sorted(list(set([tag for sentence in sentences for _, tag in sentence])))
    unique_tags = sorted(list(set([tag for item in sentences for token, tag in item['sentence']]))) # Corrected line

    if '<PAD>' not in unique_tags:
        unique_tags.append('<PAD>')
    if '<UNK>' not in unique_tags:
        unique_tags.append('<UNK>')

    train_sentences, test_sentences = train_test_split(sentences, test_size=test_size, random_state=random_state)
    return train_sentences, test_sentences, unique_tags

def create_mappings(sentences, unique_tags):
    all_words = [str(token) for sentence in sentences for token, _ in sentence]
    unique_words = sorted(list(set(all_words)))

    word2index = {word: index for index, word in enumerate(unique_words)}
    index2word = {index: word for word, index in word2index.items()}
    word2index['<UNK>'] = len(word2index)
    index2word[len(index2word)] = '<UNK>'
    word2index['<PAD>'] = len(word2index)
    index2word[len(index2word)] = '<PAD>'

    tag2index = {tag: index for index, tag in enumerate(unique_tags)}
    index2tag = {index: tag for tag, index in tag2index.items()}

    return word2index, index2word, tag2index, index2tag

def sentence_to_indices(sentence, word2index, tag2index):
    tokens = [word2index[token] if token in word2index and pd.notna(token) else word2index['<UNK>'] for token, _ in sentence]
    tags = [tag2index[tag] if tag in tag2index and pd.notna(tag) else tag2index['<UNK>'] for _, tag in sentence]
    return tokens, tags

def pad_sequences(sequences, max_len, padding_value):
    padded_sequences = []
    for seq in sequences:
        padding_length = max_len - len(seq)
        padded_seq = seq + [padding_value] * padding_length
        padded_sequences.append(padded_seq)
    return padded_sequences

def process_data(data, max_len, word2index, tag2index):
    sentence_indices = []
    tag_indices = []

    #for sentence in data:
    for item in data:
        sentence = item['sentence']  #access the sentence within each dictionary
        tokens, tags = sentence_to_indices(sentence, word2index, tag2index)
        sentence_indices.append(tokens)
        tag_indices.append(tags)

    padded_sentence_indices = pad_sequences(sentence_indices, max_len, word2index['<PAD>'])
    padded_tag_indices = pad_sequences(tag_indices, max_len, tag2index['<PAD>'])

    sentence_tensor = torch.tensor(padded_sentence_indices, dtype=torch.long)
    tag_tensor = torch.tensor(padded_tag_indices, dtype=torch.long)

    return sentence_tensor, tag_tensor

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_test_data_lung_rads(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    grouped = df.groupby('report_index')
    sentences = []
    #for _, group in grouped:
    for report_index, group in grouped:
        sentence = list(zip(group['token'], group['iob_tag']))
        #sentences.append(sentence)
        sentences.append({"report_index": report_index, "sentence": sentence})  # Include report_index

    return sentences

if __name__ == '__main__':
    folder_path = "bilstmcrf_pytorch/lung_rads_data/"
    csv_file_path = "bilstmcrf_pytorch/lung_rads_data/lung_rads_test.csv"  # Replace with your CSV file path
    
    test_data = preprocess_test_data_lung_rads(csv_file_path)
    save_json(test_data, folder_path+"test_data.json")

    """
    folder_path = "1_bilstmcrf_pytorch/train_test_70_30/data/train_10/"
    csv_file_path = "1_bilstmcrf_pytorch/train_test_70_30/df_tokens_labeled_iob.csv"  # Replace with your CSV file path

    num = random.randrange(42000)
    print(num)
    train_data, test_data, unique_tags = preprocess_ner_data(csv_file_path, test_size=0.3, random_state=num)
    #word2index, index2word, tag2index, index2tag = create_mappings(train_data + test_data, unique_tags)
    word2index, index2word, tag2index, index2tag = create_mappings([item['sentence'] for item in train_data + test_data], unique_tags) # Pass sentences only

    save_json(train_data, folder_path+"train_data.json")
    save_json(test_data, folder_path+"test_data.json")
    save_json(unique_tags, folder_path+"unique_tags.json")
    save_json(word2index, folder_path+"word2index.json")
    save_json({str(k): v for k, v in index2word.items()}, folder_path+"index2word.json")
    save_json(tag2index, folder_path+"tag2index.json")
    save_json({str(k): v for k, v in index2tag.items()}, folder_path+"index2tag.json")
    """
