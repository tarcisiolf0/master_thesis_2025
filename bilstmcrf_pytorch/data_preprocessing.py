import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import json

def preprocess_ner_data(csv_file, test_size=0.3, random_state=42):
    """
    Preprocesses data for Named Entity Recognition (NER) from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        test_size (float, optional): Proportion of the data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for train/test split. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_data (list): List of training sentences (list of tuples (token, tag)).
            - test_data (list): List of testing sentences (list of tuples (token, tag)).
            - unique_tags (list): List of unique IOB tags (including padding and unknown).
    """

    df = pd.read_csv(csv_file, encoding='utf-8')  # Important: Specify encoding

    # Group by report index to create sentences
    grouped = df.groupby('report_index')
    sentences = []
    for _, group in grouped:
        sentence = list(zip(group['token'], group['iob_tag']))  # Combine tokens and tags
        sentences.append(sentence)

    # Create unique tags list (including padding)
    unique_tags = sorted(list(set([tag for sentence in sentences for _, tag in sentence])))
    # Add padding tag if not already present
    if '<PAD>' and '<UNK' not in unique_tags:
        unique_tags.append('<PAD>') # Padding tag
        unique_tags.append('<UNK>') # Unknown tag

    # Split into training and testing sets
    train_sentences, test_sentences = train_test_split(
        sentences, test_size=test_size, random_state=random_state
    )

    return train_sentences, test_sentences, unique_tags



def create_mappings(sentences, unique_tags):
    """Creates mappings between tags and indices."""

    # Create word2index and index2word
    all_words = [str(token) for sentence in sentences for token, _ in sentence]
    unique_words = sorted(list(set(all_words)))
 
 # Create word2index and index2word dictionaries
    word2index = {}
    index2word = {}

    word2index = {word: index for index, word in enumerate(unique_words)}
    index2word = {index: word for word, index in word2index.items()}
    word2index['<UNK>'] = len(word2index) # Add unknown token
    index2word[len(index2word)] = '<UNK>'
    word2index['<PAD>'] = len(word2index) # Add unknown token
    index2word[len(index2word)] = '<PAD>'

    tag2index = {tag: index for index, tag in enumerate(unique_tags)}
    index2tag = {index: tag for tag, index in tag2index.items()}

    #print(word2index)
    return word2index, index2word, tag2index, index2tag  # Return all mappings


def sentence_to_indices(sentence, word2index, tag2index):
    """Converts a sentence (list of tuples) to lists of token indices and tag indices."""
  
    #tokens = [word2index[token] for token, _ in sentence]
    #tags = [tag2index[tag] for _, tag in sentence]

    tokens = [word2index[token] if token in word2index and pd.notna(token) else word2index.get('<UNK>') for token, _ in sentence]
    tags = [tag2index[tag] if tag in tag2index and pd.notna(tag) else tag2index.get('<UNK>') for _, tag in sentence]
    return tokens, tags


def pad_sequences(sequences, max_len, padding_value):
    """Pads sequences to a fixed length."""
    padded_sequences = []
    for seq in sequences:
        padding_length = max_len - len(seq)
        padded_seq = seq + [padding_value] * padding_length
        padded_sequences.append(padded_seq)
    return padded_sequences


def process_data(data, max_len, word2index, tag2index):
    """
    Converts sentences to indices, pads them, and returns PyTorch tensors.

    Args:
        data: A list of sentences (presumably lists of words and tags).
        max_len: The maximum sequence length for padding.
        word2index: A dictionary mapping words to indices.
        tag2index: A dictionary mapping tags to indices.

    Returns:
        A tuple containing two lists:
            - sentence_indices: A list of PyTorch tensors representing padded sentence indices.
            - tag_indices: A list of PyTorch tensors representing padded tag indices.
    """

    sentence_indices = []
    tag_indices = []

    for sentence in data:
        tokens, tags = sentence_to_indices(sentence, word2index, tag2index)
        padded_tokens = pad_sequences([tokens], max_len, word2index['<PAD>'])
        padded_tags = pad_sequences([tags], max_len, tag2index['<PAD>'])
        sentence_indices.append(torch.tensor(padded_tokens))
        tag_indices.append(torch.tensor(padded_tags))

    return sentence_indices, tag_indices

# Function to save a variable as JSON
def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == '__main__':
    csv_file_path = "df_tokens_labeled_iob.csv"  # Replace with your CSV file path
    train_data, test_data, unique_tags = preprocess_ner_data(csv_file_path, test_size=0.2)
    word2index, index2word, tag2index, index2tag = create_mappings(train_data + test_data, unique_tags)

    # Save data
    save_json(train_data, "train_data.json")
    save_json(test_data, "test_data.json")
    save_json(unique_tags, "unique_tags.json")
    save_json(word2index, "word2index.json")

    # Convert integer keys to string for JSON compatibility
    save_json({str(k): v for k, v in index2word.items()}, "index2word.json")
    save_json(tag2index, "tag2index.json")
    save_json({str(k): v for k, v in index2tag.items()}, "index2tag.json")



    


