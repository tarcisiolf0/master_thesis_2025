import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
import data_preprocessing as dp
from seqeval.metrics import classification_report
import itertools
import json
import numpy as np

# Configuration (can be passed as arguments or read from a file)
class Config:
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=64, learning_rate=0.01, epochs=10, batch_size=2, padding_idx=-1): # Padding for tags
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.padding_idx = padding_idx


# Model definition
class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=word2index['<PAD>'])
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(config.hidden_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions


# Data processing functions
def prepare_data(sentences, tags, config, word2index):
    # Pad sequences
    padded_sentences = pad_sequence([torch.tensor(s) for s in sentences], batch_first=True, padding_value=word2index['<PAD>'])
    padded_tags = pad_sequence([torch.tensor(t) for t in tags], batch_first=True, padding_value=config.padding_idx) # Use config padding value
    return padded_sentences, padded_tags


# Training function
def train(model, train_sentences, train_tags, config, word2index):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        for i in range(0, len(train_sentences), config.batch_size):
            batch_sentences = train_sentences[i:i + config.batch_size]
            batch_tags = train_tags[i:i + config.batch_size]


            #print(batch_sentences)
            #print()
            #print(batch_tags)

            batch_sentences = torch.cat(batch_sentences, dim=0)
            batch_tags = torch.cat(batch_tags, dim=0)

            #print(type(batch_sentences))
            #print(batch_sentences.shape)
            #print(batch_sentences)

            #print(type(batch_tags))
            #print(batch_tags.shape)
            #print(batch_tags)
    
            model.zero_grad()
            emissions = model(batch_sentences)
            #mask = batch_sentences != 0  # Assuming 0 is the padding value for tokens
            mask = batch_sentences != word2index['<PAD>']  # Assuming 0 is the padding value for tokens
            loss = -model.crf(emissions, batch_tags, mask=mask)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}, Batch: {i//config.batch_size+1}, Loss: {loss.item()}")

# Prediction function
def predict(model, sentence, config):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        emissions = model(sentence)
        mask = sentence != 0
        predicted_tags = model.crf.decode(emissions, mask=mask)
    return predicted_tags[0]  # Return the predicted sequence

def evaluate_model(model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag, config):
    """
    Evaluates the model by converting predicted and actual tag indices back to tag names.

    Args:
        model: The trained model for prediction.
        test_sentences_indices (list): List of test sentence indices.
        test_tags_indices (list): List of test tag indices.
        test_data (list): Original test data (token, tag).
        word2index (dict): Dictionary mapping words to indices.
        tag2index (dict): Dictionary mapping tags to indices.
        index2tag (dict): Dictionary mapping indices to tags.
        config: Configuration object containing model parameters.

    Returns:
        tuple: A tuple containing:
            - test_actual_tags (list): List of actual tags.
            - test_predicted_tags (list): List of predicted tags.
    """
    test_actual_tags = []
    test_predicted_tags = []

    for i in range(len(test_sentences_indices)):
        # Convert indices back to tokens and tags
        _, tags = dp.sentence_to_indices(test_data[i], word2index, tag2index)
        predicted_tags = predict(model, test_sentences_indices[i], config)

        # Convert actual indices to tags
        actual_tags = test_tags_indices[i].tolist()[0]
        actual_tags = [index2tag[idx] for idx in actual_tags]

        # Convert predicted indices to tags
        predicted_tags = [index2tag[idx] for idx in predicted_tags]

        # Ensure both lists are truncated to the correct sentence length
        actual_tags = actual_tags[:len(tags)]
        predicted_tags = predicted_tags[:len(tags)]

        test_actual_tags.append(actual_tags)
        test_predicted_tags.append(predicted_tags)

    return test_actual_tags, test_predicted_tags

def convert(o):
    if isinstance(o, np.integer):  # Convert numpy.int64, int32, etc. to Python int
        return int(o)
    elif isinstance(o, np.floating):  # Convert numpy.float to Python float
        return float(o)
    elif isinstance(o, np.ndarray):  # Convert NumPy arrays to lists
        return o.tolist()
    else:
        return o  # Return as is if it's already a native Python type
# Example usage
if __name__ == "__main__":
    # Find max length for padding
    #max_len = max(len(sentence) for sentence in train_data + test_data)
    max_len = 512

    train_data = dp.load_json("train_data.json")
    test_data = dp.load_json("test_data.json")
    unique_tags = dp.load_json("unique_tags.json")
    word2index = dp.load_json("word2index.json")

    # Convert keys back to integers for index2word and index2tag
    index2word = {int(k): v for k, v in dp.load_json("index2word.json").items()}
    tag2index = dp.load_json("tag2index.json")
    index2tag = {int(k): v for k, v in dp.load_json("index2tag.json").items()}

    # Convert sentences to indices and pad
    train_sentences_indices, train_tags_indices = dp.process_data(train_data, max_len, word2index, tag2index)
    test_sentences_indices, test_tags_indices = dp.process_data(test_data, max_len, word2index, tag2index)
    
    """
    # Define configuration (Adjust vocab_size and num_tags)
    #vocab_size = len(set([token for sentence in train_data + test_data for token, _ in sentence])) # Update vocab_size
    vocab_size = len(word2index)
    num_tags = len(unique_tags)
    config = Config(vocab_size, num_tags, epochs=10, batch_size=64, padding_idx=word2index['<PAD>'])  # Update vocab_size and num_tags

    # Define the hyperparameter grid
    param_grid = {
    "batch_size": [4, 8, 16, 32, 64],
    "embedding_dim": [32, 64, 128, 256],
    "hidden_dim": [32, 64, 128, 256]  # LSTM units
}
    # Initialize model
    model = BiLSTM_CRF(config)

    # Train the model (using the processed data)
    train(model, train_sentences_indices, train_tags_indices, config, word2index)
    
    # Evaluate the model (using the processed data)
    test_actual_tags, test_predicted_tags = evaluate_model(model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag, config)
    report = classification_report(test_actual_tags, test_predicted_tags)
    print(report) 

    #Saving the model
    torch.save(model.state_dict(), "bilstm_crf_model.pth") #This saves only the model's learned parameters, making it more flexible for future loading.

    #Loading the model
    # Reinitialize the model with the same configuration
    #model = BiLSTM_CRF(config)
    #model.load_state_dict(torch.load("bilstm_crf_model.pth"))
    #model.eval()  # Set to evaluation mode
    

    """
    vocab_size = len(word2index)
    num_tags = len(unique_tags)
    best_f1_score = -1

    param_grid = {
    "batch_size": [4, 8, 16, 32, 64],
    "embedding_dim": [100, 200, 300],
    "hidden_dim": [32, 64, 128, 256]  # LSTM units
}
    for batch_size, embedding_dim, hidden_dim in itertools.product(param_grid["batch_size"], param_grid["embedding_dim"], param_grid["hidden_dim"]):
        print(f"Training with batch_size={batch_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")

        # Define model configuration
        config = Config(vocab_size, num_tags, embedding_dim=embedding_dim, hidden_dim=hidden_dim, epochs=10, batch_size=batch_size, padding_idx=word2index['<PAD>'])

        # Initialize model
        model = BiLSTM_CRF(config)

        # Train the model
        train(model, train_sentences_indices, train_tags_indices, config, word2index)

        # Evaluate the model
        test_actual_tags, test_predicted_tags = evaluate_model(model, test_sentences_indices, 
                                                                test_tags_indices, test_data, 
                                                                word2index, tag2index, index2tag, config)

        # Calculate F1-score
        report = classification_report(test_actual_tags, test_predicted_tags, output_dict=True)
        f1_score = report["macro avg"]["f1-score"]

        print(f"F1-score: {f1_score:.4f}\n")

        # Save report as a JSON file
        with open(f"classification_report_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}.json", "w") as f:
            json.dump(report, f, indent=4, default=convert)

        # Save the best f1_score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_params = {"batch_size": batch_size, "embedding_dim": embedding_dim, "hidden_dim": hidden_dim}
        #    best_model = model
        #   torch.save(model.state_dict(), "best_bilstm_crf.pth")

        torch.save(model.state_dict(), f"bilstm_crf_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}.pth")
        #print(f"Best Model - batch_size={best_params['batch_size']}, embedding_dim={best_params['embedding_dim']}, hidden_dim={best_params['hidden_dim']}, F1-score={best_f1_score:.4f}")
        print("\n\n")

    print(f"Best Model - batch_size={best_params['batch_size']}, embedding_dim={best_params['embedding_dim']}, hidden_dim={best_params['hidden_dim']}, F1-score={best_f1_score:.4f}")
        # Load the best model for further use
        #best_model.load_state_dict(torch.load("best_bilstm_crf.pth"))
        #best_model.eval() 
    