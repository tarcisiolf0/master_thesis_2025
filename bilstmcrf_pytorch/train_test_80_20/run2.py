import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import dp2 as dp
from seqeval.metrics import classification_report
import itertools
import json
import logging
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Config:
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=64, lstm_dropout=0.1, learning_rate=0.01, epochs=10, batch_size=2, padding_idx=-1):
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_dropout = lstm_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.padding_idx = padding_idx

class BiLSTM_CRF(nn.Module):
    def __init__(self, config, word2index):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=word2index['<PAD>'])
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=config.lstm_dropout)
        self.hidden2tag = nn.Linear(config.hidden_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions


#def train(model, train_sentences_tensor, train_tags_tensor, config, word2index, device):
def train(model, train_sentences_tensor, train_tags_tensor, config, word2index):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    dataset = TensorDataset(train_sentences_tensor, train_tags_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    best_loss = float('inf')  # Initialize with infinity
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        batch_num = 0

        for batch_sentences, batch_tags in dataloader:

            batch_num += 1
            #batch_sentences = batch_sentences.to(device)
            #batch_tags = batch_tags.to(device)

            model.zero_grad()
            emissions = model(batch_sentences)
            mask = batch_sentences != word2index['<PAD>']
            loss = -model.crf(emissions, batch_tags, mask=mask)

            # added
            loss.backward()  # Ensure we don't retain unnecessary graphs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()
            # end added

            total_loss += loss.item()  # Correctly accumulate loss

            avg_batch_loss = loss.item()
            logging.info(f"Epoch: {epoch+1}, Batch: {batch_num}, Loss: {avg_batch_loss:.4f}")

        avg_epoch_loss = total_loss / len(dataloader)
        logging.info(f"Epoch: {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

                
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0  # Reset counter
            torch.save(model.state_dict(), "best_bilstm_crf.pth") # Save best model
            logging.info(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}. Loss has not improved for {patience} epochs.")
            break  # Exit training loop

    model.load_state_dict(torch.load("best_bilstm_crf.pth")) # Load best model
    return model

#def predict(model, sentence, word2index, device):
def predict(model, sentence, word2index):
    model.eval()
    with torch.no_grad():
        #sentence = sentence.to(device)
        emissions = model(sentence)
        mask = sentence != word2index['<PAD>']
        predicted_tags = model.crf.decode(emissions, mask=mask)
    return predicted_tags[0]

#def evaluate_model(model, test_sentences_tensor, test_tags_tensor, test_data, word2index, tag2index, index2tag, config, device):
def evaluate_model(model, test_sentences_tensor, test_tags_tensor, test_data, word2index, tag2index, index2tag):
    test_actual_tags = []
    test_predicted_tags = []

    for i in range(len(test_sentences_tensor)):
        _, tags = dp.sentence_to_indices(test_data[i], word2index, tag2index) # Correctly retrieve original tags
        #predicted_tags = predict(model, test_sentences_tensor[i].unsqueeze(0), word2index, device)
        predicted_tags = predict(model, test_sentences_tensor[i].unsqueeze(0), word2index)

        actual_tags = test_tags_tensor[i].tolist()
        actual_tags = [index2tag[idx] for idx in actual_tags]

        predicted_tags = [index2tag[idx] for idx in predicted_tags]

        actual_tags = actual_tags[:len(tags)]  # Truncate to original length
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
    
if __name__ == "__main__":
    train_data = dp.load_json("train_data.json")
    test_data = dp.load_json("test_data.json")
    unique_tags = dp.load_json("unique_tags.json")
    word2index = dp.load_json("word2index.json")
    index2word = {int(k): v for k, v in dp.load_json("index2word.json").items()}
    tag2index = dp.load_json("tag2index.json")
    index2tag = {int(k): v for k, v in dp.load_json("index2tag.json").items()}


    # Convert sentences to indices and pad
    max_len = 512
    train_sentences_indices, train_tags_indices = dp.process_data(train_data, max_len, word2index, tag2index)
    test_sentences_indices, test_tags_indices = dp.process_data(test_data, max_len, word2index, tag2index)

    vocab_size = len(word2index)
    num_tags = len(unique_tags)
    best_f1_score = -1

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #logging.info(f"Using device: {device}")

    param_grid = {
        "batch_size": [4, 8, 16],
        "embedding_dim": [50, 100, 200],
        "hidden_dim": [64, 128, 256],
        "lstm_dropout": [0.1],
        "learning_rate": [0.01],
    }

    best_f1_score = -1
    for batch_size, embedding_dim, hidden_dim, lstm_dropout, learning_rate in itertools.product(
        param_grid["batch_size"], param_grid["embedding_dim"], param_grid["hidden_dim"], param_grid["lstm_dropout"], param_grid["learning_rate"]):

        logging.info(f"Training with batch_size={batch_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}")
        print("\n")
        config = Config(vocab_size, num_tags, embedding_dim, hidden_dim, lstm_dropout, learning_rate, epochs=10, batch_size=batch_size, padding_idx=word2index['<PAD>'])

        #model = BiLSTM_CRF(config, word2index).to(device)
        model = BiLSTM_CRF(config, word2index)

        #trained_model = train(model, train_sentences_indices, train_tags_indices, config, word2index, device)
        trained_model = train(model, train_sentences_indices, train_tags_indices, config, word2index)
        #test_actual_tags, test_predicted_tags = evaluate_model(trained_model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag, config, device)
        test_actual_tags, test_predicted_tags = evaluate_model(trained_model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag)

        report = classification_report(test_actual_tags, test_predicted_tags, output_dict=True)
        f1_score = report["macro avg"]["f1-score"]

        logging.info(f"F1-score: {f1_score:.4f}")

        # Save report as a JSON file
        with open(f"classification_report_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}.json", "w") as f:
            json.dump(report, f, indent=4, default=convert)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_params = {"batch_size": batch_size, "embedding_dim": embedding_dim, "hidden_dim": hidden_dim}
            #torch.save(trained_model.state_dict(), "best_bilstm_crf.pth")
        torch.save(trained_model.state_dict(), f"bilstm_crf_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}.pth")

    logging.info(f"Best Model - batch_size={best_params['batch_size']}, embedding_dim={best_params['embedding_dim']}, hidden_dim={best_params['hidden_dim']}, F1-score={best_f1_score:.4f}")