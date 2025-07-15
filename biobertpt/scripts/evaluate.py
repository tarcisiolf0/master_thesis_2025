import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader
from src.dataset import NerDataset
from src.model import BioBERTptNER
from src.utils import load_data, tags_mapping_v2, retrieve_token_tag_and_tag_pred, create_dataframe_with_predictions
from src.metrics import MetricsTracking
from transformers import AutoTokenizer
import pandas as pd


def evaluate_test_texts(model, df_test, tag2idx, idx2tag, batch_size = 1):
    tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-all")

    dev_dataset = NerDataset(df_test, tag2idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dev_metrics = MetricsTracking()
    total_loss_dev = 0

    i = 0

    tokens = []
    text_labels_dev = []
    text_labels_pred = []
    with torch.no_grad():
        for dev_data, dev_label in dev_dataloader:
            dev_label = dev_label.to(device)

            mask = dev_data['attention_mask'].squeeze(1).to(device)
            input_id = dev_data['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask, dev_label)
            loss, logits = output.loss, output.logits
        
            predictions = logits.argmax(dim= -1)

            tag = df_test.tags.iloc[i]
            text = df_test.sentence.iloc[i]

            text_tokenized = tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
            labels_dev, labels_pred = retrieve_token_tag_and_tag_pred(text_tokenized, predictions.tolist(), dev_label.tolist(), idx2tag)
            
            text_labels_dev.append(labels_dev)
            text_labels_pred.append(labels_pred)

            dev_metrics.update(predictions, dev_label)
            total_loss_dev += loss.item()
            i += 1
            

    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))

    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" )

    return text_labels_dev, text_labels_pred



if __name__ == "__main__":
    input_csv_file_name = "bilstmcrf_pytorch/lung_rads_data/lung_rads_test.csv"
    output_csv_file_name = "biobertpt/lung_rads/lung_rads_test_predicted.csv"
    # Load Test Data
    dev_df = load_data("biobertpt/lung_rads/df_test_tokens_labeled_iob_bert_format.csv")
    df_total = load_data("biobertpt/data/df_tokens_labeled_iob_bert_format.csv")

    tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping_v2(df_total["tags"])

    # Load Model
    model = BioBERTptNER(13) # unique tags length == 13 labels
    model.load_state_dict(torch.load("biobertpt/models/split_1/model_lr=3e-05_batch_size=4.pth"))
    model.eval()

    labels_dev, labels_pred = evaluate_test_texts(model, dev_df, tag2idx, idx2tag)

    create_dataframe_with_predictions(input_csv_file_name, output_csv_file_name, dev_df, labels_dev, labels_pred)