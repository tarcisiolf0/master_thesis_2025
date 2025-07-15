import torch
import torch.nn as nn
from transformers import BertForTokenClassification

class BioBERTptNER(nn.Module):
    """
    Implement NN class based on distilbert pretrained from Hugging face.
    Inputs :
    tokens_dim : int specifyng the dimension of the classifier
    """

    def __init__(self, tokens_dim):
        super(BioBERTptNER,self).__init__()

        if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

        if tokens_dim <= 0:
            raise ValueError('Classification layer dimension should be at least 1')

        self.pretrained = BertForTokenClassification.from_pretrained("pucpr/biobertpt-all", num_labels = tokens_dim) #set the output of each token classifier = unique_lables


    def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
        """
        Forwad computation of the network
        Input:
        - inputs_ids : from model tokenizer
        - attention :  mask from model tokenizer
        - labels : if given the model is able to return the loss value
        """

        #inference time no labels
        if labels is None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out

        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out