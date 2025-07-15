import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class NerDataset(Dataset):
    """
    Custom dataset implementation to get (text,labels) tuples
    Inputs:
    - df : dataframe with columns [tags, sentence]
    """

    def __init__(self, df, tag2idx, tokenizer_name="pucpr/biobertpt-all", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tag2idx = tag2idx

        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input should be a dataframe')

        if "tags" not in df.columns or "sentence" not in df.columns:
            raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")

        tags_list = [i.split() for i in df["tags"].values.tolist()]
        texts = df["sentence"].values.tolist()

        self.texts = [self.tokenizer(text, padding = "max_length", max_length = max_length, truncation = True, return_tensors = "pt") for text in texts]
        self.labels = [self.match_tokens_labels(text, tags) for text,tags in zip(self.texts, tags_list)]

        #text_tokenized = self.tokenizer(texts , padding = "max_length", max_length = 512, truncation = True, return_tensors = "pt" )
        #self.word_ids = text_tokenized.word_ids()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_text = self.texts[idx]
        batch_labels = self.labels[idx]
        batch_labels_tensor = torch.LongTensor(batch_labels)

        return batch_text, batch_labels_tensor


    def match_tokens_labels(self, tokenized_input, tags, ignore_token = -100):
        '''
        Used in the custom dataset.
        -100 will be tha label used to match additional tokens like [CLS] [PAD] that we dont care about.
        Inputs :
        - tokenized_input : tokenizer over the imput text -> {input_ids, attention_mask}
        - tags : is a single label array -> [O O O O O O O O O O O O O O B-tim O]

        Returns a list of labels that match the tokenized text -> [-100, 3,5,6,-100,...]
        '''

        #gives an array [ None , 0 , 1 ,2 ,... None]. Each index tells the word of reference of the token
        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(ignore_token)

            #if its equal to the previous word we can add the same label id of the provious or -100
            else :
                try:
                    reference_tag = tags[word_idx]
                    label_ids.append(self.tag2idx[reference_tag])
                except:
                    label_ids.append(ignore_token)

            previous_word_idx = word_idx

        return label_ids