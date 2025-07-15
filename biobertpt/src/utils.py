import os
import pandas as pd


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8")
    df.rename(columns={"text": "sentence", "iob_labels": "tags"}, inplace=True)
    return df

def tags_mapping(df_train):
    """
    tag_series = df column with tags for each sentence.
    Returns:
    - dictionary mapping tags to indexes (label)
    - dictionary mappign inedexes to tags
    - The label corresponding to tag 'O'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
    """
  

    #if not isinstance(tags_series, pd.Series):
    #    raise TypeError('Input should be a padas Series')

    print(df_train.columns)
    unique_tags = set()

    for tag_list in df_train["tags"]:
        for tag in tag_list.split():
            unique_tags.add(tag)


    tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
    idx2tag = {k:v for v,k in tag2idx.items()}

    unseen_label = tag2idx["O"]

    return tag2idx, idx2tag, unseen_label, unique_tags

def tags_2_labels(unseen_label, tags : str, tag2idx : dict):
    '''
    Method that takes a list of tags and a dictionary mapping and returns a list of labels (associated).
    Used to create the "label" column in df from the "tags" column.
    '''
    return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in tags.split()]

def tags_mapping_v2(tags_series):  # More descriptive argument name
    """
    Maps tags to indices and vice-versa.

    Args:
        tags_series: A Pandas Series containing lists of tags.

    Returns:
        tag2idx: Dictionary mapping tags to indices.
        idx2tag: Dictionary mapping indices to tags.
        unseen_label: Index of the 'O' tag.
        unique_tags: Set of unique tags.
    """

    if not isinstance(tags_series, pd.Series):
        raise TypeError('Input should be a pandas Series')

    unique_tags = set()

    for tag_list in tags_series:  # Iterate directly over the Series
        if isinstance(tag_list, str): # Handle cases where tag_list might be a string, not a list
            for tag in tag_list.split():
                unique_tags.add(tag)
        elif isinstance(tag_list, list):
            for tag in tag_list:
                unique_tags.add(tag)
        else:
            print(f"Unexpected tag_list type: {type(tag_list)}")
            continue # or raise an exception if you want to be strict


    tag2idx = {k: v for v, k in enumerate(sorted(unique_tags))}
    idx2tag = {k: v for v, k in tag2idx.items()}

    unseen_label = tag2idx.get("O") # safer way to access the "O" tag
    if unseen_label is None:
      raise ValueError("The 'O' tag was not found in the data")

    return tag2idx, idx2tag, unseen_label, unique_tags


def retrieve_token_tag_and_tag_pred(text_tokenized, predictions, dev_label, idx2tag):
    word_ids = text_tokenized.word_ids()
    previous_index = None

    retrieved_tags_pred = []
    retrieved_tags_dev = []
    i = 0
    predictions = predictions[0]
    dev_label = dev_label[0]

    for word_idx in word_ids:
        if word_idx == None:
            pass
        elif word_idx == previous_index:
            pass
        else:
            retrieved_tags_pred.append(idx2tag[predictions[i]])
            if dev_label[i] == -100 or dev_label[i] == "-100":
                retrieved_tags_dev.append("O")
            else:
                retrieved_tags_dev.append(idx2tag[dev_label[i]])

        i += 1
        previous_index = word_idx

    return retrieved_tags_dev, retrieved_tags_pred


def create_dataframe_with_predictions(input_csv_file_name, output_file_name, dev_df, labels_dev, labels_pred):

    df = pd.DataFrame(columns=['token', 'iob_tag', 'predicted_iob_tag'])
    j=0
    for list_labels_dev, list_labels_pred in zip(labels_dev, labels_pred):
        list_tokens = dev_df.sentence.iloc[j].split(' ')
        for i in range(len(list_labels_dev)):
            data = []
            token = list_tokens[i]
            tag = list_labels_dev[i]
            predicted_tag = list_labels_pred[i]
            data.append({'token': token, 'iob_tag': tag, 'predicted_iob_tag': predicted_tag})
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        j += 1

    test_df = pd.read_csv(input_csv_file_name)
    report_index_df = test_df.report_index
    df.insert(0, 'report_idx', report_index_df)
    df.to_csv(output_file_name, index=False)