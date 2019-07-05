import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
from os.path import join, isfile
from utils import validate
import warnings

warnings.filterwarnings("ignore")


def parse_txt(txt_path, debug_till_row=None, join_desc=False, return_max_len=False, fraction=1,
              label_prefix="__label__", seed=None):
    """
    *** to be optimized ***
    Read fasttext format txt file and create data and labels
    :param txt_path: str, path to txt file of fasttext format
    :param debug_till_row: int, till which row to read the file
    :param join_desc: bool, join the words to form string
    :param return_max_len: bool, return tuple (descriptions, labels, max_len)
    :param fraction: float, what fraction of data to use, if < 1, a random fraction will be chosen
    :param label_prefix: str, prefix before the label
    :param seed: int
    """

    with open(txt_path, "r") as infile:
        if debug_till_row not in [None, -1]:
            data = infile.read().split("\n")[:debug_till_row]
        else:
            data = infile.read().split("\n")

    max_len = -1
    assert 0 < fraction <= 1
    if fraction < 1:
        if seed is not None:
            np.random.seed(seed)
        size = int(round(fraction * len(data)))
        inds = np.arange(len(data))
        np.random.shuffle(inds)
        data = [data[i] for i in inds[:size]]

    descs, labels = [], []
    for row in data:
        row_splitted = row.split()
        num_words = len(row_splitted)
        if num_words == 1:
            continue
        max_len = max(max_len, len(row_splitted))

        tmp = []
        for ind, w in enumerate(row_splitted):
            if not w.startswith(label_prefix):
                break
            tmp.append(w[len(label_prefix):])

        labels.append(" ".join(tmp))
        if join_desc:
            descs.append(" ".join(row_splitted[ind:]))
        else:
            descs.append(row_splitted[ind:])

    if return_max_len:
        return descs, labels, max_len
    return descs, labels


def get_all(splitted_string, word_ngram, sort_ngrams=False):
    """Get all word ngrams from the splitted string
    :param splitted_string: list or array, splitted text
    :param word_ngram: int
    :param sort_ngrams: bool, sort words of ngram before storing
    (ex: "used car" and "car used" both will be read as "car used")
    """
    for ngram in range(1, word_ngram + 1):
        for word_pos in range(len(splitted_string) - ngram + 1):
            if sort_ngrams:
                yield ("_".join(sorted(splitted_string[word_pos:word_pos + ngram])))
            else:
                yield ("_".join(splitted_string[word_pos:word_pos + ngram]))


def make_train_vocab(list_of_descriptions, word_n_grams=1, sort_ngrams=False, return_inverse=False):
    """
    :param list_of_descriptions: list or array, list of descriptions
    :param word_n_grams: int
    :param sort_ngrams: bool, sort words of ngram before storing
    (ex: "used car" and "car used" both will be read as "car used")
    :param return_inverse: bool, return tuple (train_vocab, inverse_vocab), where keys to inverse_vocab are the word ids
    """
    print("\n\nCreating train vocabulary ...")
    cnt, id_cnt, train_vocab = 0, 1, {}
    for cur_desc_split in tqdm(list_of_descriptions):
        cur_len = len(cur_desc_split)
        for ng in get_all(cur_desc_split, min(cur_len, word_n_grams), sort_ngrams=sort_ngrams):
            cnt += 1
            if ng in train_vocab:
                train_vocab[ng]["cnt"] += 1
            else:
                train_vocab[ng] = {"cnt": 1, "id": id_cnt}
                id_cnt += 1
    print("Read {}m words and phrases".format(round(cnt / 1e6, 1)))
    print("Number of unique words and phrases: {}".format(len(train_vocab)))

    if return_inverse:
        inverse_vocab = {v["id"]: {"cnt": v["cnt"], "phrase": k} for k, v in train_vocab.items()}
        return train_vocab, inverse_vocab
    return train_vocab


def make_label_vocab(list_of_labels):
    """
    :param list_of_labels: list or array, list of labels
    """
    cnt = Counter(list_of_labels)
    label_vocab = {}
    for i, label in enumerate(sorted(cnt.keys())):
        label_vocab[label] = {"id": i, "cnt": cnt[label]}
    return label_vocab


def construct_label(index, num_classes):
    """
    :param index: int, index of the class
    :param num_classes: int, number of classes
    """
    label = np.zeros(num_classes)
    label[index] = 1
    return label


def next_batch(data, batch_size, shuffle=False):
    """
    :param data: list or array
    :param batch_size: int, the size of the batch
    :param shuffle: bool, shuffle data before selecting the batch
    """

    if len(data) <= batch_size:
        return [], data
    else:
        if shuffle:
            np.random.shuffle(data)
        return data[batch_size:], data[:batch_size]


def preprocess_and_save(df, val_mask, text_field, label_field, preprocessing_function=None,
                        additional_fields_and_preps={}, save_dir="./", postfix="", verbose=False, print_items=[]):
    val = df[val_mask]
    train = df[~val_mask]
    if preprocessing_function is not None:
        print_items.append("preprocessing descriptions")
        train[text_field] = train[text_field].map(lambda s: preprocessing_function(str(s)))
        train = train[train[text_field] != ""]
        val[text_field] = val[text_field].map(lambda s: preprocessing_function(str(s)))

    if verbose:
        [print(i) for i in print_items]
    train["label"] = train[label_field].apply(lambda x: "__label__{}".format(x))
    val["label"] = val[label_field].apply(lambda x: "__label__{}".format(x))
    use_cols = ["label", text_field] + list(additional_fields_and_preps.keys())
    train_data = [" ".join(i) for i in train[use_cols].values]
    val_data = [" ".join(i) for i in val[use_cols].values]
    save_dir = validate(save_dir)
    save_path_train = join(save_dir, "train{}.txt".format(postfix))
    save_path_val = join(save_dir, "val{}.txt".format(postfix))
    with open(save_path_train, "w+") as outfile:
        outfile.write("\n".join(train_data))
    with open(save_path_val, "w+") as outfile:
        outfile.write("\n".join(val_data))
    return save_path_train, save_path_val


def train_val_split_from_df(path_to_df, text_field, label_field, split_params={}, save_dir="./",
                            preprocessing_function=None, additional_fields_and_preps={}, postfix="", verbose=False):
    """
    Create train and validation files from csv or parquet file with preprocessing
    :param path_to_df: str, path to csv or parquet file
    :param text_field: str, column of the dataframe in which is the text that should be classified
    :param label_field: str, column of the dataframe in which is the label of the corresponding text
    :param split_params: dict, input format: {"seed": int, default 17, "fraction": float, default: 0.1}
    :param save_dir: str, directory to save the txt files
    :param preprocessing_function: function, function to apply on text_field column
    :param additional_fields_and_preps: dict. Dictionary in the following format
    {field_name1: preprocessing_function1, field_name2: preprocessing_function2} to enable custom preprocessing for
    different fields
    :param postfix: str, postfix to add to train and validation files
    :param verbose: bool
    :return: tuple, the train and validation data paths
    """
    if path_to_df.endswith("parquet"):
        df = pd.read_parquet(path_to_df)
    else:
        df = pd.read_csv(path_to_df)

    print_items = []

    if "seed" in split_params:
        seed = split_params["seed"]
    else:
        print_items.append("no 'seed' parameter specified in split_params, the default is 17")
        seed = 17
    np.random.seed(seed)
    if "fraction" in split_params:
        f = split_params["fraction"]
        assert type(f) == float
    else:
        print_items.append("no 'fraction' parameter specified in split_params, the default is 0.1")
        f = 0.1

    for added_field, prep_f in additional_fields_and_preps.items():
        if df[added_field].dtype != "object":
            df[added_field] = df[added_field].astype(str)
        if prep_f:
            df[added_field] = df[added_field].map(prep_f)
        df[text_field] = df[text_field] + " " + df[added_field]

    val_mask = np.random.choice([True, False], size=len(df),  p=[f, 1 - f])

    return preprocess_and_save(df, val_mask, text_field, label_field, preprocessing_function,
                               additional_fields_and_preps, save_dir, postfix, verbose, print_items)


def check_model_presence(log_dir, n_epochs):
    return isfile(join(log_dir, "results.json")) and (join(log_dir, "model_params.json")) and \
           (join(log_dir, "model_ep{}.pb".format(n_epochs)))
