import os
import sys
import json
from collections import Counter
from shutil import rmtree

import numpy as np
from tqdm import tqdm

from utils import hash_


def clean_directory(log_dir, remove_model=False):
    for child_dir in os.listdir(log_dir):
        dir_tmp = os.path.join(log_dir, child_dir)
        if os.path.isdir(dir_tmp):
            rmtree(dir_tmp)
        if dir_tmp.endswith(".pb"):
            if remove_model:
                os.remove(dir_tmp)


def preprocess_data(data_path, preprocessing_function):
    with open(data_path) as infile:
        data = infile.read().split('\n')
        data = [preprocessing_function(text) for text in data]
    prep_data_path = "{}_prep.txt".format(data_path.split(".txt")[0])
    with open(prep_data_path, "w") as outfile:
        outfile.write("\n").join(data)
    return prep_data_path


def parse_txt(path, debug_till_row=None, join_desc=False, return_max_len=False, fraction=1,
              label_prefix="__label__", seed=None):
    """
    *** to be optimized ***
    Read fasttext format txt file and create data and labels
    :param path: str, path to txt file of fasttext format
    :param debug_till_row: int, till which row to read the file
    :param join_desc: bool, join the words to form string
    :param return_max_len: bool, return tuple (descriptions, labels, max_len)
    :param fraction: float, what fraction of data to use, if < 1, a random fraction will be chosen
    :param label_prefix: str, prefix before the label
    :param seed: int
    """
    with open(path, "r") as infile:
        data = infile.read().split("\n")[:-1]
        if debug_till_row:
            data = data[:debug_till_row]

    max_len = -1
    if (fraction <= 0) or (fraction > 1):
        raise ValueError("fraction should be in (0, 1]")
    if fraction < 1:
        if seed is not None:
            np.random.seed(seed)

        size = int(round(fraction * len(data)))
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = [data[i] for i in indices[:size]]

    descriptions, labels = [], []
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
            descriptions.append(" ".join(row_splitted[ind:]))
        else:
            descriptions.append(row_splitted[ind:])

    if return_max_len:
        return descriptions, labels, max_len
    return descriptions, labels


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
                yield "_".join(sorted(splitted_string[word_pos:word_pos + ngram]))
            else:
                yield "_".join(splitted_string[word_pos:word_pos + ngram])


def make_word_vocab(list_of_descriptions, word_n_grams=1, sort_ngrams=False, return_inverse=False, show_progress=True,
                    flush=False):
    """
    :param list_of_descriptions: list or array, list of descriptions
    :param word_n_grams: int
    :param sort_ngrams: bool, sort words of ngram before storing
    (ex: "used car" and "car used" both will be read as "car used")
    :param return_inverse: bool, return tuple (word_vocab, inverse_vocab), where keys to inverse_vocab are the word ids
    :param show_progress: bool, show progress bar
    :param flush: bool, flush after printing
    """
    cnt, id_cnt, word_vocab = 0, 1, {"__MEAN_EMBEDDING__": {"cnt": len(list_of_descriptions), "id": 0}}
    disable_progressbar = not show_progress

    if disable_progressbar:
        print("Creating train vocabulary", flush=flush)
    for cur_desc_split in tqdm(list_of_descriptions, disable=disable_progressbar, desc="Creating train vocabulary",
                               file=sys.stdout):
        cur_len = len(cur_desc_split)
        for ng in get_all(cur_desc_split, min(cur_len, word_n_grams), sort_ngrams=sort_ngrams):
            cnt += 1
            if ng in word_vocab:
                word_vocab[ng]["cnt"] += 1
            else:
                word_vocab[ng] = {"cnt": 1, "id": id_cnt}
                id_cnt += 1
    print("Read {}m words and phrases".format(round(cnt / 1e6, 1)), flush=flush)
    print("Number of unique words and phrases: {}".format(len(word_vocab)), flush=flush)

    if return_inverse:
        inverse_vocab = {v["id"]: {"cnt": v["cnt"], "phrase": k} for k, v in word_vocab.items()}
        return word_vocab, inverse_vocab
    return word_vocab


def make_label_vocab(list_of_labels):
    """
    :param list_of_labels: list or array, list of labels
    """
    cnt = Counter(list_of_labels)
    label_vocab = dict()
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


def to_fasttext_format(data, labels, save_path, label_prefix="__label__"):
    ft_format = ["{}{} {}".format(label_prefix, l, d) for d, l in zip(data, labels)]
    with open(save_path, "w+") as outfile:
        outfile.write("\n".join(ft_format))


def get_max_words_with_ngrams(max_words, word_ngrams):
    max_words_with_ng = 1
    for ng in range(word_ngrams):
        max_words_with_ng += max_words - ng
    return max_words_with_ng


def check_model_presence(log_dir, n_epochs=None):
    if n_epochs:
        return os.path.isfile(os.path.join(log_dir, "results.json")) and (os.path.join(log_dir, "model_params.json")) \
               and (os.path.join(log_dir, "model_ep{}.pb".format(n_epochs)))
    else:
        return os.path.isfile(os.path.join(log_dir, "results.json")) and (os.path.join(log_dir, "model_params.json")) \
               and (os.path.join(log_dir, "model_best.pb"))


def batch_generator(description_hashes, labels, batch_size, label_vocab, cache, shuffle=False, show_progress=True,
                    progress_desc=None):
    num_datapoints = len(description_hashes)
    indices = np.arange(num_datapoints)
    rem_indices, batch_indices = next_batch(indices, batch_size, shuffle)

    if num_datapoints <= batch_size:
        show_progress = False

    disable_progressbar = not show_progress
    if disable_progressbar:
        if progress_desc:
            print(progress_desc)
    progress_bar = tqdm(total=int(np.ceil(num_datapoints / batch_size)), disable=disable_progressbar,
                        desc=progress_desc, file=sys.stdout)

    while len(batch_indices) > 0:
        batch_hashes = [description_hashes[i] for i in batch_indices]
        batch_phrase_indices = [cache[_hash]["i"] for _hash in batch_hashes]
        batch_phrase_weights = [cache[_hash]["w"] for _hash in batch_hashes]
        batch_labels = [label_vocab[labels[index]]["id"] for index in batch_indices]

        cur_lens = np.array([len(phrase_indices) for phrase_indices in batch_phrase_indices])
        mx_len = max(cur_lens)
        to_pad = mx_len - cur_lens

        batch = [i + [0 for _ in range(pad)] for i, pad in zip(batch_phrase_indices, to_pad)]
        batch_weights = [i + [0 for _ in range(pad)] for i, pad in zip(batch_phrase_weights, to_pad)]

        rem_indices, batch_indices = next_batch(rem_indices, batch_size, shuffle)
        progress_bar.update()
        yield batch, batch_weights, batch_labels

    progress_bar.close()


def cache_data(descriptions, labels, word_vocab, label_vocab, word_ngrams, sort_ngrams, cache=None,
               is_test_data=False, show_progress=True, progress_desc=None, print_postfix="\n", flush=False):
    if cache is None:
        cache = dict()

    description_hashes, labels2 = [], []

    descriptions_thrown, labels_thrown = 0, 0
    disable_progressbar = not show_progress
    if disable_progressbar:
        if progress_desc:
            print(progress_desc, flush=flush)
    for description, label in \
            zip(tqdm(descriptions, disable=disable_progressbar, desc=progress_desc, file=sys.stdout), labels):

        phrase_indices = [0] + [word_vocab[phrase]["id"] for phrase in get_all(description, word_ngrams, sort_ngrams) if
                                phrase in word_vocab]
        if len(phrase_indices) == 1:
            descriptions_thrown += 1
            continue

        if label not in label_vocab:
            if is_test_data:
                labels_thrown += 1
                continue

        tmp_hash = hash_(str(description))
        if tmp_hash not in cache:
            desc_weights = [1. / len(phrase_indices) for _ in range(len(phrase_indices))]
            cache[tmp_hash] = {
                "i": phrase_indices,
                "w": desc_weights
            }
        labels2.append(label)
        description_hashes.append(tmp_hash)

    if labels_thrown > 0:
        print("{} datapoints thrown because of empty description".format(descriptions_thrown), flush=flush)
        print("{} datapoints thrown because of label {}".format(labels_thrown, print_postfix), flush=flush)
    else:
        print("{} datapoints thrown because of empty description {}".format(descriptions_thrown, print_postfix),
              flush=flush)
    return description_hashes, labels2, cache


def get_word_label_vocabs(train_descriptions, train_labels, word_ngrams, min_word_count, sort_ngrams,
                          cache_dir, force, show_progress=False, flush=False):
    label_dict_path = os.path.join(cache_dir, "label_dict.json")
    word_id_path = os.path.join(cache_dir, "word_id.json")

    if os.path.isfile(label_dict_path) and os.path.isfile(word_id_path) and not force:
        print("\n*** using cached dicts ***", flush=flush)
        using_cached = True
        with open(label_dict_path, "r") as infile:
            label_vocab = json.load(infile)
        with open(word_id_path, "r") as infile:
            word_vocab = json.load(infile)
        tmp_cnt = sum([i["cnt"] for i in word_vocab.values()])
        print("read {}m words and phrases".format(round(tmp_cnt / 1e6, 1)), flush=flush)
        print("number of unique words and phrases: {}\n".format(len(word_vocab)), flush=flush)
    else:
        using_cached = False
        word_vocab = make_word_vocab(train_descriptions, word_ngrams, sort_ngrams=sort_ngrams,
                                     show_progress=show_progress, flush=flush)
        label_vocab = make_label_vocab(train_labels)

    if min_word_count > 1:
        tmp_cnt = 0
        word_vocab_thresholded = dict()
        for k, v in sorted(word_vocab.items(), key=lambda t: t[0]):
            if v["cnt"] >= min_word_count:
                v["id"] = tmp_cnt
                word_vocab_thresholded[k] = v
                tmp_cnt += 1

        word_vocab = word_vocab_thresholded.copy()
        del word_vocab_thresholded

        print("Number of unique words and phrases after thresholding: {}".format(len(word_vocab)), flush=flush)

    print("\nNumber of labels in train: {}".format(len(set(label_vocab.keys()))), flush=flush)

    if not using_cached:
        with open(label_dict_path, "w+") as outfile:
            json.dump(label_vocab, outfile)
        with open(word_id_path, "w+") as outfile:
            json.dump(word_vocab, outfile)
    return word_vocab, label_vocab


def get_accuracy_log_dir(process_output, k, verbose):
    for line in iter(process_output.stdout.readline, b""):
        line = line.rstrip().decode("utf-8")
        if "stored at" in line:
            log_dir_line = line

        if "accuracy" in line:
            line_split = line.split()
            if "test" in line:
                top_1_accuracy = float(line_split[-4][:-1])
                top_k_accuracy = float(line_split[-1])
            else:
                if str(1) in line.split():
                    top_1_accuracy = float(line_split[-1])
                if str(k) in line.split():
                    top_k_accuracy = float(line_split[-1])

        if verbose:
            print(line)
    process_output.stdout.close()

    log_dir = log_dir_line.split("at ")[-1]
    return top_1_accuracy, top_k_accuracy, log_dir
