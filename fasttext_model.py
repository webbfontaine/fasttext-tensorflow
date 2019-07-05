import tensorflow as tf
import numpy as np
import pandas as pd
from fasttext_utils import next_batch, get_all, train_val_split_from_df, parse_txt, preprocess_and_save
from utils import load_graph, hash_, validate, hash_function, handle_space_paths, copy_all, split_list
import os
import gc
import json
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT
import warnings
import inspect

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FastTextModel:
    def __init__(self, model_path, model_params_path, label_prefix="__label__", preprocessing_function=None,
                 use_gpu=False, gpu_fraction=0.5, hyperparams={}):
        """
        Load already trained fasttext model
        :param model_path: str, path to model file (with .pb extension)
        :param model_params_path: str, path to model_params.json
        :param label_prefix: str
        :param preprocessing_function: function, function to apply on data before prediction
        :param use_gpu: bool, use gpu
        :param gpu_fraction: float, how much of gpu to use (ignored if use_gpu is False)
        :param hyperparams: dict, do not pass this if you are just loading model, it is used internally
        """
        tf.reset_default_graph()
        self.label_prefix = label_prefix
        self.hyperparams = hyperparams
        self.info = {"model_path": os.path.abspath(model_path), "model_params_path": os.path.abspath(model_params_path)}
        with open(model_params_path, "r") as infile:
            model_params = json.load(infile)
        for k, v in model_params.items():
            self.info[k] = v
        if os.path.isfile(model_params["label_dict_path"]):
            with open(model_params["label_dict_path"], "r") as infile:
                self.label_vocab = json.load(infile)
        else:
            new_path = os.path.join(os.path.dirname(model_params_path), "label_dict.json")
            print("{} not found, switching to model_params' path {}".format(model_params["label_dict_path"], new_path))
            with open(new_path, "r") as infile:
                self.label_vocab = json.load(infile)
            self.info["label_dict_path"] = os.path.abspath(new_path)
        if os.path.isfile(model_params["word_id_path"]):
            with open(model_params["word_id_path"], "r") as infile:
                self.train_vocab = json.load(infile)
        else:
            new_path = os.path.join(os.path.dirname(model_params_path), "word_id.json")
            print("{} not found, switching to model_params' path {}".format(model_params["word_id_path"], new_path))
            with open(new_path, "r") as infile:
                self.train_vocab = json.load(infile)
            self.info["word_id_path"] = os.path.abspath(new_path)
        self.preprocessing_function = preprocessing_function

        get_list = ["input", "input_weights", "embeddings/embedding_matrix/read",
                    "mean_sentece_vector/sentence_embedding", "logits/kernel/read", "prediction"]
        get_list = [i + ":0" for i in get_list]

        self._device = "/cpu:0"
        if use_gpu:
            self._device = "/gpu:0"
            config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                                              allow_growth=True))
        else:
            config = tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)

        with tf.device(self._device):
            self._sess = tf.Session(config=config)
            self._input_ph, self._weights_ph, self._input_mat, self._sent_vec, self._output_mat, self._output = \
                load_graph(model_path, get_list)

        self._dim = self.get_dimension()
        _ = self.predict([""] * 3, show_progress=False)  # warm-up)

    def __del__(self):
        with tf.device(self._device):
            self._sess.close()

    def get_dimension(self):
        """
        Get the dimension (size) of a lookup vector (hidden layer).
        :return: int
        """
        return int(self._sent_vec.shape[1])

    def get_input_matrix(self):
        """
        Get a copy of the full input matrix of a Model.
        :return: np.ndarray, size: word_count * dim
        """
        return self._sess.run(self._input_mat)

    def get_input_vector(self, ind):
        """
        Given an index, get the corresponding vector of the Input Matrix.
        :param ind: int
        :return: np.ndarray, size: dim
        """
        return self._sess.run(self._input_mat[ind])

    def get_labels(self, include_freq=False):
        """
        Get the entire list of labels of the dictionary optionally including the frequency of the individual labels.
        :param include_freq: bool, returns tuple with labels and their frequencies
        :return: list / tuple of lists
        """
        labels = sorted(self.label_vocab.keys())
        if include_freq:
            return labels, [self.label_vocab[k]["cnt"] for k in labels]
        return labels

    def get_line(self, text):
        """
        Preprocess the text and split it into words and labels. Labels must start with the prefix used to create the
        model (__label__ by default) and have been used in training.
        :param text: str
        :return: (list, list)
        """
        words = " ".join([i for i in text.split() if not i.startswith(self.label_prefix)])
        if self.preprocessing_function:
            words = self.preprocessing_function(words)
        return words.split(), [i for i in text.split() if i.startswith(self.label_prefix) and
                               (i[len(self.label_prefix):]) in self.label_vocab]

    def get_output_matrix(self):
        """
        Get a copy of the full output matrix of a Model.
        :return: np.ndarray, size: dim * label_count
        """
        return self._sess.run(self._output_mat)

    def get_sentence_vector(self, text, batch_size=100):
        """
        Given a string or list of string, get its (theirs) vector represenation(s). This function applies
        preprocessing function on the strings.
        :param text: str or list/array
        :param batch_size: int
        :return: np.ndarray, size: dim
        """
        t = type(text)
        assert t in [list, str, np.ndarray]
        if t == str:
            text = [text]
        embs = []

        for batch, batch_weights in self._batch_generator(text, batch_size):
            embs.extend(self._sess.run(self._sent_vec, feed_dict={self._input_ph: batch,
                                                                  self._weights_ph: batch_weights}))
        return np.squeeze(embs)

    def get_word_id(self, word):
        """
        Given a word, get the word id within the dictionary. Returns -1 if word is not in the dictionary.
        :param word:
        :return: int. Returns -1 if is not in vocabulary
        """
        return self.train_vocab[word]["id"] if word in self.train_vocab else -1

    def get_word_vector(self, word):
        """
        Get the vector representation of word.
        :param word: str
        :return: np.ndarray, size: dim. returns 0s if not from vocabulary
        """
        if self.preprocessing_function:
            word_id = self.get_word_id(self.preprocessing_function(word))
        else:
            word_id = self.get_word_id(word)
        return self.get_input_vector(word_id) if word_id != -1 else np.zeros(self._dim, dtype=np.float32)

    def get_words(self, include_freq=False):
        """
        Get the entire list of words of the dictionary optionally including the frequency of the individual words.
        :param include_freq: bool, returns tuple with words and their frequencies
        :return: list / tuple of lists
        """
        words = sorted(self.train_vocab.keys())
        if include_freq:
            return words, [self.train_vocab[k]["cnt"] for k in words]
        return words

    def _batch_generator(self, list_of_texts, batch_size):
        """
        Generate batch from list of texts
        :param list_of_texts: list/array
        :param batch_size: int
        :return: batch word indices, batch word weights
        """
        if self.preprocessing_function:
            list_of_texts = [self.preprocessing_function(str(t)) for t in list_of_texts]
        else:
            list_of_texts = [str(t) for t in list_of_texts]
        inds = np.arange(len(list_of_texts))
        rem_inds, batch_inds = next_batch(inds, batch_size)

        while len(batch_inds) > 0:
            batch, batch_weights = [], []

            descs_words = [list(get_all(list_of_texts[ind].split(), self.info["word_ngrams"], self.info["sort_ngrams"]))
                           for ind in batch_inds]
            num_max_words = max([len(desc_split) for desc_split in descs_words]) + 1

            for desc_words in descs_words:
                init_test_inds = [0] + [self.train_vocab[phrase]["id"] for phrase in desc_words
                                        if phrase in self.train_vocab]

                test_desc_inds = init_test_inds + [0 for _ in range(num_max_words - len(init_test_inds))]
                test_desc_weights = np.zeros_like(test_desc_inds, dtype=float)
                test_desc_weights[:len(init_test_inds)] = 1. / len(init_test_inds)

                batch.append(test_desc_inds)
                batch_weights.append(test_desc_weights)
            rem_inds, batch_inds = next_batch(rem_inds, batch_size)
            batch_weights = np.expand_dims(batch_weights, 2)
            batch = np.array(batch)

            yield batch, batch_weights

    def predict(self, list_of_texts, k=1, batch_size=100, threshold=-0.1, show_progress=True):
        """
        Predict top k predictions on given texts
        :param list_of_texts: list/array
        :param k: int, top k predictions
        :param batch_size: int
        :param threshold: float, from 0 to 1, default -0.1 meaining no threshold
        :param show_progress: bool, ignored if list of text is string or has smaller or equal length to batch size
        :return: top k predictions and probabilities
        """
        if type(list_of_texts) == str:
            list_of_texts = [list_of_texts]

        labels = self.get_labels()
        preds, probs = [], []

        if len(list_of_texts) <= batch_size:
            show_progress = False

        if show_progress:
            progress_bar = tqdm(total=int(np.ceil(len(list_of_texts) / batch_size)))
        for batch, batch_weights in self._batch_generator(list_of_texts, batch_size):
            batch_probs = self._sess.run(self._output, feed_dict={self._input_ph: batch,
                                                                  self._weights_ph: batch_weights})

            top_k_probs, top_k_preds = [], []
            for i in batch_probs:
                pred_row, prob_row = [], []
                if k == -1:
                    top_k_inds = np.argsort(i)[::-1]
                else:
                    top_k_inds = np.argsort(i)[-k:][::-1]
                for ind, prob in zip(top_k_inds, i[top_k_inds]):
                    if prob > threshold:
                        pred_row.append(ind)
                        prob_row.append(prob)
                top_k_preds.append([labels[i] for i in pred_row])
                top_k_probs.append(prob_row)
            preds.extend(top_k_preds)
            probs.extend(top_k_probs)
            if show_progress:
                progress_bar.update()
        if show_progress:
            progress_bar.close()
        return preds, probs

    def test(self, list_of_texts, list_of_labels, batch_size=100, k=1, threshold=-0.1, show_progress=True):
        """
        Predict top k predictions on given texts
        :param list_of_texts: list/array
        :param list_of_labels: list/array
        :param k: int, top k predictions
        :param batch_size: int
        :param threshold: float, from 0 to 1. Default is -0.1 meaining no threshold
        :param show_progress: bool
        :return: top k predictions and probabilities
        """
        assert len(list_of_texts) == len(list_of_labels), 'the lengths of list_of_texts and list_of_labels must match'

        preds, probs = self.predict(list_of_texts=list_of_texts, batch_size=batch_size, k=k,
                                    threshold=threshold, show_progress=show_progress)
        recall, precision = 0, 0
        total_lbs, total_preds = 0, 0
        for lbs, prds in zip(list_of_labels, preds):
            if type(lbs) != list:
                lbs = [lbs]

            total_lbs += len(lbs)
            total_preds += len(prds)
            for lb in lbs:
                if lb in prds:
                    recall += 1
            for prd in prds:
                if prd in lbs:
                    precision += 1

        return len(list_of_texts), round(precision / total_preds, 5), round(recall / total_lbs, 5)

    def test_file(self, test_data_path, batch_size=100, k=1, threshold=-0.1, show_progress=True):
        """
        Predict top k predictions on given texts
        :param test_data_path: str, path to test file
        :param batch_size: int
        :param k: int, top k predictions
        :param threshold: float, from 0 to 1, default -0.1 meaining no threshold
        :param show_progress: bool
        :return: top k predictions and probabilities
        """
        data, labels = parse_txt(test_data_path, label_prefix=self.label_prefix, join_desc=True)
        return self.test(data, labels, batch_size=batch_size, k=k, threshold=threshold, show_progress=show_progress)

    def export_model(self, destination_path):
        """
        Extract all the needed files for model loading to the specified destination.
        Also copies the training and validation files if available
        :param destination_path: str
        :return: None
        """
        all_paths = [v for k, v in self.info.items() if "path" in k]
        if "train_path" in self.hyperparams:
            all_paths.append(self.hyperparams["train_path"])

        if "validation_path" in self.hyperparams:
            all_paths.append(self.hyperparams["validation_path"])

        if "original_train_path" in self.hyperparams:
            all_paths.append(self.hyperparams["original_train_path"])
            all_paths.extend(self.hyperparams["additional_data_paths"])

        if "split_and_train_params" in self.hyperparams:
            all_paths.append(self.hyperparams["split_and_train_params"]["df_path"])
        copy_all(all_paths, destination_path)


class train_supervised(FastTextModel):
    def __init__(self, train_data_path, val_data_path=None, additional_data_paths=None, hyperparams={},
                 preprocessing_function=None, log_dir="./", use_gpu=False, verbose=True, remove_extra_labels=True):
        """
        Train a supervised fasttext model
        :param train_data_path: str, path to train.txt file
        :param val_data_path: str, path to val.txt file. if val_data_path is None the score won't be keeped in
        history.json
        :param additional_data_paths: list of str, paths of fasttext format additional data to concat with train file
        :param hyperparams: dict, all hyperparams for train_supervised
        :param preprocessing_function: function, function to apply on text data before feeding into network
        :param log_dir: str, directory to save the training files and the model
        :param use_gpu: bool, use gpu for training
        :param verbose: bool
        :param remove_extra_labels: bool, remove datapoints with labels which appear in additional_data_paths but not in
        train_data_path. Ignored if additional_data_paths is None
        :return: object, the trained model
        """
        log_dir = validate(log_dir)
        self.hyperparams = \
            {"train_path": handle_space_paths("./train.txt"),
             "validation_path": handle_space_paths(""),
             "min_word_count": 1,
             "min_label_count": 1,
             "label_prefix": "__label__",
             "dim": 100,
             "n_epochs": 10,
             "word_ngrams": 1,
             "sort_ngrams": 0,
             "batch_size": 1024,
             "batch_size_inference": 1024,
             "batch_norm": 0,
             "seed": 17,
             "top_k": 5,
             "learning_rate": 0.3,
             "learning_rate_multiplier": 0.8,
             "dropout": 0.5,
             "l2_reg_weight": 1e-06,
             "data_fraction": 1,
             "save_models": 0,
             "use_validation": 0,
             "use_gpu": 0,
             "gpu_fraction": 0.5,
             "force": 0,
             "cache_dir": handle_space_paths(os.path.abspath(os.path.join(log_dir, "cache"))),
             "result_dir": handle_space_paths(os.path.abspath(os.path.join(log_dir, "results"))),
             "flush": 1}

        assert os.path.exists(train_data_path), "train_data_path is incorrect"
        if val_data_path:
            assert os.path.exists(val_data_path), "val_data_path is incorrect"
            self.hyperparams["use_validation"] = 1
            self.hyperparams["validation_path"] = val_data_path

        to_restore = {}
        if len(hyperparams) != 0:
            for k, v in hyperparams.items():
                if k not in self.hyperparams:
                    to_restore[k] = v
                    if k != "split_and_train_params":
                        print("WARNING! {} not in hyperparams, ignoring it".format(k))
                else:
                    if k in ["train_path", "validation_path", "cache_dir", "result_dir"]:
                        self.hyperparams[k] = handle_space_paths(v)
                    else:
                        self.hyperparams[k] = v

        train_data_path = os.path.abspath(train_data_path)
        if additional_data_paths:
            data_to_save = []
            paths_joined_hashed = hash_(" ".join(additional_data_paths))
            concat_path = "/tmp/tmp.txt"
            joined_path = "/tmp/{}.txt".format(paths_joined_hashed)
            os.system("cat {} {} > {}".format(train_data_path, val_data_path, concat_path))
            _, all_labels = parse_txt(train_data_path)
            unique_labels = set(all_labels)
            assert type(additional_data_paths) == list, "type of additional_data_paths should be list"
            for additional_data_path in additional_data_paths:
                assert os.path.exists(additional_data_path), "val_data_path is incorrect"
                current_data, current_labels = parse_txt(additional_data_path, join_desc=True)
                if remove_extra_labels:
                    needed_inds = [i for i, j in enumerate(current_labels) if j in unique_labels]
                    current_data = [current_data[i] for i in needed_inds]
                    current_labels = [current_labels[i] for i in needed_inds]
                data_to_save.extend(["{}{} {}".format(self.hyperparams["label_prefix"], i, j) for i, j
                                     in zip(current_labels, current_data)])
            with open(concat_path, "w+") as outfile:
                outfile.write("\n".join(data_to_save))
            os.system("cat {} {} > {}".format(concat_path, train_data_path, joined_path))
            self.hyperparams["train_path"] = joined_path
            to_restore["original_train_path"] = train_data_path
            to_restore["additional_data_paths"] = additional_data_paths
        else:
            self.hyperparams["train_path"] = train_data_path

        if use_gpu:
            self.hyperparams["use_gpu"] = 1

        command = self._get_command()
        process = Popen(command, stdout=PIPE, shell=True, stderr=STDOUT, bufsize=1, close_fds=True)

        for line in iter(process.stdout.readline, b""):
            line = line.rstrip().decode("utf-8")
            if "stored at" in line:
                log_dir_line = line

            if "accuracy" in line:
                line_split = line.split()
                if "val" in line:
                    self.top_1_accuracy = float(line_split[-4][:-1])
                    self.top_k_accuracy = float(line_split[-1])
                else:
                    if str(1) in line.split():
                        self.top_1_accuracy = float(line_split[-1])
                    if str(self.hyperparams["top_k"]) in line.split():
                        self.top_k_accuracy = float(line_split[-1])

            if verbose:
                print(line)
        process.stdout.close()

        log_dir_split = log_dir_line.split("at ")
        for k, v in to_restore.items():
            self.hyperparams[k] = v
        super(train_supervised, self). \
            __init__(model_path=os.path.join(log_dir_split[-1], "model_ep{}.pb".format(self.hyperparams["n_epochs"])),
                     model_params_path=os.path.join(log_dir_split[-1], "model_params.json"),
                     use_gpu=use_gpu, label_prefix=self.hyperparams["label_prefix"],
                     preprocessing_function=preprocessing_function,
                     hyperparams=self.hyperparams)

    def _get_command(self):
        args = ["--{} {}".format(k, v) for k, v in self.hyperparams.items()]
        cur_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
        command = " ".join(["python3 {}".format(os.path.join(cur_dir, "main.py"))] + args)
        return command


def split_and_train(path_to_df, text_field, label_field, split_params={}, save_dir="./", preprocessing_function=None,
                    additional_fields_and_preps={}, additional_data_paths=[], hyperparams={}, log_dir="./",
                    use_gpu=False, postfix="", verbose=True, remove_extra_labels=True):
    """
    Split dataframe with the given params into train and test. Train a model on train and
    :param path_to_df: str, path to csv or parquet file
    :param text_field: str, column of the dataframe in which is the text that should be classified
    :param label_field: str, column of the dataframe in which is the label of the corresponding text
    :param split_params: dict, input format: {"seed": int, default 17, "fraction": float, default: 0.1}
    :param save_dir: str, directory to save the txt files
    :param preprocessing_function: function, function to apply on text_field column
    :param additional_fields_and_preps: dict. Dictionary in the following format
    {field_name1: preprocessing_function1, field_name2: preprocessing_function2} to enable custom preprocessing for
    different fields
    :param additional_data_paths: list of str, paths of fasttext format additional data to concat with train file
    :param hyperparams: dict, all hyperparams for train_supervised
    :param log_dir: str, directory to save the training files and the model
    :param postfix: str, postfix to add to train and validation files
    :param verbose: bool
    :param use_gpu: bool, use gpu for training
    :param verbose: bool
    :param remove_extra_labels: remove datapoints with labels which appear in additional_data_paths but not in
    train_data_path
    :return: object. FastTextModel
    """

    train_data_path, val_data_path = \
        train_val_split_from_df(path_to_df=path_to_df, text_field=text_field, label_field=label_field,
                                split_params=split_params, save_dir=save_dir,
                                preprocessing_function=preprocessing_function, verbose=verbose,
                                additional_fields_and_preps=additional_fields_and_preps, postfix=postfix)
    if verbose:
        print("train path {}".format(train_data_path))
        print("val path {}".format(val_data_path))

    hypers_new = hyperparams.copy()

    if additional_fields_and_preps:
        hypers_new["result_dir"] = os.path.join(log_dir, "{}_{}".format(hash_function(preprocessing_function),
                                                                        "_".join(additional_fields_and_preps.keys())))
    else:
        hypers_new["result_dir"] = os.path.join(log_dir, hash_function(preprocessing_function))
    hypers_new["use_gpu"] = int(use_gpu)
    hypers_new["split_and_train_params"] = {
        "df_path": path_to_df, "split_params": split_params,
        "additional_fields_and_preps": additional_fields_and_preps, "remove_extra_labels": remove_extra_labels
    }

    return train_supervised(train_data_path=train_data_path, val_data_path=val_data_path,
                            additional_data_paths=additional_data_paths, hyperparams=hypers_new,
                            preprocessing_function=preprocessing_function, remove_extra_labels=remove_extra_labels,
                            log_dir=log_dir, use_gpu=use_gpu, verbose=verbose)


def cross_validate(path_to_df, text_field, label_field, n_folds=5, preprocessing_function=None,
                   additional_fields_and_preps={}, additional_data_paths=[], hyperparams={}, report_top_k=True,
                   log_dir="./", use_gpu=False, return_models=False, seed=17, verbose=False, remove_extra_labels=True):
    """

    :param path_to_df: str, path to csv or parquet file
    :param text_field: str, column of the dataframe in which is the text that should be classified
    :param label_field: str, column of the dataframe in which is the label of the corresponding text
    :param n_folds: int, number of folds
    :param preprocessing_function: function, function to apply on text_field column
    :param additional_fields_and_preps: dict. Dictionary in the following format
    {field_name1: preprocessing_function1, field_name2: preprocessing_function2} to enable custom preprocessing for
    different fields
    :param additional_data_paths: list of str, paths of fasttext format additional data to concat with train file
     :param hyperparams: dict, all hyperparams for train_supervised
    :param report_top_k: bool. If True will return top k scores, otherwise top 1 scores
    :param log_dir: str, directory to save the training files and the model
    :param use_gpu: bool, use gpu for training
    :param return_models: bool. If True will return tuple (scores, models)
    :param seed: int
    :param verbose: bool.
    :param remove_extra_labels: remove datapoints with labels which appear in additional_data_paths but not in
    train_data_path
    :return: list. The scores for each split
    """
    models, scores = [], []

    if path_to_df.endswith("parquet"):
        df = pd.read_parquet(path_to_df)
    else:
        df = pd.read_csv(path_to_df)

    for added_field, prep_f in additional_fields_and_preps.items():
        if df[added_field].dtype != "object":
            df[added_field] = df[added_field].astype(str)
        if prep_f:
            df[added_field] = df[added_field].map(prep_f)
        df[text_field] = df[text_field] + " " + df[added_field]

    for fold_number, val_mask in enumerate(split_list(len(df), n_folds, seed)):
        train_data_path, val_data_path = preprocess_and_save(df, val_mask, text_field, label_field,
                                                             preprocessing_function, additional_fields_and_preps,
                                                             "./tmp_txt/", "_split{}".format(fold_number), verbose, [])

        if verbose:
            print("train path {}".format(train_data_path))
            print("val path {}".format(val_data_path))

        hypers_new = hyperparams.copy()

        if additional_fields_and_preps:
            hypers_new["result_dir"] = os.path.join(log_dir, "{}_{}".format(hash_function(preprocessing_function),
                                                                            "_".join(
                                                                                additional_fields_and_preps.keys())))
        else:
            hypers_new["result_dir"] = os.path.join(log_dir, hash_function(preprocessing_function))
        hypers_new["use_gpu"] = int(use_gpu)
        hypers_new["split_and_train_params"] = {
            "df_path": path_to_df,
            "additional_fields_and_preps": additional_fields_and_preps, "remove_extra_labels": remove_extra_labels
        }

        model = train_supervised(train_data_path=train_data_path, val_data_path=val_data_path,
                                 additional_data_paths=additional_data_paths, hyperparams=hypers_new,
                                 preprocessing_function=preprocessing_function, remove_extra_labels=remove_extra_labels,
                                 log_dir=log_dir, use_gpu=use_gpu, verbose=verbose)

        if report_top_k:
            scores.append(model.top_k_accuracy)
        else:
            scores.append(model.top_1_accuracy)
        if return_models:
            models.append(model)
        del model
        gc.collect()
    if return_models:
        return scores, models
    return scores


def test_on_df(model_or_path, path_to_df, text_field, label_field, preprocessing_function=None,
               additional_fields_and_preps={}, label_prefix="__label__", use_gpu=False, top_k=5, batch_size=100):
    """
     Test the given model on a parquet or csv file.
    :param model_or_path: str or object. FastTextModel or the directory where it is located
    :param path_to_df: str. Path to csv or parquet file
    :param text_field: str. Field of dataframe which contains text to be classified
    :param label_field: str. Field of dataframe which contains the labels
    :param preprocessing_function: function. Function to apply on texts before prediction
    :param additional_fields_and_preps: dict. Dictionary in the following format
    {field_name1: preprocessing_function1, field_name2: preprocessing_function2} to enable custom preprocessing for
    different fields
    :param label_prefix: str. Default - "__label__"
    :param use_gpu: bool. Use gpu for the inference
    :param top_k: int. Calculate scores for top k prediction
    :param batch_size: int
    :return: tuple. (Recall on top k, Recall on top 1)
    """
    if type(model_or_path) == str:
        all_models = [i for i in os.listdir(model_or_path) if i.endswith("pb")]
        last_model = max(all_models, key=lambda m: int(m.split("_ep")[-1].split(".")[0]))

        model = FastTextModel(model_path=os.path.join(model_or_path, last_model),
                              model_params_path=os.path.join(model_or_path, "model_params.json"),
                              use_gpu=use_gpu, label_prefix=label_prefix,
                              preprocessing_function=preprocessing_function)

    else:
        model = model_or_path
        if preprocessing_function:
            model.preprocessing_function = preprocessing_function

    if path_to_df.endswith("parquet"):
        df = pd.read_parquet(path_to_df)
    elif path_to_df.endswith("csv"):
        df = pd.read_csv(path_to_df)
    else:
        print("dataframe should be either csv or parquet file.\nexiting.")
        exit()
    model_unique_labels = set(model.get_labels())
    if model.preprocessing_function:
        if df[text_field].dtype == "object":
            df[text_field] = df[text_field].map(model.preprocessing_function)
        else:
            df[text_field] = df[text_field].map(lambda s: model.preprocessing_function(str(s)))

    for added_field, prep_f in additional_fields_and_preps.items():
        if df[added_field].dtype != "object":
            df[added_field] = df[added_field].astype(str)
        if prep_f:
            df[added_field] = df[added_field].map(prep_f)
        df[text_field] = df[text_field] + " " + df[added_field]

    preds, _ = model.predict(list_of_texts=df[text_field], batch_size=batch_size, k=top_k)
    right_preds_top_1, right_preds_top_k, new_label_cnt = 0, 0, 0
    for true_label, preds_k in zip(df[label_field], preds):
        if true_label not in model_unique_labels:
            new_label_cnt += 1
        if true_label == preds_k[0]:
            right_preds_top_1 += 1
        if true_label in preds_k:
            right_preds_top_k += 1
    if new_label_cnt > 0:
        print('{}% of data is missclassified because of new label'.format(round(100 * new_label_cnt / len(df), 2)))
    return np.round([100 * right_preds_top_k / len(df), 100 * right_preds_top_1 / len(df)], 2)
