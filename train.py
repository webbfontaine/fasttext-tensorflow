import os
import json
import time
import tracemalloc

import tensorflow as tf
import numpy as np

from fasttext_utils import (
    batch_generator,
    parse_txt,
)
from utils import (
    percent_array,
    freeze_save_graph,
    hash_,
)
from fasttext_model import FastTextModel


def run_train(data, train_specific, train_params, data_specific, train_history, train_history_path):
    """
    Run training with the given data, parameters and hyperparameters
    :param data: dict, data
    :param train_specific: dict, train hyper-parameters
    :param train_params: dict, train parameters
    :param data_specific: dict, data-specific parameters
    :param train_history: dict, train history
    :param train_history_path: str, path to train history
    :return: None, prints the training outputs
    """

    seed = train_specific["seed"]
    learning_rate = train_specific["learning_rate"]
    embedding_dim = train_specific["embedding_dim"]
    use_batch_norm = train_specific["use_batch_norm"]
    l2_reg_weight = train_specific["l2_reg_weight"]
    num_epochs = train_specific["num_epochs"]
    batch_size = train_specific["batch_size"]
    train_dropout_keep_rate = train_specific["dropout"]
    learning_rate_multiplier = train_specific["learning_rate_multiplier"]
    cache_dir = train_specific["cache_dir"]
    train_path = train_specific["train_path"]

    train_description_hashes = data["train_description_hashes"]
    train_labels = data["train_labels"]
    test_description_hashes = data["test_description_hashes"]
    test_labels = data["test_labels"]
    label_vocab = data["label_vocab"]
    cache = data["cache"]
    num_words_in_train = data["num_words_in_train"]
    test_path = data["test_path"]
    initial_test_len = data["initial_test_len"]
    num_labels = len(label_vocab)

    use_gpu = train_params["use_gpu"]
    gpu_fraction = train_params["gpu_fraction"]
    top_k = train_params["top_k"]
    save_all_models = train_params["save_all_models"]
    compare_top_k = train_params["compare_top_k"]
    use_test = train_params["use_test"]
    log_dir = train_params["log_dir"]
    batch_size_inference = train_params["batch_size_inference"]
    progress_bar = train_params["progress_bar"]
    flush = train_params["flush"]

    hyperparameter_hash = hash_("".join([str(hyperparam) for hyperparam in train_specific.values()]))

    if use_gpu:
        device = "/gpu:0"
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                                          allow_growth=True))
    else:
        device = "/cpu:0"
        config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device(device):
        with tf.Session(config=config) as sess:
            input_ph = tf.placeholder(tf.int32, shape=[None, None], name="input")
            weights_ph = tf.placeholder(tf.float32, shape=[None, None], name="input_weights")
            labels_ph = tf.placeholder(tf.int32, shape=[None], name="label")
            learning_rate_ph = tf.placeholder_with_default(learning_rate, shape=[], name="learning_rate")
            dropout_drop_rate_ph = tf.placeholder_with_default(0., shape=[], name="dropout_rate")
            is_training = tf.placeholder_with_default(False, shape=[], name="do_dropout")

            tf.set_random_seed(seed)

            with tf.name_scope("embeddings"):
                look_up_table = tf.Variable(tf.random.uniform([num_words_in_train, embedding_dim]),
                                            name="embedding_matrix")

            with tf.name_scope("mean_sentece_vector"):
                gathered_vectors = tf.gather(look_up_table, input_ph)
                weights_broadcasted = tf.expand_dims(weights_ph, axis=2)
                mean_emb = tf.reduce_sum(tf.multiply(weights_broadcasted, gathered_vectors), axis=1,
                                         name="sentence_embedding")
            if use_batch_norm:
                mean_emb = tf.layers.batch_normalization(mean_emb, training=is_training)
            mean_emb_dr = tf.layers.dropout(mean_emb, rate=dropout_drop_rate_ph, training=is_training)
            logits = tf.layers.dense(mean_emb_dr, num_labels, use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(), name="logits")
            output = tf.nn.softmax(logits, name="prediction")
            # this is not used in the training, but will be used for inference

            with tf.name_scope("Accuracy"):
                correctly_predicted = tf.nn.in_top_k(logits, labels_ph, 1, name="Top_1")
                correctly_predicted_top_k = tf.nn.in_top_k(logits, labels_ph, top_k, name="Top_k")

            train_writer = tf.summary.FileWriter(os.path.join(log_dir, "Train"), sess.graph)
            train_end_writer = tf.summary.FileWriter(os.path.join(log_dir, "End_epoch_train"))

            if use_test:
                batch_counter = 0
                test_end_writer = tf.summary.FileWriter(os.path.join(log_dir, "End_epoch_val"))
                test_end_batch_writer = tf.summary.FileWriter(os.path.join(log_dir, "End_epoch_test_batch"))

            ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph,
                                                                                    logits=logits), name="CE_loss")

            l2_vars = tf.trainable_variables()
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]), l2_reg_weight, name="L2_loss")
            total_loss = tf.add(ce_loss, l2_loss, name="Total_loss")

            tf.summary.scalar("Cross_entropy_loss", ce_loss)
            tf.summary.histogram("Mean_embedding", mean_emb)
            summary_op = tf.summary.merge_all()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(total_loss)
            sess.run(tf.global_variables_initializer())

            it = 0
            train_start = time.time()
            best_score = -1
            logs = {1: [], top_k: [], "best": -1}

            for epoch in range(1, num_epochs + 1):
                print("\n\nEpoch {} started".format(epoch), flush=flush)
                losses = []
                end_epoch_accuracy, end_epoch_accuracy_k = [], []
                end_epoch_loss = []
                end_epoch_l2_loss = []
                for batch, batch_weights, batch_labels in \
                        batch_generator(train_description_hashes, train_labels, batch_size, label_vocab, cache,
                                        shuffle=True, show_progress=progress_bar, progress_desc="Fit train"):
                    _, train_summary, _loss, correct, correct_k, batch_loss, batch_l2 = \
                        sess.run([train_op, summary_op, total_loss, correctly_predicted,
                                  correctly_predicted_top_k, ce_loss, l2_loss],
                                 feed_dict={input_ph: batch,
                                            weights_ph: batch_weights,
                                            labels_ph: batch_labels,
                                            learning_rate_ph: learning_rate,
                                            dropout_drop_rate_ph: 1 - train_dropout_keep_rate,
                                            is_training: True})
                    losses.append(_loss)
                    end_epoch_accuracy.extend(correct)
                    end_epoch_accuracy_k.extend(correct_k)
                    end_epoch_loss.append(batch_loss)
                    end_epoch_l2_loss.append(batch_l2)

                    train_writer.add_summary(train_summary, it)
                    it += 1
                print("Current learning rate: {}".format(round(learning_rate, 7)), flush=flush)
                learning_rate *= learning_rate_multiplier

                print("Epoch {} ended".format(epoch), flush=flush)
                print("Epoch moving mean loss: {}".format(percent_array(losses)), flush=flush)

                mean_acc = percent_array(end_epoch_accuracy)
                mean_acc_k = percent_array(end_epoch_accuracy_k)
                summary_acc = tf.Summary(value=[tf.Summary.Value(tag="Accuracy", simple_value=mean_acc)])
                summary_acc_k = tf.Summary(value=[tf.Summary.Value(tag="Accuracy_top_{}".format(top_k),
                                                                   simple_value=mean_acc_k)])
                summary_loss = tf.Summary(value=[tf.Summary.Value(tag="Loss", simple_value=np.mean(end_epoch_loss))])
                summary_loss_l2 = tf.Summary(
                    value=[tf.Summary.Value(tag="L2", simple_value=np.mean(end_epoch_l2_loss))])
                train_end_writer.add_summary(summary_acc, epoch)
                train_end_writer.add_summary(summary_acc_k, epoch)
                train_end_writer.add_summary(summary_loss, epoch)
                train_end_writer.add_summary(summary_loss_l2, epoch)
                print("Train moving average accuracy: {}, top {}: {}".format(mean_acc, top_k, mean_acc_k), flush=flush)

                if use_test:
                    end_epoch_accuracy, end_epoch_accuracy_k = [], []
                    end_epoch_loss = []

                    for batch, batch_weights, batch_labels in \
                            batch_generator(test_description_hashes, test_labels, batch_size_inference, label_vocab,
                                            cache, show_progress=progress_bar, progress_desc="Test"):
                        correct, correct_k, batch_loss = sess.run(
                            [correctly_predicted, correctly_predicted_top_k, ce_loss],
                            feed_dict={input_ph: batch,
                                       weights_ph: batch_weights,
                                       labels_ph: batch_labels})

                        end_epoch_accuracy.extend(correct)
                        end_epoch_accuracy_k.extend(correct_k)
                        end_epoch_loss.append(batch_loss)
                        summary_loss = tf.Summary(
                            value=[tf.Summary.Value(tag="Batch_loss", simple_value=np.mean(batch_loss))])
                        test_end_batch_writer.add_summary(summary_loss, batch_counter)
                        batch_counter += 1
                    mean_acc = np.round(100 * np.sum(end_epoch_accuracy) / initial_test_len, 2)
                    mean_acc_k = np.round(100 * np.sum(end_epoch_accuracy_k) / initial_test_len, 2)
                    summary_acc = tf.Summary(value=[tf.Summary.Value(tag="Accuracy", simple_value=mean_acc)])
                    summary_acc_k = tf.Summary(value=[tf.Summary.Value(tag="Accuracy_top_{}".format(top_k),
                                                                       simple_value=mean_acc_k)])
                    summary_loss = tf.Summary(
                        value=[tf.Summary.Value(tag="Loss", simple_value=np.mean(end_epoch_loss))])

                    test_end_writer.add_summary(summary_acc, epoch)
                    test_end_writer.add_summary(summary_acc_k, epoch)
                    test_end_writer.add_summary(summary_loss, epoch)
                    print("End epoch mean test accuracy: {}, top {}: {}".format(mean_acc, top_k, mean_acc_k),
                          flush=flush)

                logs[1].append(mean_acc)
                logs[top_k].append(mean_acc_k)

                comparable = mean_acc
                if compare_top_k:
                    comparable = mean_acc_k

                if comparable > best_score:
                    best_score = comparable
                    freeze_save_graph(sess, log_dir, "model_best.pb", "prediction")
                    logs["best"] = epoch

                if save_all_models:
                    freeze_save_graph(sess, log_dir, "model_ep{}.pb".format(epoch), "prediction")
                else:
                    if epoch == num_epochs:
                        freeze_save_graph(sess, log_dir, "model_ep{}.pb".format(epoch), "prediction")
            print("Best model mean test accuracy: {}, top {}: {}".format(logs[1][logs["best"] - 1], top_k,
                                                                         logs[top_k][logs["best"] - 1]), flush=flush)
            print("The model is stored at {}".format(log_dir), flush=flush)
            if use_test:
                results = {"hyperparams": train_specific,
                           "scores": {test_path: {top_k: mean_acc_k, 1: mean_acc}}}
            else:
                results = {"hyperparams": train_specific,
                           "scores": {train_path: {top_k: mean_acc_k, 1: mean_acc}}}
            train_history[hyperparameter_hash] = results
            with open(os.path.join(log_dir, "results.json"), "w+") as outfile:
                json.dump(results, outfile)
            with open(os.path.join(cache_dir, "details.json"), "w+") as outfile:
                json.dump(data_specific, outfile)
            with open(train_history_path, "w+") as outfile:
                json.dump(train_history, outfile)
            with open(os.path.join(log_dir, "accuracy_logs.json"), "w+") as outfile:
                json.dump(logs, outfile)

            print("The training took {} seconds".format(round(time.time() - train_start, 0)), flush=flush)
    print("Peak memory usage: {}".format(round(tracemalloc.get_traced_memory()[1] / 1e6, 0)), flush=flush)


def get_accuracy(log_dir, train_params, train_history_path, hyperparameter_hash, train_history, test_path,
                 label_prefix="__label__", flush=True):
    """
    Use an existing model to measure accuracy on the test file
    :param log_dir: str, path to model directory
    :param train_params: dict, training parameters
    :param train_history_path: str, path train history (json file)
    :param hyperparameter_hash: str, hyperparameter hash kept in training history
    :param train_history: dict, train history
    :param test_path: path to test file
    :param label_prefix: label prefix
    :param flush: flush after printing
    :return: None, prints the accuracy of the trained model on the test file
    """
    print("Already trained with those hyper-parameters", flush=flush)
    not_done = False
    top_k = train_params["top_k"]
    if test_path in train_history[hyperparameter_hash]["scores"]:
        if (str(top_k) in train_history[hyperparameter_hash]["scores"][test_path]) and \
                (str(1) in train_history[hyperparameter_hash]["scores"][test_path]):
            for k, v in list(train_history[hyperparameter_hash]["scores"][test_path].items())[::-1]:
                print("The accuracy on top {} was {}".format(k, v), flush=flush)
        else:
            not_done = True
    else:
        not_done = True

    if not_done:
        test_descriptions, test_labels = parse_txt(test_path, join_desc=True, label_prefix=label_prefix)
        model = FastTextModel(model_path=os.path.join(log_dir, "model_best.pb"),
                              model_params_path=os.path.join(log_dir, "model_params.json"), label_prefix=label_prefix,
                              use_gpu=train_params["use_gpu"], gpu_fraction=train_params["gpu_fraction"])
        preds, _ = model.predict(list_of_texts=test_descriptions, k=top_k,
                                 batch_size=train_params["batch_size_inference"])
        right_preds_top_1, right_preds_top_k = 0, 0
        for true_label, preds_k in zip(test_labels, preds):
            if true_label == preds_k[0]:
                right_preds_top_1 += 1
            if true_label in preds_k:
                right_preds_top_k += 1
        top_1_score = round(100 * right_preds_top_1 / len(test_descriptions), 2)
        top_k_score = round(100 * right_preds_top_k / len(test_descriptions), 2)
        print("The accuracy on top {} was {}".format(1, top_1_score), flush=flush)
        print("The accuracy on top {} was {}".format(top_k, top_k_score), flush=flush)
        if test_path not in train_history[hyperparameter_hash]["scores"]:
            train_history[hyperparameter_hash]["scores"][test_path] = {}
        train_history[hyperparameter_hash]["scores"][test_path][1] = top_1_score
        train_history[hyperparameter_hash]["scores"][test_path][top_k] = top_k_score
        with open(train_history_path, "w+") as outfile:
            json.dump(train_history, outfile)
