import tensorflow as tf
import numpy as np
from fasttext_utils import parse_txt, make_train_vocab, make_label_vocab, next_batch, construct_label, \
    check_model_presence, get_all
from utils import validate, freeze_save_graph, hash_, get_cache_hash, percent
from fasttext_model import FastTextModel
import os
import shutil
from tqdm import tqdm
import json
import time
import argparse
from sys import exit, stdout
import tracemalloc

print_default = print
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


cache = {}


def batch_generator(descs, labels, batch_size, train_vocab, labels_lookup, word_ngrams, sort_ngrams, shuffle=False,
                    show_progress=True):
    global cache
    inds = np.arange(len(descs))
    rem_inds, batch_inds = next_batch(inds, batch_size, shuffle)

    if show_progress:
        progress_bar = tqdm(total=int(np.ceil(len(descs) / batch_size)))
    while len(batch_inds) > 0:
        batch_descs = [descs[i] for i in batch_inds]
        desc_hashes = [hash(str(desc)) for desc in batch_descs]
        batch = [[0] + [train_vocab[phrase]["id"] for phrase in get_all(desc, word_ngrams, sort_ngrams) if
                        phrase in train_vocab] if h not in cache else cache[h] for
                 desc, h in zip(batch_descs, desc_hashes)]

        for h, inds in zip(desc_hashes, batch):
            if h not in cache:
                cache[h] = inds
        batch_weights = [[1 / len(i) for _ in range(len(i))] for i in batch]
        batch_labels = [labels[i] for i in batch_inds]
        batch_labels = [labels_lookup[label] for label in batch_labels]

        cur_lens = np.array([len(i) for i in batch])
        mx_len = max(cur_lens)
        to_pad = mx_len - cur_lens

        batch = [i + [0 for _ in range(pad)] for i, pad in zip(batch, to_pad)]
        batch_weights = [i + [0 for _ in range(pad)] for i, pad in zip(batch_weights, to_pad)]

        rem_inds, batch_inds = next_batch(rem_inds, batch_size, shuffle)
        if show_progress:
            progress_bar.update()
        yield batch, np.expand_dims(batch_weights, axis=2), batch_labels

    if show_progress:
        progress_bar.close()


def print_and_flush(any: object):
    print_default(any)
    stdout.flush()


def main():
    main_start = time.time()
    tracemalloc.start()
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, help="path to train file", default="./train.txt")
    parser.add_argument("--validation_path", type=str, help="path to validation file", default="")
    parser.add_argument("--label_prefix", type=str, help="label prefix", default="__label__")
    parser.add_argument("--min_word_count", type=int, default=1,
                        help="discard words which appear less than this number")
    parser.add_argument("--min_label_count", type=int, default=1,
                        help="discard labels which appear less than this number")
    parser.add_argument("--dim", type=int, default=100, help="length of embedding vector")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--word_ngrams", type=int, default=1, help="word ngrams")
    parser.add_argument("--sort_ngrams", type=int, default=0, help="sort ngrams alphabetically")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for train")
    parser.add_argument("--batch_size_inference", type=int, default=1024,
                        help="batch size for inference, ignored if validation_path is not provided")
    parser.add_argument("--batch_norm", type=int, default=0, help="use batch norm")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top_k", type=int, default=5, help="report results for top k predictions")
    parser.add_argument("--learning_rate", type=float, default=0.3, help="learning rate")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.8,
                        help="learning rate multiplier after each epoch")
    parser.add_argument("--dropout", type=float, default=0.5, help="train dropout keep rate")
    parser.add_argument("--l2_reg_weight", type=float, default=1e-6, help="regularization weight")
    parser.add_argument("--data_fraction", type=float, default=1,
                        help="data fraction, if < 1, train (and validation) data will be randomly sampled")
    parser.add_argument("--save_models", type=int, default=0, help="save model after each epoch")
    parser.add_argument("--use_validation", type=int, default=0, help="evaluate on validation data")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu for training")
    parser.add_argument("--gpu_fraction", type=float, default=0.5, help="what fraction of gpu to allocate")
    parser.add_argument("--cache_dir", type=str, help="cache dir", default="./cache/")
    parser.add_argument("--result_dir", type=str, help="result dir", default="./results/")
    parser.add_argument("--force", type=int, default=0, help="force retraining")
    parser.add_argument("--flush", type=int, default=0,
                        help="flush after print, only for running from jupyter notebooks")

    args = parser.parse_args()
    for bool_param in [args.batch_norm, args.save_models, args.use_validation, args.sort_ngrams, args.use_gpu,
                       args.force, args.flush]:
        assert bool_param in [0, 1]
    train_path = args.train_path
    validation_path = args.validation_path
    label_prefix = args.label_prefix
    min_word_count = args.min_word_count
    min_label_count = args.min_label_count
    emb_dim = args.dim
    n_epochs = args.n_epochs
    word_ngrams = args.word_ngrams
    sort_ngrams = bool(args.sort_ngrams)
    batch_size = args.batch_size
    batch_size_inference = args.batch_size_inference
    use_batch_norm = bool(args.batch_norm)
    initial_learning_rate = args.learning_rate
    learning_rate_multiplier = args.learning_rate_multiplier
    l2_reg_weight = args.l2_reg_weight
    train_dropout_keep_rate = args.dropout
    top_k = args.top_k
    save_all_models = bool(args.save_models)
    use_validation = bool(args.use_validation)
    seed = args.seed
    data_fraction = args.data_fraction
    use_gpu = bool(args.use_gpu)
    gpu_fraction = args.gpu_fraction
    cache_dir = args.cache_dir
    result_dir = validate(args.result_dir)
    force = bool(args.force)
    flush = bool(args.flush)

    if flush:
        print = print_and_flush
    else:
        print = print_default

    print('training with arguments:')
    print(args)
    print('\n')

    cache_dir = validate(cache_dir)
    hyperparams_path = "history.json"
    data_specific = {
        "train_path": train_path, "data_fraction": data_fraction, "min_word_count": min_word_count,
        "min_label_count": min_label_count, "word_ngrams": word_ngrams, "sort_ngrams": sort_ngrams, "seed": seed
    }

    np.random.seed(seed)

    train_descs, train_labels, max_words = parse_txt(train_path, return_max_len=True, debug_till_row=-1,
                                                     fraction=data_fraction, seed=seed, label_prefix=label_prefix)

    cache_dir = os.path.abspath(validate(os.path.join(cache_dir, get_cache_hash(list_of_texts=train_descs,
                                                                                data_specific_params=data_specific))))

    train_specific = {"emb_dim": emb_dim, "nep": n_epochs, "bs": batch_size, "lr": initial_learning_rate,
                      "use_batch_norm": use_batch_norm, "lrm": learning_rate_multiplier, "l2_reg": l2_reg_weight,
                      "dropout": train_dropout_keep_rate, "data_cache_dir": cache_dir}

    for k, v in data_specific.items():
        train_specific[k] = v

    hypers_hashed = hash_("".join([str(v) for k, v in sorted(train_specific.items(), key=lambda t: t[0])]))

    model_params = {
        "word_ngrams": word_ngrams,
        "sort_ngrams": sort_ngrams,
        "word_id_path": os.path.abspath(os.path.join(cache_dir, "word_id.json")),
        "label_dict_path": os.path.abspath(os.path.join(cache_dir, "label_dict.json"))
    }

    result_dir = validate(os.path.join(result_dir, hypers_hashed))

    if os.path.exists(hyperparams_path):
        with open(hyperparams_path) as infile:
            already_trained = json.load(infile)
        if hypers_hashed in already_trained and check_model_presence(result_dir, n_epochs):
            if not force:
                if use_validation:
                    print("already trained with those hyper-parameters")
                    if validation_path in already_trained[hypers_hashed]["scores"]:
                        for k, v in list(already_trained[hypers_hashed]["scores"][validation_path].items())[::-1]:
                            print("the accuracy on top {} was {}".format(k, v))

                    else:
                        val_descs, val_labels = parse_txt(validation_path, join_desc=True, label_prefix=label_prefix,
                                                          seed=seed, fraction=data_fraction)
                        model = FastTextModel(model_path=os.path.join(result_dir, "model_ep{}.pb".format(n_epochs)),
                                              model_params_path=os.path.join(result_dir, "model_params.json"),
                                              use_gpu=use_gpu, label_prefix=label_prefix)
                        preds, _ = model.predict(list_of_texts=val_descs, k=top_k, batch_size=batch_size_inference,
                                                 show_progress=False)
                        right_preds_top_1, right_preds_top_k = 0, 0
                        for true_label, preds_k in zip(val_labels, preds):
                            if true_label == preds_k[0]:
                                right_preds_top_1 += 1
                            if true_label in preds_k:
                                right_preds_top_k += 1
                        print("the accuracy on top {} was {}".
                              format(1, round(100 * right_preds_top_1 / len(val_descs), 2)))
                        print("the accuracy on top {} was {}".
                              format(top_k, round(100 * right_preds_top_k / len(val_descs), 2)))

                print("the model is stored at {}".format(result_dir))
                exit()
            else:
                print("forced retraining")
                print("training hyper-parameters hashed: {}".format(hypers_hashed))
        else:
            print("training hyper-parameters hashed: {}".format(hypers_hashed))
    else:
        already_trained = {}

    for child_dir in os.listdir(result_dir):
        dir_tmp = os.path.join(result_dir, child_dir)
        if os.path.isdir(dir_tmp):
            shutil.rmtree(dir_tmp)
        if dir_tmp.endswith(".pb"):
            os.remove(dir_tmp)

    max_words_with_ng = 1
    for ng in range(word_ngrams):
        max_words_with_ng += max_words - ng

    print("preparing dataset")
    print("total number of datapoints: {}".format(len(train_descs)))
    print("max number of words in description: {}".format(max_words))
    print("max number of words with n-grams in description: {}".format(max_words_with_ng))

    label_dict_path = os.path.join(cache_dir, "label_dict.json")
    word_id_path = os.path.join(cache_dir, "word_id.json")

    if os.path.isfile(label_dict_path) and os.path.isfile(word_id_path) and not force:
        print("\n*** using cached dicts ***")
        using_cached = True
        with open(label_dict_path, "r") as infile:
            label_vocab = json.load(infile)
        with open(word_id_path, "r") as infile:
            train_vocab = json.load(infile)
        tmp_cnt = sum([i["cnt"] for i in train_vocab.values()])
        print("read {}m words and phrases".format(round(tmp_cnt / 1e6, 1)))
        print("number of unique words and phrases: {}\n".format(len(train_vocab)))
    else:
        using_cached = False
        train_vocab = make_train_vocab(train_descs, word_ngrams, sort_ngrams=sort_ngrams)
        label_vocab = make_label_vocab(train_labels)

    if min_word_count > 1:
        tmp_cnt = 1
        train_vocab_thresholded = {}
        for k, v in sorted(train_vocab.items(), key=lambda t: t[0]):
            if v["cnt"] >= min_word_count:
                v["id"] = tmp_cnt
                train_vocab_thresholded[k] = v
                tmp_cnt += 1

        train_vocab = train_vocab_thresholded.copy()
        del train_vocab_thresholded

        print("number of unique words and phrases after thresholding: {}".format(len(train_vocab)))

    print("\nnumber of labels in train: {}".format(len(set(label_vocab.keys()))))
    if min_label_count > 1:
        label_vocab_thresholded = {}
        tmp_cnt = 0
        for k, v in sorted(label_vocab.items(), key=lambda t: t[0]):
            if v["cnt"] >= min_label_count:
                v["id"] = tmp_cnt
                label_vocab_thresholded[k] = v
                tmp_cnt += 1

        label_vocab = label_vocab_thresholded.copy()
        del label_vocab_thresholded

        print("number of unique labels after thresholding: {}".format(len(label_vocab)))

    final_train_labels = set(label_vocab.keys())

    if not using_cached:
        with open(label_dict_path, "w+") as outfile:
            json.dump(label_vocab, outfile)
        with open(word_id_path, "w+") as outfile:
            json.dump(train_vocab, outfile)
    with open(os.path.join(result_dir, "model_params.json"), "w+") as outfile:
        json.dump(model_params, outfile)

    num_words_in_train = len(train_vocab)
    num_labels = len(label_vocab)

    train_descs2, train_labels2 = [], []
    labels_lookup = {}

    labels_thrown, descs_thrown = 0, 0
    for train_desc, train_label in zip(tqdm(train_descs), train_labels):
        final_train_inds = [0] + [train_vocab[phrase]["id"] for phrase in
                                  get_all(train_desc, word_ngrams, sort_ngrams) if
                                  phrase in train_vocab]
        if len(final_train_inds) == 1:
            descs_thrown += 1
            continue

        if train_label not in labels_lookup:
            if train_label in final_train_labels:
                labels_lookup[train_label] = construct_label(label_vocab[train_label]["id"], num_labels)
            else:
                labels_thrown += 1
                continue

        train_labels2.append(train_label)
        train_descs2.append(train_desc)
    del train_descs, train_labels

    print("\n{} datapoints thrown because of empty description".format(descs_thrown))
    if min_label_count > 1:
        print("{} datapoints thrown because of label".format(labels_thrown))

    if use_validation:
        val_descs, val_labels, max_words_val = parse_txt(validation_path, return_max_len=True,
                                                         label_prefix=label_prefix, seed=seed, fraction=data_fraction)
        max_words_with_ng_val = 1
        for ng in range(word_ngrams):
            max_words_with_ng_val += max_words_val - ng

        print("\ntotal number of val datapoints: {}".format(len(val_descs)))
        val_descs2, val_labels2 = [], []
        num_thrown_for_label = 0
        batch_counter = 0

        for val_desc, val_label in zip(val_descs, val_labels):
            if val_label not in labels_lookup:
                num_thrown_for_label += 1
                continue

            val_descs2.append(val_desc)
            val_labels2.append(val_label)

        val_labels_set = set(val_labels2)

        print("{} datapoints thrown because of label".format(num_thrown_for_label))
        print("number of val datapoints after cleaning: {}".format(len(val_descs2)))
        print("number of unique labels in val after cleaning: {}".format(len(val_labels_set)))
        initial_val_len = len(val_descs)
        del val_descs, val_labels

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
            weights_ph = tf.placeholder(tf.float32, shape=[None, None, 1], name="input_weights")
            labels_ph = tf.placeholder(tf.float32, shape=[None, num_labels], name="label")
            learning_rate_ph = tf.placeholder_with_default(initial_learning_rate, shape=[], name="learning_rate")
            dropout_drop_rate_ph = tf.placeholder_with_default(0., shape=[], name="dropout_rate")
            is_training = tf.placeholder_with_default(False, shape=[], name="do_dropout")

            tf.set_random_seed(seed)

            with tf.name_scope("embeddings"):
                look_up_table = tf.Variable(tf.random_uniform([num_words_in_train + 1, emb_dim]),
                                            name="embedding_matrix")

            with tf.name_scope("mean_sentece_vector"):
                gath_vecs = tf.gather(look_up_table, input_ph)
                weights_broadcasted = tf.tile(weights_ph, tf.stack([1, 1, emb_dim]))
                mean_emb = tf.reduce_sum(tf.multiply(weights_broadcasted, gath_vecs), axis=1, name="sentence_embedding")
            if use_batch_norm:
                mean_emb = tf.layers.batch_normalization(mean_emb, training=is_training)
            mean_emb_dr = tf.layers.dropout(mean_emb, rate=dropout_drop_rate_ph, training=is_training)
            logits = tf.layers.dense(mean_emb_dr, num_labels, use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(), name="logits")
            output = tf.nn.softmax(logits, name="prediction")
            # this is not used in the training, but will be used for inference

            label_argmax = tf.argmax(labels_ph, axis=1)
            with tf.name_scope("accuracy"):
                correctly_predicted = tf.nn.in_top_k(logits, label_argmax, 1, name="top_1")
                correctly_predicted_top_k = tf.nn.in_top_k(logits, label_argmax, top_k, name="top_k")

            train_writer = tf.summary.FileWriter(os.path.join(result_dir, "train"), sess.graph)
            train_end_writer = tf.summary.FileWriter(os.path.join(result_dir, "end_epoch_train"))

            if use_validation:
                val_end_writer = tf.summary.FileWriter(os.path.join(result_dir, "end_epoch_val"))
                val_end_batch_writer = tf.summary.FileWriter(os.path.join(result_dir, "end_epoch_val_batch"))

            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph,
                                                                                logits=logits), name="ce_loss")

            l2_vars = tf.trainable_variables()
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]), l2_reg_weight, name="l2_loss")
            total_loss = tf.add(ce_loss, l2_loss, name="total_loss")

            tf.summary.scalar("cross_entropy_loss", ce_loss)
            tf.summary.histogram("mean_embedding", mean_emb)
            summary_op = tf.summary.merge_all()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(total_loss)
            sess.run(tf.global_variables_initializer())

            it = 0
            train_start = time.time()

            for epoch in range(n_epochs):
                print("\nepoch {} started".format(epoch + 1))
                losses = []
                end_epoch_accuracy, end_epoch_accuracy_k = [], []
                end_epoch_loss = []
                end_epoch_l2_loss = []
                for batch, batch_weights, batch_labels in \
                        batch_generator(train_descs2, train_labels2, batch_size, train_vocab, labels_lookup,
                                        word_ngrams, sort_ngrams, shuffle=True):
                    _, train_summary, _loss, correct, correct_k, batch_loss, batch_l2 = \
                        sess.run([train_op, summary_op, total_loss, correctly_predicted,
                                  correctly_predicted_top_k, ce_loss, l2_loss],
                                 feed_dict={input_ph: batch,
                                            weights_ph: batch_weights,
                                            labels_ph: batch_labels,
                                            learning_rate_ph: initial_learning_rate,
                                            dropout_drop_rate_ph: 1 - train_dropout_keep_rate,
                                            is_training: True})
                    losses.append(_loss)
                    end_epoch_accuracy.extend(correct)
                    end_epoch_accuracy_k.extend(correct_k)
                    end_epoch_loss.append(batch_loss)
                    end_epoch_l2_loss.append(batch_l2)

                    train_writer.add_summary(train_summary, it)
                    it += 1
                print('\ncurrent learning rate: {}'.format(round(initial_learning_rate, 7)))
                initial_learning_rate *= learning_rate_multiplier

                print("epoch {} ended".format(epoch + 1))
                print("epoch moving mean loss: {}".format(percent(losses)))

                mean_acc = percent(end_epoch_accuracy)
                mean_acc_k = percent(end_epoch_accuracy_k)
                summary_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=mean_acc)])
                summary_acc_k = tf.Summary(value=[tf.Summary.Value(tag="accuracy_top_{}".format(top_k),
                                                                   simple_value=mean_acc_k)])
                summary_loss = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=np.mean(end_epoch_loss))])
                summary_loss_l2 = tf.Summary(
                    value=[tf.Summary.Value(tag="l2", simple_value=np.mean(end_epoch_l2_loss))])
                train_end_writer.add_summary(summary_acc, epoch + 1)
                train_end_writer.add_summary(summary_acc_k, epoch + 1)
                train_end_writer.add_summary(summary_loss, epoch + 1)
                train_end_writer.add_summary(summary_loss_l2, epoch + 1)
                print("train moving average accuracy: {}, top {}: {}".format(mean_acc, top_k, mean_acc_k))

                if use_validation:
                    end_epoch_accuracy, end_epoch_accuracy_k = [], []
                    end_epoch_loss = []

                    for batch, batch_weights, batch_labels in \
                            batch_generator(val_descs2, val_labels2, batch_size_inference, train_vocab, labels_lookup,
                                            word_ngrams, sort_ngrams, show_progress=False):
                        correct, correct_k, batch_loss = sess.run(
                            [correctly_predicted, correctly_predicted_top_k, ce_loss],
                            feed_dict={input_ph: batch,
                                       weights_ph: batch_weights,
                                       labels_ph: batch_labels})

                        end_epoch_accuracy.extend(correct)
                        end_epoch_accuracy_k.extend(correct_k)
                        end_epoch_loss.append(batch_loss)
                        summary_loss = tf.Summary(
                            value=[tf.Summary.Value(tag="batch_loss", simple_value=np.mean(batch_loss))])
                        val_end_batch_writer.add_summary(summary_loss, batch_counter)
                        batch_counter += 1
                    mean_acc = np.round(100 * np.sum(end_epoch_accuracy) / initial_val_len, 2)
                    mean_acc_k = np.round(100 * np.sum(end_epoch_accuracy_k) / initial_val_len, 2)
                    summary_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=mean_acc)])
                    summary_acc_k = tf.Summary(value=[tf.Summary.Value(tag="accuracy_top_{}".format(top_k),
                                                                       simple_value=mean_acc_k)])
                    summary_loss = tf.Summary(
                        value=[tf.Summary.Value(tag="loss", simple_value=np.mean(end_epoch_loss))])

                    val_end_writer.add_summary(summary_acc, epoch + 1)
                    val_end_writer.add_summary(summary_acc_k, epoch + 1)
                    val_end_writer.add_summary(summary_loss, epoch + 1)
                    print("end epoch mean val accuracy: {}, top {}: {}".format(mean_acc, top_k, mean_acc_k))

                if save_all_models:
                    freeze_save_graph(sess, result_dir, "model_ep{}.pb".format(epoch + 1), "prediction")
                else:
                    if epoch + 1 == n_epochs:
                        freeze_save_graph(sess, result_dir, "model_ep{}.pb".format(epoch + 1), "prediction")
                        print("the model is stored at {}".format(result_dir))
                        if use_validation:
                            results = {"hyperparams": train_specific,
                                       "scores": {validation_path: {top_k: mean_acc_k, 1: mean_acc}}}
                            already_trained[hypers_hashed] = results
                            with open(os.path.join(result_dir, "results.json"), "w+") as outfile:
                                json.dump(results, outfile)
                            with open(os.path.join(cache_dir, "details.json"), "w+") as outfile:
                                json.dump(data_specific, outfile)
                            with open(hyperparams_path, "w+") as outfile:
                                json.dump(already_trained, outfile)

            print("the training took {} seconds".format(round(time.time() - train_start, 0)))
    print("peak memory usage: {}".format(round(tracemalloc.get_traced_memory()[1] / 1e6, 0)))
    print("all process took {} seconds".format(round(time.time() - main_start, 0)))


if __name__ == "__main__":
    main()
