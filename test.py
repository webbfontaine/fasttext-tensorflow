import os
import json
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1.logging as logging
import tensorflow as tf

from fasttext_utils import (
    parse_txt,
    next_batch,
    get_all,
)
from utils import load_graph

logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="path where model.pb and model_params.json are")
    parser.add_argument("--test_path", type=str, help="path to test file")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for inference")
    parser.add_argument("--k", type=int, default=1, help="calculate accuracy on top k predictions")
    parser.add_argument("--hand_check", type=bool, default=False, help="test on manually inputted data")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu for inference")
    parser.add_argument("--gpu_fraction", type=float, default=0.4, help="what fraction of gpu to allocate")
    args = parser.parse_args()

    model_dir = args.model_dir
    model_params_path = os.path.join(model_dir, "model_params.json")
    model_path = os.path.join(model_dir, "model_best.pb")
    test_path = args.test_path
    batch_size = args.batch_size
    hand_check = args.hand_check
    k = args.k
    use_gpu = args.use_gpu
    gpu_fraction = args.gpu_fraction

    if use_gpu:
        device = "/gpu:0"
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                                          allow_growth=True))
    else:
        device = "/cpu:0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.ConfigProto(allow_soft_placement=True)

    num_thrown_for_label = 0
    with open(model_params_path, "r") as infile:
        model_params = json.load(infile)
    if os.path.isfile(model_params["label_dict_path"]):
        with open(model_params["label_dict_path"], "r") as infile:
            label_vocab = json.load(infile)
    else:
        with open(os.path.join(model_dir, "label_dict.json"), "r") as infile:
            label_vocab = json.load(infile)
    if os.path.isfile(model_params["word_dict_path"]):
        with open(model_params["word_dict_path"], "r") as infile:
            word_id = json.load(infile)
    else:
        with open(os.path.join(model_dir, "word_id.json"), "r") as infile:
            word_id = json.load(infile)
    word_ngrams = model_params["word_ngrams"]
    sort_ngrams = model_params["sort_ngrams"]

    labels_vocab_inverse = {}

    for label, label_id in label_vocab.items():
        labels_vocab_inverse[label_vocab[label]["id"]] = label

    with tf.device(device):
        with tf.Session(config=config) as sess:
            run_arg = load_graph(model_path, ["input:0", "input_weights:0", "prediction:0"])
            if hand_check:
                while True:
                    query_desc = input("Enter the description: ")
                    label = query_desc[9:19]
                    query_desc = query_desc[20:]
                    test_desc_inds = np.expand_dims([0] + [word_id[phrase]["id"] for phrase in
                                                           get_all(query_desc.split(), word_ngrams, sort_ngrams) if
                                                           phrase in word_id], axis=0)

                    test_desc_weights = np.zeros_like(test_desc_inds, dtype=np.float32)
                    test_desc_weights[0][:len(test_desc_inds[0])] = 1. / len(test_desc_inds[0])

                    if label not in label_vocab:
                        print("new label")
                        continue

                    probs = np.squeeze(sess.run(run_arg[-1], feed_dict={run_arg[0]: test_desc_inds,
                                                                        run_arg[1]: test_desc_weights}))

                    max_ind = np.argmax(probs)
                    max_prob = probs[max_ind]
                    pred_label = labels_vocab_inverse[max_ind]
                    print(pred_label == label, pred_label, max_prob)
            else:
                test_descriptions, test_labels = parse_txt(test_path, join_desc=True)
                test_inds = np.arange(len(test_descriptions))
                print("The total number of test datapoints: {}".format(len(test_descriptions)))

                pbar = tqdm(total=int(np.ceil(len(test_descriptions) / batch_size)))
                rem_inds, batch_inds = next_batch(test_inds, batch_size)
                accuracy_top_1, accuracy_top_k = 0, 0
                cnt = 0

                while len(batch_inds) > 0:
                    batch_descriptions = [test_descriptions[i] for i in batch_inds]
                    batch_labels = [test_labels[i] for i in batch_inds]

                    batch, batch_weights, batch_labels2 = [], [], []

                    max_words = -1
                    for test_desc in batch_descriptions:
                        max_words = max(max_words, len(test_desc.split()))

                    num_max_words = 1
                    for ng in range(word_ngrams):
                        num_max_words += max_words - ng

                    for test_desc, test_label in zip(batch_descriptions, batch_labels):
                        if test_label not in label_vocab:
                            num_thrown_for_label += 1
                            continue
                        init_test_inds = [0] + [word_id[phrase]["id"] for phrase in
                                                get_all(test_desc.split(), word_ngrams, sort_ngrams)
                                                if phrase in word_id]

                        cnt += 1
                        test_desc_inds = np.array(init_test_inds + [0 for _ in
                                                                    range(num_max_words - len(init_test_inds))])
                        test_desc_weights = np.zeros_like(test_desc_inds, dtype=np.float32)
                        test_desc_weights[:len(init_test_inds)] = 1. / len(init_test_inds)

                        batch.append(test_desc_inds)
                        batch_weights.append(test_desc_weights)
                        batch_labels2.append(label_vocab[test_label]["id"])

                    probs = sess.run(run_arg[-1], feed_dict={run_arg[0]: batch,
                                                             run_arg[1]: batch_weights})
                    top_k = [np.argsort(i)[-k:] for i in probs]

                    accuracy_top_k += sum([True if i in j else False for i, j in zip(batch_labels2, top_k)])
                    accuracy_top_1 += sum([True if i == j[-1] else False for i, j in zip(batch_labels2, top_k)])
                    rem_inds, batch_inds = next_batch(rem_inds, batch_size)
                    pbar.update()
                pbar.close()

                print("{} datapoint thrown because of label".format(num_thrown_for_label))
                print("Number of test datapoints after cleaning: {}".format(len(test_descriptions) -
                                                                            num_thrown_for_label))
                print("Number of unique labels in test after cleaning: {}".format(len(set(test_labels))))
                print("Accuracy: {}".format(round(100 * accuracy_top_1 / len(test_descriptions), 2)))
                print("Accuracy top {}: {}".format(k, round(100 * accuracy_top_k / len(test_descriptions), 2)))


if __name__ == "__main__":
    main()
