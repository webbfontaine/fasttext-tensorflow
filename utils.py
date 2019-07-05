import os
import hashlib
from tensorflow.python import graph_util
import tensorflow as tf
from shutil import copy
import numpy as np


def validate(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def freeze_save_graph(sess, log_dir, name, output_node):
    for node in sess.graph.as_graph_def().node:
        node.device = ""

    variable_graph_def = sess.graph.as_graph_def()
    optimized_net = graph_util.convert_variables_to_constants(sess, variable_graph_def, [output_node])
    tf.train.write_graph(optimized_net, log_dir, name, False)


def load_graph(graph_path, return_elements=[]):
    with tf.gfile.GFile(graph_path, 'rb') as infile:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(infile.read())
        output_nodes = tf.import_graph_def(graph_def, return_elements=return_elements)
        return output_nodes


def hash_(st):
    return hashlib.md5(st.encode("utf-8")).hexdigest()


def hash2_(st):
    return int.from_bytes(hashlib.md5(st.encode("utf-8")).digest(), "little")


def hash_xor_data(list_of_texts):
    init_hash = hash2_(str(list_of_texts[0]))
    for text in list_of_texts[1:]:
        init_hash = init_hash ^ hash2_(str(text))
    return str(init_hash)


def hash_function(f):
    if f is None:
        return "no_prep"
    return "{}_{}".format(f.__code__.co_name, hash_(f.__code__.co_code.decode('utf-16')))


def get_cache_hash(list_of_texts, data_specific_params):
    data_hashed = "".join([str(v) for k, v in sorted(data_specific_params.items(), key=lambda t: t[0])])
    hash_xor = hash_xor_data(list_of_texts=list_of_texts)
    return hash_(data_hashed + hash_xor)


def handle_space_paths(path):
    return '"{}"'.format(path)


def copy_all(list_of_paths, destination_path):
    for src_path in list_of_paths:
        if os.path.isfile(src_path):
            copy(src_path, os.path.join(destination_path, os.path.basename(src_path)))
        else:
            print("invalid path, no such file {}".format(src_path))


def percent(x, multiplier=100, precision=2):
    return np.round(multiplier * np.mean(x), precision)


def dummy_print(x):
    print('\n' + '*' * 20)
    print(x)
    print('*' * 20 + '\n')


def split_list(n_items, n, seed=None):
    if seed:
        np.random.seed(seed)
    indices = np.arange(n_items)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n)
    for fold in folds:
        fold_mask = np.zeros(n_items)
        fold_mask[fold] = 1
        yield fold_mask.astype(bool)
