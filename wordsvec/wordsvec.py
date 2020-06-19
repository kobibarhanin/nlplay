import os
import sys
import time

import numpy as np
from gensim.models import KeyedVectors

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold


DEV_MODE = False
le = preprocessing.LabelEncoder()


results = {
    'arithmetic_mean': {
        'pre_trained_word2vec': {},
        'my_word2vec': {}
    },
    'random_weights': {
        'pre_trained_word2vec': {},
        'my_word2vec': {}
    },
    'custom_weights_nostopwords': {
        'pre_trained_word2vec': {},
        'my_word2vec': {}
    },
}


def parse_data(_data, _label, class_unit_size):
    _max = 1000 if DEV_MODE else None
    _items = _data.split('\n')[:_max]
    composite_items = [" ".join(_items[x:x + class_unit_size]) for x in range(0, len(_items), class_unit_size)]
    _labels = [_label] * len(composite_items)
    return composite_items, _labels


def prepare_data(corpus_path, class_unit_size):
    input_source = [filename for filename in os.listdir(corpus_path)]
    labels = le.fit_transform(input_source)
    inputs = list(le.classes_)
    total_data = []
    total_labels = []
    for file, label in zip(inputs, labels):
        with open(f'{corpus_path}/{file}', 'r') as data_file:
            data, labels = parse_data(data_file.read(), label, class_unit_size)
            total_data += data
            total_labels += labels
    return total_data, total_labels


def run(_model_pre_trained, _model_my_word2vec, corpus_path, class_unit_size, chunk_to_vec_method, _results):
    total_data, total_labels = prepare_data(corpus_path,
                                            class_unit_size)
    _results['pre_trained_word2vec'] = run_with_model(_model_pre_trained, total_data, total_labels, chunk_to_vec_method)
    _results['my_word2vec'] = run_with_model(_model_my_word2vec, total_data, total_labels, chunk_to_vec_method)


def run_with_model(_model, total_data, total_labels, chunk_to_vec_method):
    counts = word_to_vec(_model,
                         total_data,
                         total_labels,
                         chunk_to_vec_method)
    scores = test_model(LogisticRegression,
                        {'solver': 'liblinear', 'multi_class': 'ovr', 'max_iter': 1000},
                        counts,
                        total_labels)
    return scores


def test_model(_algorithm, _algorithm_params, _counts, _labels):
    if _algorithm_params is None:
        algorithm = _algorithm()
    else:
        algorithm = _algorithm(**_algorithm_params)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(_counts, _labels)
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(algorithm, _counts, _labels, scoring=scoring, cv=skf)
    _results = {}
    for key, value in scores.items():
        _results[key] = round(value.mean(), 4) * 100
    return _results


def word_to_vec(_model, _total_data, _total_labels, chunk_to_vec_method):
    counts = []
    for item in _total_data:
        counts.append(chunk_to_vec_method(item, _model))
    return counts


def chunk_to_vec_by_average(item, model):
    tokens = item.split(" ")
    chunk_vec = np.zeros(300)
    tokens_used = 1
    for token in tokens:
        try:
            word_vec = model.get_vector(token.lower())
            chunk_vec = chunk_vec + word_vec
            tokens_used += 1
        except Exception:
            # This handles the case of token not existing in model
            pass
    return chunk_vec / tokens_used


def chunk_to_vec_by_random(item, model):
    tokens = item.split(" ")
    chunk_vec = np.zeros(300)
    vectors = []
    for token in tokens:
        try:
            vectors.append(model.get_vector(token.lower()))
        except Exception:
            # This handles the case of token not existing in model
            pass
    if len(vectors) < 2:
        print("No vectors in chunk")
        return chunk_vec
    rand_coeff = np.random.dirichlet(np.ones(len(vectors)), size=1)[0]
    for i, vec in enumerate(vectors):
        chunk_vec = chunk_vec + rand_coeff[i] * vec
    return chunk_vec


def chunk_to_vec_linear(item, model):
    tokens = item.split(" ")
    chunk_vec = np.zeros(300)
    vectors = []
    for token in tokens:
        try:
            vectors.append(model.get_vector(token.lower()))
        except Exception:
            # This handles the case of token not existing in model
            pass
    if len(vectors) < 2:
        print("No vectors in chunk")
        return chunk_vec
    array = np.arange(len(vectors))[::-1]
    rand_coeff = (array / (array.mean())) / len(array)
    for i, vec in enumerate(vectors):
        chunk_vec = chunk_vec + rand_coeff[i] * vec
    return chunk_vec


STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


def chunk_to_vec_nostopwords(item, model):
    tokens = item.split(" ")
    chunk_vec = np.zeros(300)
    tokens_used = 1
    for token in tokens:
        try:
            if token in STOP_WORDS:
                continue
            word_vec = model.get_vector(token.lower())
            chunk_vec = chunk_vec + word_vec
            tokens_used += 1
        except Exception:
            # This handles the case of token not existing in model
            pass
    return chunk_vec / tokens_used


def write_results(_results, output_file):
    with open(output_file, 'w') as output:
        output.write('Arithmetic mean:\n')
        output.write('Pre-trained word2vec model performance:\n')
        output.write(f"Accuracy: {_results['arithmetic_mean']['pre_trained_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['arithmetic_mean']['pre_trained_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['arithmetic_mean']['pre_trained_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['arithmetic_mean']['pre_trained_word2vec']['test_f1_macro']}\n")
        output.write('My word2vec model performance:\n')
        output.write(f"Accuracy: {_results['arithmetic_mean']['my_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['arithmetic_mean']['my_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['arithmetic_mean']['my_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['arithmetic_mean']['my_word2vec']['test_f1_macro']}\n")
        output.write('-----------------------------------------\n')
        output.write('Random weights:\n')
        output.write('Pre-trained word2vec model performance:\n')
        output.write(f"Accuracy: {_results['random_weights']['pre_trained_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['random_weights']['pre_trained_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['random_weights']['pre_trained_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['random_weights']['pre_trained_word2vec']['test_f1_macro']}\n")
        output.write('My word2vec model performance:\n')
        output.write(f"Accuracy: {_results['random_weights']['my_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['random_weights']['my_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['random_weights']['my_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['random_weights']['my_word2vec']['test_f1_macro']}\n")
        output.write('-----------------------------------------\n')
        output.write('My weights:\n')
        output.write('Pre-trained word2vec model performance:\n')
        output.write(f"Accuracy: {_results['custom_weights_nostopwords']['pre_trained_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['custom_weights_nostopwords']['pre_trained_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['custom_weights_nostopwords']['pre_trained_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['custom_weights_nostopwords']['pre_trained_word2vec']['test_f1_macro']}\n")
        output.write('My word2vec model performance:\n')
        output.write(f"Accuracy: {_results['custom_weights_nostopwords']['my_word2vec']['test_accuracy']}\n")
        output.write(f"Precision: {_results['custom_weights_nostopwords']['my_word2vec']['test_precision_macro']}\n")
        output.write(f"Recall: {_results['custom_weights_nostopwords']['my_word2vec']['test_recall_macro']}\n")
        output.write(f"F1: {_results['custom_weights_nostopwords']['my_word2vec']['test_f1_macro']}\n")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        input_dir = sys.argv[1]
        pre_trained_word2vec = sys.argv[2]
        my_word2vec = sys.argv[3]
        output_path = sys.argv[4]
    else:
        input_dir = 'inputs/tokenized_corpus/tokenized_unbalanced'
        pre_trained_word2vec = 'models/wiki.en.100k.vec'
        my_word2vec = 'models/model.vec'
        output_path = 'outputs/results.txt'

    # chunk_size = 5
    chunk_size = 20

    start_time = time.time()

    model_pre_trained_word2vec = KeyedVectors.load_word2vec_format(pre_trained_word2vec, binary=False)
    model_my_word2vec = KeyedVectors.load_word2vec_format(my_word2vec, binary=False)

    print("Models loaded: --- %s seconds ---" % (time.time() - start_time))

    try:
        run(model_pre_trained_word2vec,
            model_my_word2vec,
            input_dir,
            chunk_size,
            chunk_to_vec_by_average,
            results['arithmetic_mean'])
    except Exception as e:
        print(f'encountered exception: {e}')

    print("Post arithmetic_mean: --- %s seconds ---" % (time.time() - start_time))

    try:
        run(model_pre_trained_word2vec,
            model_my_word2vec,
            input_dir,
            chunk_size,
            chunk_to_vec_by_random,
            results['random_weights'])
    except Exception as e:
        print(f'encountered exception: {e}')

    print("Post random_weights: --- %s seconds ---" % (time.time() - start_time))

    try:
        run(model_pre_trained_word2vec,
            model_my_word2vec,
            input_dir,
            chunk_size,
            chunk_to_vec_nostopwords,
            results['custom_weights_nostopwords'])
    except Exception as e:
        print(f'encountered exception: {e}')

    print("Post custom_weights_nostopwords: --- %s seconds ---" % (time.time() - start_time))

    from pprint import pprint
    print(f'run results:')
    pprint(results)

    write_results(results, output_path)
