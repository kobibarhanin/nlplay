from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from itertools import compress


import os
import sys
import string


DEV_MODE = False
# DEV_MODE = True


le = preprocessing.LabelEncoder()


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


PUNCS = "?!(),:'@"


rename = {
    'MultinomialNB': 'Naïve Bayes',
    'LogisticRegression': 'Logistic Regression',
}


results = {
    'bag_of_words': {
        'author': {},
        'language': {}
    },
    'my_features': {
        'author': {},
        'language': {}
    },
    'best_features': {
        'author': {},
        'language': {}
    }
}


def check_num(s):
    if len(s) < 1:
        return False
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def check_repeat(token):
    for i, char in enumerate(token):
        try:
            if char == token[i - 1]:
                return True
        except Exception as e:
            pass
    return False


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


def kbest(_data, _labels, k):
    selector = SelectKBest(chi2, k=k)
    fit = selector.fit_transform(_data, _labels)
    kbest_indices = selector.get_support(indices=False)
    return fit, kbest_indices


def fit_vectorize(_data, _labels):
    cv = CountVectorizer()
    _counts = cv.fit_transform(_data)
    counts_tf = TfidfTransformer(use_idf=False).fit_transform(_counts)
    return counts_tf, cv


def fit_features(_data, _labels):
    _counts = []
    for i, data_point in enumerate(_data):
        tokens = data_point.split(' ')
        features = [
            len(tokens),
            len(data_point),
            sum(1 for char in data_point if char in string.punctuation),
            sum(1 for token in tokens if token in STOP_WORDS),
            sum(1 for token in tokens if check_num(token)),
            sum(1 for token in tokens if check_repeat(token)),
        ]
        for punct in PUNCS:
            features.append(sum(1 for char in data_point if char == punct))
        for stop in STOP_WORDS:
            features.append(sum(1 for token in tokens if token == stop))
        _counts.append(features)
    return _counts, None


def test_model(_algorithm, _algorithm_params, _counts, _labels):
    if _algorithm_params is None:
        algorithm = _algorithm()
    else:
        algorithm = _algorithm(**_algorithm_params)
    scores = cross_val_score(algorithm, _counts, _labels, cv=10)
    print(f'cross_val_scores = {scores}')
    return round(scores.mean(), 4) * 100


def produce_kbest(corpus_path, class_unit_size, output_file, best_factor=100):
    total_data, total_labels = prepare_data(corpus_path,
                                            class_unit_size)
    counts, cv = fit_vectorize(total_data,
                               total_labels)
    counts, kbest_indices = kbest(counts,
                                  total_labels,
                                  best_factor)
    write_kbest(cv, kbest_indices, output_file)
    return counts


def write_kbest(cv, kbest_indices, output):
    with open(output, 'w') as output_file:
        for word in list(compress(cv.get_feature_names(), kbest_indices)):
            output_file.write(word + '\n')


def run(fit_method, corpus_path, class_unit_size, _kbest=None, _results=None):
    print(f'running with: '
          f'fit_method={fit_method}, '
          f'corpus_path={corpus_path}, '
          f'class_unit_size={class_unit_size}')
    total_data, total_labels = prepare_data(corpus_path,
                                            class_unit_size)
    counts, cv = fit_method(total_data,
                            total_labels)
    counts = counts if _kbest is None else _kbest
    _nb = test_model(MultinomialNB,
                     None,
                     counts,
                     total_labels)
    _lr = test_model(LogisticRegression,
                     {'solver': 'liblinear', 'multi_class': 'ovr', 'max_iter': 1000},
                     counts,
                     total_labels)
    _results['nb'] = _nb
    _results['lr'] = _lr
    print(f'MultinomialNB accuracy: {_nb}')
    print(f'LogisticRegression accuracy: {_lr}')
    print(f'=====================================')
    return _nb, _lr


def write_results(_results, output_file):
    with open(output_file, 'w') as output:
        output.write('Phase1 (Bag of Words):\n')
        output.write('Author Identification:\n')
        output.write(f"Naïve Bayes: {_results['bag_of_words']['author']['nb']}\n")
        output.write(f"Logistic Regression: {_results['bag_of_words']['author']['lr']}\n")
        output.write('Native Language Identification::\n')
        output.write(f"Naïve Bayes: {_results['bag_of_words']['language']['nb']}\n")
        output.write(f"Logistic Regression: {_results['bag_of_words']['language']['lr']}\n")
        output.write('-----------------------------------------\n')
        output.write('Phase2 (My features):\n')
        output.write('Author Identification:\n')
        output.write(f"Naïve Bayes: {_results['my_features']['author']['nb']}\n")
        output.write(f"Logistic Regression: {_results['my_features']['author']['lr']}\n")
        output.write('Native Language Identification::\n')
        output.write(f"Naïve Bayes: {_results['my_features']['language']['nb']}\n")
        output.write(f"Logistic Regression: {_results['my_features']['language']['lr']}\n")
        output.write('-----------------------------------------\n')
        output.write('Phase2 (Best Features):\n')
        output.write('Author Identification:\n')
        output.write(f"Naïve Bayes: {_results['best_features']['author']['nb']}\n")
        output.write(f"Logistic Regression: {_results['best_features']['author']['lr']}\n")
        output.write('Native Language Identification::\n')
        output.write(f"Naïve Bayes: {_results['best_features']['language']['nb']}\n")
        output.write(f"Logistic Regression: {_results['best_features']['language']['lr']}\n")


if __name__ == '__main__':
    if len(sys.argv) == 6:
        input_dir_1 = sys.argv[1]
        input_dir_2 = sys.argv[2]
        output_path = sys.argv[3]
        best_words_path_1 = sys.argv[4]
        best_words_path_2 = sys.argv[5]
    else:
        input_dir_1 = 'inputs/user'
        input_dir_2 = 'inputs/lang'
        output_path = 'outputs/results_kbest.txt'
        best_words_path_1 = 'outputs/kbest_user.txt'
        best_words_path_2 = 'outputs/kbest_lang.txt'

    kbest_counts_1 = produce_kbest(input_dir_1, 1, best_words_path_1, best_factor=5000)
    kbest_counts_2 = produce_kbest(input_dir_2, 20, best_words_path_2, best_factor=5000)

    run(fit_vectorize, input_dir_1, 1, _results=results['bag_of_words']['author'])
    run(fit_vectorize, input_dir_2, 20, _results=results['bag_of_words']['language'])
    run(fit_features, input_dir_1, 1, _results=results['my_features']['author'])
    run(fit_features, input_dir_2, 20, _results=results['my_features']['language'])
    run(fit_vectorize, input_dir_1, 1, _kbest=kbest_counts_1, _results=results['best_features']['author'])
    run(fit_vectorize, input_dir_2, 20, _kbest=kbest_counts_2, _results=results['best_features']['language'])

    write_results(results, output_path)

    print(f'run results: {results}')
