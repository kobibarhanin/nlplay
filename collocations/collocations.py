import os
import sys
import random

# ============================= Constants ================================

# IMPORTANT - this solutions assumes that these characters are cleaned from the corpus.
# If not - change them to two different special characters, excluding: [?!()]
line_begin = '#'
line_end = '%'

LAMBDA_UNI = 0.5
LAMBDA_BI = 0.5


# ========================== END Constants ===============================
# ========================= Utility Functions ============================


def add_or_init(dictionary, key, value):
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value


def random_by_weight(dct):
    rand_point = random.random()
    pointer = 0
    for k, v in dct.items():
        pointer += v
        if rand_point <= pointer:
            return k


# ======================= END Utility Functions ==========================
# ====================== Build Corpus Functions ==========================


def build_corpus(directory, build_method, file=None):
    corpus_dict = dict()
    count_total = 0
    if file:
        return read_corpus_file(os.path.join(directory, file), build_method)
    for filename in os.listdir(directory):
        _tokens, _count = read_corpus_file(os.path.join(directory, filename), build_method)
        merge_corpuses(corpus_dict, _tokens)
        count_total += _count
    return corpus_dict, count_total


def read_corpus_file(corpus_file, build_method):
    corpus_dict = dict()
    tokens_count = 0
    with open(corpus_file, 'r') as raw_data:
        for line in raw_data:
            tokens = line.replace('\n', ' '+line_end).split(' ')
            tokens_count += build_method(corpus_dict, tokens)
    return corpus_dict, tokens_count


def merge_corpuses(main_corpus, added_corpus):
    for token, count in added_corpus.items():
        add_or_init(main_corpus, token, count)


def populate_unigram_corpus(_corpus, tokens):
    for token in tokens:
        add_or_init(_corpus, token, 1)
    return len(tokens)


def populate_bigram_corpus(_corpus, tokens):
    _total = 0
    tokens.insert(0, line_begin)
    for first, second in zip(tokens, tokens[1:]):
        if first != line_begin:
            _total += 1
            bigram = first+','+second
            add_or_init(_corpus, bigram, 1)
    return _total


def populate_trigram_corpus(_corpus, tokens):
    _total = 0
    tokens.insert(0, line_begin)
    for first, second, third in zip(tokens, tokens[1:], tokens[2:]):
        if first != line_begin:
            _total += 1
            trigram = first+','+second+','+third
            add_or_init(_corpus, trigram, 1)
    return _total


# ==================== END Build Corpus Functions ========================
# ======================= Evaluation Functions ===========================


def calculate_mle(_corpus, total_corpus):
    for token, count in _corpus.items():
        _corpus[token] = count/total_corpus


def calculate_mle_laplace(_corpus, total_corpus):
    for token, count in _corpus.items():
        _corpus[token] = (count + 1) / (total_corpus + len(_corpus))


def evaluate_unigram_token(_corpus, token, total_tokens):
    return _corpus[token] if token in _corpus else 1 / (total_tokens + len(_corpus))


def evaluate_bigram_token(bi_corpus, bigram, uni_corpus, total_tokens, factor=1.0):
    return bi_corpus[bigram] \
        if bigram in bi_corpus \
        else factor * evaluate_unigram_token(uni_corpus, bigram.split(',')[1], total_tokens)


def evaluate_sentence_unigrams(_corpus, total_tokens, _sentence):
    tokens = _sentence.split(' ')
    p = 1
    for token in tokens:
        p *= evaluate_unigram_token(_corpus, token, total_tokens)
    return p


def evaluate_sentence_bigrams(bi_corpus, sentence, uni_corpus, total_tokens):
    p = 1
    data = sentence.split(' ')
    for first, second in zip(data, data[1:]):
        bigram = first+','+second
        p *= evaluate_bigram_token(bi_corpus, bigram, uni_corpus, total_tokens)
    return p


def evaluate_sentence_trigrams(tri_corpus, sentence, bi_corpus, uni_corpus, total_tokens):
    p = 1
    data = sentence.split(' ')
    if len(data) < 3:
        return evaluate_sentence_bigrams(bi_corpus, sentence, uni_corpus, total_tokens)
    for first, second, third in zip(data, data[1:], data[2:]):
        trigram = first+','+second+','+third
        if trigram in tri_corpus:
            p *= tri_corpus[trigram]
        else:
            bigram = first+','+second
            p *= LAMBDA_BI * evaluate_bigram_token(bi_corpus, bigram, uni_corpus, total_tokens, LAMBDA_UNI)
    return p


# ===================== END Evaluation Functions =========================
# ======================= Generation Functions ===========================


def generate_random_sentence(_corpus):
    _sentence = ''
    while True:
        token = random_by_weight(_corpus)
        if token is None:
            continue
        if ',' in token:
            token = token.split(',')
        else:
            if token == line_end:
                break
            else:
                _sentence += ' ' + token
                continue
        if token[-1] == line_end:
            _sentence += ' ' + ' '.join([tok for tok in token[0:-1]])
            break
        else:
            _sentence += ' ' + ' '.join([tok for tok in token])
    return _sentence


# ===================== END Generation Functions =========================


def run(corpus_path, output_file):
    with open(output_file, 'w') as output:
        output.write("Unigrams model based on complete dataset:\n")
        corpus, total = build_corpus(corpus_path, populate_unigram_corpus)
        calculate_mle_laplace(corpus, total)
        for _ in range(3):
            output.write(generate_random_sentence(corpus) + "\n")

        output.write("\nBigrams model based on complete dataset:\n")
        bigram_corpus, bigrams_total = build_corpus(corpus_path, populate_bigram_corpus)
        calculate_mle(bigram_corpus, bigrams_total)
        for _ in range(3):
            output.write(generate_random_sentence(bigram_corpus) + "\n")

        output.write("\nTrigrams model based on complete dataset:\n")
        trigram_corpus, trigrams_total = build_corpus(corpus_path, populate_trigram_corpus)
        calculate_mle(trigram_corpus, trigrams_total)
        for _ in range(3):
            output.write(generate_random_sentence(trigram_corpus) + "\n")

        for filename in os.listdir(corpus_path)[0:2]:
            output.write("\nBigrams model based on text written by users from " + str(filename) + ":\n")
            bigram_corpus, bigrams_total = build_corpus(corpus_path, populate_bigram_corpus, filename)
            calculate_mle(bigram_corpus, bigrams_total)
            for _ in range(3):
                output.write(generate_random_sentence(bigram_corpus) + "\n")

            output.write("\nTrigrams model based on text written by users from " + str(filename) + ":\n")
            trigram_corpus, trigrams_total = build_corpus(corpus_path, populate_trigram_corpus, filename)
            calculate_mle(trigram_corpus, trigrams_total)
            for _ in range(3):
                output.write(generate_random_sentence(trigram_corpus) + "\n")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_directory = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_directory = 'corpus'
        output_file = 'output.txt'

    run(input_directory, output_file)
