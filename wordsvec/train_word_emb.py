import sys
import os

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

sys.path.append('corpus/corpus.py')
from corpus.corpus import generate_corpus


def similarity(model, a, b):
    sim = model.similarity(a, b)
    print(f'similarity between {a} and {b}: {sim}')
    return sim


def load_model(model_path):
    return KeyedVectors.load_word2vec_format(model_path, binary=False)


def generate_model(corpus_path, model_path, save_to_file=False):
    with open(corpus_path, 'r') as f:
        corpus = f.readlines()
    corpus_tokenized = [token.strip().split(' ') for token in corpus]
    _model = Word2Vec(corpus_tokenized, size=300, min_count=10)
    if save_to_file:
        _model.wv.save_word2vec_format(model_path)
    return _model


def test_model(_model):

    similarity(_model, 'good', 'well')
    similarity(_model, 'pretty', 'beautiful')
    similarity(_model, 'dog', 'cat')
    similarity(_model, 'computer', 'ball')
    similarity(_model, 'sky', 'clouds')

    boy = _model.most_similar('boy')
    man = _model.most_similar('man')
    woman = _model.most_similar('woman')

    print(f'boy = {boy}')
    print(f'man = {man}')
    print(f'woman = {woman}')


if __name__ == "__main__":

    if len(sys.argv) == 2:
        input_dir = sys.argv[1]
    else:
        input_dir = 'inputs/raw_corpus'

    curren_dir = os.getcwd()

    GENERATE_CORPUS = True
    GENERATE_MODEL = True

    corpus_file = f'total_corpus'
    custom_model_file = 'model.vec'
    pretrained_model_file = 'wiki.en.100k.vec'

    if GENERATE_CORPUS:
        print('--- Generating Corpus ---')
        generate_corpus(input_dir, curren_dir)
        print('--- Done! ---')
    if GENERATE_MODEL:
        print('--- Generating Model ---')
        custom_model = generate_model(corpus_file, custom_model_file, save_to_file=True)
        print('--- Done! ---')
    else:
        custom_model = load_model(custom_model_file)

    pretrained_model = load_model(pretrained_model_file)

    print(f'pretrained_model results:')
    test_model(pretrained_model)
    print(f'custom_model results:')
    test_model(custom_model)
