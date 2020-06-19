import re
import string
import os
import time
import sys
import shutil


# ========================= Utility Functions ============================


def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def pprint(printable):
    print('====================')
    for line in printable:
        print(line)
    print('====================')


def build_regex(regex_list):
    rv = r'('
    for regex in regex_list:
        rv += regex
        rv += '|'

    rv = rv[:-1]
    rv += ')'
    return rv


# ======================= END Utility Functions ==========================

# ========================= Cleaning Functions ===========================


def clean_complex(raw_data, filters):
    for i in range(0, len(raw_data)):
        for text_filter in filters.values():
            raw_data[i] = re.sub(text_filter[0], text_filter[1], raw_data[i])
    return raw_data


def remove_punctuation(text, punctuation_allowed):
    return "".join([char for char in text if char not in re.sub(punctuation_allowed, '', string.punctuation)])


def clean(raw_data, filters_params):
    filters = filters_params['filters']
    filters_order = filters_params['order']

    for i in range(0, len(raw_data)):
        for filter in filters_order:
            filter_engine = filters[filter]
            cleaned = re.sub(filter_engine, '', raw_data[i])
            raw_data[i] = cleaned
    return raw_data


def replace(raw_data, substitution_map):
    for i in range(0, len(raw_data)):
        for initial, final in substitution_map.items():
            raw_data[i] = re.sub(initial, final, raw_data[i])
    return raw_data


def split_to_tokens(data, regex):
    return re.split(regex, data)


def remove_empty_sentences(data, params):
    return list(filter(params, data))


def is_date(date):
    pattern = '[0-9]{4}-[0-9]{2}-[0-9]{2}'
    return False if re.match(pattern, date) is None else True


# ======================= END Cleaning Functions =========================

# ========================= Execution Functions ==========================


@timing
def run(input_directory, lead_factor, output_directory, cleaning_pipeline, tokenization_pipeline):
    total_sentences = 0
    for filename in os.listdir(input_directory):
        print('running for file: ' + filename +
              '\n=================================')
        total = ingest(input_directory + '/' + filename,
                       lead_factor,
                       output_directory,
                       cleaning_pipeline,
                       tokenization_pipeline)
        total_sentences += total
        print('total sentences: ' + str(total) +
              '\n=================================')

    print('total sentences in run: ' + str(total_sentences) +
          '\n=================================')


@timing
def ingest(target_file, lead_factor, output_directory, cleaning_pipeline, tokenization_pipeline):
    country = target_file.split('.')[1]

    # read file:
    with open(target_file, 'r') as raw_data:
        raw_data = raw_data.read().split('\n')
        raw_posts = cluster_lines_to_posts(raw_data)

    map_users_scores = dict()
    map_users_parsed = dict()

    print(f'file {target_file}, writers: {len(raw_posts)}')

    # parse sentences
    for user, data in raw_posts.items():
        for stage in cleaning_pipeline:
            data = stage['function'](data, stage['param'])

        map_users_scores[user] = len(data)
        map_users_parsed[user] = data

    # sort to get leaders
    leaders = sorted(map_users_scores.items(), key=lambda item: item[1], reverse=True)[0:lead_factor]

    # write output
    total_sentences = 0
    for user, value in leaders:
        print('user:' + user + ' , sentences: ' + str(value))
        tokenize(map_users_parsed[user], user, country, output_directory, tokenization_pipeline)
        total_sentences += value
    return total_sentences


def tokenize(sentences_parsed, _user, _country, output_directory, tokenization_pipeline):
    with open(output_directory + '/' + _user + '_' + _country, 'w') as f:
        for data in sentences_parsed:
            for stage in tokenization_pipeline:
                data = stage['function'](data, stage['param'])
            if len(data) != 0:
                f.write(" ".join(data) + '\n')


def cluster_lines_to_posts(data):
    posts = {}
    for line in data:
        try:
            new_post = True
            user, forum, date, post = line.split(',', 3)

            # test date by regex YYYY-MM-DD
            if is_date(date):
                last_defined_user = user
            else:
                new_post = False
                user = last_defined_user

        except Exception as e:
            # in case of newline in post, csv unpacking fails, concat to previous entry
            if str(e).startswith('not enough values to unpack'):
                user = last_defined_user
                post = line
                new_post = False
            else:
                raise Exception(e)

        if user not in posts:
            posts[user] = []

        if not new_post:
            last = posts[user][-1]
            posts[user][-1] = last + ' ' + line
        else:
            posts[user].append(post)
    return posts


def split_to_sentences(sentences, delimiter):
    sentences_parsed = []
    for sentence in sentences:
        if delimiter in sentence:
            sentences_parsed += sentence.split(delimiter)
        else:
            sentences_parsed.append(sentence)
    return sentences_parsed


def tokenize_total(sentences_parsed, output_directory, tokenization_pipeline):
    with open(output_directory + '/total_corpus', 'a') as f:
        for data in sentences_parsed:
            for stage in tokenization_pipeline:
                data = stage['function'](data, stage['param'])
            if len(data) != 0:
                f.write(" ".join(data) + '\n')


def ingest_total(target_file, output_directory, cleaning_pipeline, tokenization_pipeline):
    country = target_file.split('.')[1]

    # read file:
    with open(target_file, 'r') as raw_data:
        raw_data = raw_data.read().split('\n')
        raw_posts = cluster_lines_to_posts(raw_data)

    map_users_scores = dict()
    map_users_parsed = dict()

    print(f'file {target_file}, writers: {len(raw_posts)}')

    sentences_list = []
    # parse sentences
    for user, data in raw_posts.items():
        for stage in cleaning_pipeline:
            data = stage['function'](data, stage['param'])

        map_users_scores[user] = len(data)
        map_users_parsed[user] = data
        # sentences_list.append(data)
        sentences_list += data

    # # sort to get leaders
    # leaders = sorted(map_users_scores.items(), key=lambda item: item[1], reverse=True)[0:lead_factor]
    #
    # # write output
    # total_sentences = 0
    # for user, value in leaders:
    #     print('user:' + user + ' , sentences: ' + str(value))
    #     tokenize(map_users_parsed[user], user, country, output_directory, tokenization_pipeline)
    #     total_sentences += value

    tokenize_total(sentences_list, output_directory, tokenization_pipeline)

    return len(sentences_list)


def run_total(input_directory, output_directory, cleaning_pipeline, tokenization_pipeline):
    total_sentences = 0
    for filename in os.listdir(input_directory):
        print('running for file: ' + filename +
              '\n=================================')
        total = ingest_total(input_directory + '/' + filename,
                             output_directory,
                             cleaning_pipeline,
                             tokenization_pipeline)
        total_sentences += total
        print('total sentences: ' + str(total) +
              '\n=================================')

    print('total sentences in run: ' + str(total_sentences) +
          '\n=================================')


# ======================= END Execution Functions ========================

# ========================= Cleaning Filters =============================


simple_filters = {
    'tagged_web_url': r'\[.*\]\(https?:\/\/[^ ]+\)',  # removes: [text]https://example.com
    'web_url': r'(\s+)?https?:\/\/[^ ]+',  # removes: https://example.com
    'partial_web_url': r'(\s+)?www.[^ ]+',  # removes: www.example.com
    'xml_populated': r'[<][^<>]+[>][^<>]+[<][/][^<>]+[>]',  # removes: <tag>text</tag>
    'xml_base': r'[<][^<>]+[>]',  # removes: <tag>
    'numbering': r'^[0-9]+.',  # removes: 1. 2. 3. etc.
    'exclude_patterns': ''  # removes: all patterns defined later under this key
}

exclude_patterns = [
    '&gt',
    '&lt',
    ':\(',
    ':\)',
    ':’\(',
    ';\)',
    ':D'
]
simple_filters['exclude_patterns'] = build_regex(exclude_patterns)

complex_filters = {
    'comma_starter': [r'^("|\')?([A-Za-z]+),', r'\1'],  # handles: Word, rest of sentence...
    'comma_ender': [r',\s+([A-Za-z]+)("|\')?$', r' \1'],  # handles: Sentence... , word.
}

filters_ordering = [
    'tagged_web_url',
    'web_url',
    'partial_web_url',
    'xml_populated',
    'xml_base',
    'numbering',
    'exclude_patterns'
]

replacers = {
    r'etc\.': r'etc',
    r'i\.e\.': r'ie',
    r"it's": r'its',
    r'([A-Za-z0-9]+)\?': r'\1 ?',
    r'([A-Za-z0-9]+)\!': r'\1 !',
    r'\s*([0-9]+)\.([0-9]+)': r'\1\2',
    r'[.]+': r'.',
    r'([0-9A-Za-z]+)\.(exe|org|com)': r'\1',  # file suffix (file.exe)
    r'(Mr|Ms|Mrs)\s*\.': r'\1',  # titles (Mr,Ms)
    r'(Co|Ltd|Inc)\s*\.': r'\1',  # other shortcuts
    r'\s([A-Za-z])\.\s(.*)': r'\1 \2',
    r'(\s?[A-Za-z])\.([A-Za-z])\.': r'\1\2',
    r'(\s?[A-Za-z])\.([A-Za-z])\.([A-Za-z])\.': r'\1\2\3'
}
# ======================= END Cleaning Filters ===========================

PUNCS_ALLOW = "[?!(),:'@]"

cleaning_pipeline = [
    {'function': clean, 'param': {'order': filters_ordering, 'filters': simple_filters}},
    {'function': replace, 'param': replacers},
    {'function': split_to_sentences, 'param': '.'},
    {'function': clean_complex, 'param': complex_filters},
    # {'function': split_to_sentences, 'param': ','}
]

tokenization_pipeline = [
    {'function': remove_punctuation, 'param': PUNCS_ALLOW},
    {'function': split_to_tokens, 'param': '\s+'},
    {'function': remove_empty_sentences, 'param': None},
]

if __name__ == '__main__':
    if len(sys.argv) == 4:
        input_directory = sys.argv[1]
        lead_factor = int(sys.argv[2])
        output_directory = sys.argv[3]
    else:
        input_directory = 'inputs'
        output_directory = 'outputs'
        lead_factor = 5
    # run(input_directory, lead_factor, output_directory, cleaning_pipeline, tokenization_pipeline)

    run_total('inputs/original', 'outputs/total_tokens', cleaning_pipeline, tokenization_pipeline)
