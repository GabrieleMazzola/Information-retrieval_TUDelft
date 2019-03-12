import random
import re

import pandas as pd

TAGS_TO_REMOVE = ['GG', 'O', 'JK']
tags_count = {}


def clean_str_lw(string):
    return clean_str(string).lower()


def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def process_tags(tags_str):
    tags = sorted(tags_str.strip().split())

    for tag_to_remove in TAGS_TO_REMOVE:
        if len(tags) > 1 and tag_to_remove in tags:
            tags.remove(tag_to_remove)

    tag_string = " ".join(tags)
    tags_count[tag_string] = tags_count.setdefault(tag_string, 0) + 1
    return tag_string


def get_corpus():
    new_corpus = []

    dataset = pd.read_json(
        "C:\\Users\\gabri\\Desktop\\TUDelft\\Q3-4\\InformationRetrieval\\MSDialog\\MSDialog-Intent.json",
        orient='index') \
        .to_dict(orient="records")
    for dialog in dataset:
        new_corpus.append(clean_str_lw(dialog['title']))
        utterances = []

        for utt_info in dialog['utterances']:
            utterances.append(clean_str_lw(utt_info['utterance']))

        new_corpus += utterances
    return new_corpus


def get_dialogs_corpus():
    new_corpus = []

    dataset = pd.read_json(
        "C:\\Users\\gabri\\Desktop\\TUDelft\\Q3-4\\InformationRetrieval\\MSDialog\\MSDialog-Intent.json",
        orient='index') \
        .to_dict(orient="records")

    tags = []

    for dialog in dataset:
        for utt_info in dialog['utterances']:
            tags.append(process_tags(utt_info['tags']))

    tags_count_desc = sorted(tags_count.items(), key=lambda tag_count: tag_count[1], reverse=True)
    best_32 = dict(tags_count_desc[:32])

    for index, tag in enumerate(tags):
        if tag not in best_32:
            tokens = tag.split(" ")
            tag_str = random.choice(tokens)
            tags[index] = tag_str

    for dialog in dataset:
        utterances = []

        for utt_info in dialog['utterances']:
            lowered_utt = clean_str_lw(utt_info['utterance'])
            user_id = utt_info['user_id']
            utt = clean_str(utt_info['utterance'])
            utterances.append((lowered_utt, user_id, utt))

        dialog_utterance_lw = (" ".join([utt[0] for utt in utterances[1:]]),)
        utterances.append(dialog_utterance_lw)
        new_corpus.append(utterances)

    return new_corpus, tags


if __name__ == '__main__':
    get_dialogs_corpus()