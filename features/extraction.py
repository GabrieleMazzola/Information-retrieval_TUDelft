import re

import numpy as np
from nltk.corpus import *
from nltk.stem.porter import *


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('opinion_lexicon')

def extract_content_features(similarity_matrix, utterance_index, utterance):
    cont_features = []
    cont_features.append(similarity_matrix[utterance_index, 0])
    cont_features.append(similarity_matrix[utterance_index, -1])
    cont_features.append(question_mark(utterance))
    cont_features.append(same_similar(utterance))
    cont_features += w5h1(utterance)
    return cont_features


def extract_structural_features(utterance_index, dialog, utt_info, starter_user_id, words):
    stemmer = PorterStemmer()
    struct_feat = []
    struct_feat.append(utterance_index)
    struct_feat.append(utterance_index / (len(dialog) - 2))
    filtered_utterance = [word for word in words if word not in stopwords.words('english')]
    struct_feat.append(len(filtered_utterance))
    struct_feat.append(count_unique_words(filtered_utterance))
    struct_feat.append(count_unique_words([stemmer.stem(word) for word in filtered_utterance]))
    struct_feat.append(np.bool(utt_info[1] == starter_user_id))
    return struct_feat


def extract_sentimental_features(utterance, utt_info, words, analyser, pos_opinion_words, neg_opinion_words):
    sent_feat = []
    sent_feat.append(np.bool(thanks(utterance)))
    sent_feat.append(np.bool(exclamation_mark(utterance)))
    sent_feat.append(np.bool(feedback(utterance)))
    sent_feat += sentiment_analyser_scores(utt_info[2], analyser)
    sent_feat += count_opinion(words, pos_opinion_words, neg_opinion_words)
    return sent_feat


def question_mark(utterance):
    return "?" in utterance


def exclamation_mark(utterance):
    return "!" in utterance


def same_similar(utterance):
    return "same" in utterance or "similar" in utterance


def thanks(utterance):
    return "thank" in utterance


def feedback(utterance):
    return "did not" in utterance or "does not" in utterance


def w5h1(utterance):
    res = []
    regex = r'\b{}\b'
    for w in ['what', 'where', 'when', 'why', 'who', 'how']:
        res.append(np.bool(bool(re.search(regex.format(w), utterance))))

    return res


def count_unique_words(seq):
    # Not order preserving
    return len({}.fromkeys(seq))


def count_opinion(words, pos_opinion_words, neg_opinion_words):
    pos = [w for w in words if w in pos_opinion_words]
    neg = [w for w in words if w in neg_opinion_words]
    return [len(pos), len(neg)]


def sentiment_analyser_scores(utterance, analyser):
    scores = analyser.polarity_scores(utterance)
    return [scores['neg'], scores['neu'], scores['pos']]
