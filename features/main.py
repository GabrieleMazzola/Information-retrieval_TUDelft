import pandas as pd
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from features.extraction import extract_content_features, extract_structural_features, extract_sentimental_features
from processing.processing import get_dialogs_corpus


def extract_features(corpus):
    feature_dict = {}
    analyser = SentimentIntensityAnalyzer()
    pos_opinion_words = set(opinion_lexicon.positive())
    neg_opinion_words = set(opinion_lexicon.negative())

    for dialog_index, dialog in enumerate(corpus):

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([diag[0] for diag in dialog])
        similarity_matrix = cosine_similarity(tfidf)

        starter_user_id = dialog[0][1]

        for utterance_index, utt_info in enumerate(dialog[:-1]):
            utterance = utt_info[0]
            key = str(dialog_index) + "_" + str(utterance_index)

            words = word_tokenize(utterance)

            content_features = extract_content_features(similarity_matrix, utterance_index, utterance)
            structural_features = extract_structural_features(utterance_index, dialog, utt_info, starter_user_id, words)
            sentimental_features = extract_sentimental_features(utterance, utt_info, words, analyser, pos_opinion_words,
                                                                neg_opinion_words)

            feature_dict[key] = content_features + structural_features + sentimental_features

    return feature_dict


if __name__ == '__main__':
    corpus, labels = get_dialogs_corpus()
    features_dict = extract_features(corpus)
    df = pd.DataFrame.from_dict(features_dict, orient='index')

    feature_names = ['InitSim', 'DlgSim', 'QuestMark', 'Dup', 'What', 'Where', 'When', 'Why', 'Who', 'How',
                     'AbsPos', 'NormPos', 'Len', 'LenUni', 'LenStem', 'Starter',
                     'Thank', 'ExMark', 'Feedback', 'SenScr(Neg)', 'SenScr(Neu)', 'SenScr(Pos)', 'Lex(Pos)', 'Lex(Neg)']

    df.columns = feature_names
    df['label'] = labels

    # df['label'] = df['label'].apply(lambda tags: tags.split(" "))

    df.to_csv("../data/features_extracted.csv")
