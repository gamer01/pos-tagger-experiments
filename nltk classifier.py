#!/usr/bin/env python
# coding: utf-8
import inspect
import json
import os
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def load_dataset():
    with open("dataset.json") as f:
        data = json.load(f)

    tagged_sentences = []
    for _, doc in data.items():
        for _, sent in doc.items():
            # remove full stop at the end of each sentence
            # and only take the fist tag
            tagged_sentences.append([(tok[0], tok[1][0]) for _, tok in list(sent.items())[:-1]])

    print(tagged_sentences[0])
    print("Tagged sentences: ", len(tagged_sentences))
    print("Tagged words:", sum(map(len, tagged_sentences)))
    return tagged_sentences


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


if __name__ == '__main__':
    tagged_sentences = load_dataset()
    X, y = transform_to_dataset(tagged_sentences)
    print(Counter(y))

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', LinearSVC())
    ])

    start = datetime.now()
    scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)
    acc = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print("Accuracy:", acc)
    with open(os.path.join("results", datetime.now().isoformat(" ", "seconds") + ".json"), "w") as f:
        json.dump(
            {"runtime": f"{datetime.now()-start}",
             "accuracy": acc,
             "featurefun": inspect.getsource(features),
             "clf": repr(clf)},
            f, indent=2)
