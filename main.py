#!/usr/bin/env python
# coding: utf-8

import inspect
import json
import os
import socket
import subprocess
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from util import convert_dataset


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
            # only take first tag
            y.append(tagged[index][1][0])

    return X, y


def fit_and_score(cls, Xtrain, ytrain, Xeval, yeval):
    cls.fit(Xtrain, ytrain)
    return clf.score(Xeval, yeval)


if __name__ == '__main__':
    docs = convert_dataset.load_dataset("dataset.json")
    k = 10
    splitters = []
    for tagged_sents in docs:
        splitters.append(convert_dataset.kfoldsplit(tagged_sents, k=k))

    splits = []
    for _ in range(k):
        train_sents = []
        eval_sents = []
        for splitter in splitters:
            train, eval = next(splitter)
            train_sents.extend(train)
            eval_sents.extend(eval)

        splits.append(transform_to_dataset(train_sents) + transform_to_dataset(eval_sents))

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', LogisticRegression())
    ])

    start = datetime.now()
    scores = np.array(Parallel(-1)(delayed(fit_and_score)(clf, *split) for split in splits))
    end = datetime.now()

    acc = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print("Accuracy:", acc)

    timestamp = start.isoformat(" ", "seconds")
    commit_id = subprocess.run("git rev-parse --short HEAD".split(" "), capture_output=True).stdout.decode(
        "utf-8").strip()

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", timestamp + ".json"), "w") as f:
        json.dump(
            {"start time": timestamp,
             "runtime": f"{end-start}",
             "git commit id": commit_id,
             "hostname": socket.gethostname(),
             "accuracy": acc,
             "n sentences": sum(len(doc) for doc in docs),
             "n words": sum(len(split) for split in splits[0][1:2:3]),
             "dataset modifications": inspect.getsource(convert_dataset.load_dataset),
             "feature func": inspect.getsource(features),
             "classifier": repr(clf)},
            f, indent=2)
