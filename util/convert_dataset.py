import json
from collections import Counter

import pandas as pd


def load_dataset():
    with open("../dataset.json") as f:
        data = json.load(f)

    tagged_sentences = []
    for _, doc in data.items():
        for _, sent in doc.items():
            # remove full stop at the end of each sentence
            # and only take the fist tag
            # and drop inheritance relations of tags
            tagged_sentences.append([(tok, tag[0].split("<")[0]) for _, (tok, tag) in list(sent.items())[:-1]])

    return tagged_sentences


def tagged_sents_tostring(sents, dlm="_"):
    lines = []
    for sent in sents:
        lines.append(" ".join([tok + dlm + tag for tok, tag in sent]))
    return "\n".join(lines)


if __name__ == '__main__':
    tagged_sentences = load_dataset()

    with open("dataset.tagged", "w") as f:
        print(tagged_sents_tostring(tagged_sentences), file=f)
