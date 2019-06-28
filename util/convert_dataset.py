import json
from itertools import chain
from logging import info
from random import shuffle, randrange


def load_dataset(file):
    with open(file) as f:
        data = json.load(f)

    docs = []
    tagged_sentences = []
    for _, doc in data.items():
        for _, sent in doc.items():
            # and drop inheritance relations of tags
            tagged_sentences.append(
                [(tok, [tag.split("<")[0] for tag in tags]) for _, (tok, tags) in list(sent.items())])
        docs.append(tagged_sentences)
        tagged_sentences = []

    return docs


def split(tagged_sents, fraction=.9, randomized=True):
    '''
    shuffle sentences
    select starttoken randomly (needed, because maybe there are to little sentences)
    select endtoken by fraction
    :param tagged_sentences:  [[(tok1,[tag1,tag2]),(tok2....
    :param fraction:
    :return:
    '''
    if randomized:
        shuffle(tagged_sents)
    n_toks = sum((len(sent) for sent in tagged_sents))

    cut1 = randrange(0, n_toks)
    cut2 = (cut1 + int(fraction * n_toks)) % n_toks

    # this means we will flip cutting positions and also flip the returned arrays
    invert = cut1 > cut2
    if invert:
        cut1, cut2 = cut2, cut1
    info(f"invert:{invert}  cuts:{cut1},{cut2} ntoks:{n_toks}")

    outer, inner = [], []
    i = -1  # preincrement
    for sent in tagged_sents:
        new_sent = []
        for tok in sent:
            i += 1
            if i == cut1:
                # first element in inner
                if new_sent:
                    outer.append(new_sent)
                new_sent = []
            if i == cut2:
                # first element in outer
                if new_sent:
                    inner.append(new_sent)
                new_sent = []
            new_sent.append(tok)
        if new_sent:
            if cut1 <= i < cut2:
                inner.append(new_sent)
            else:
                outer.append(new_sent)

    return (outer, inner) if invert else (inner, outer)


assert sum((len(sent) for sent in split([
    "a b c".split(),
    "4 4 3 2 3 4 4 3 2 9 9 9 5 4 3 2 3".split()
])[1])) == 2


def tagged_sents_tostring(sents, dlm="/"):
    lines = []
    for sent in sents:
        lines.append(" ".join([tok + dlm + tags[0] for tok, tags in sent]))
    return "\n".join(lines)


if __name__ == '__main__':
    docs = load_dataset("dataset.json")
    train_sents = []
    eval_sents = []
    for tagged_sents in docs:
        train, eval = split(tagged_sents)
        train_sents.extend(train)
        eval_sents.extend(eval)

    with open("dataset.tagged", "w") as f:
        print(tagged_sents_tostring(chain.from_iterable(docs)), file=f)
    with open("train.tagged", "w") as f:
        print(tagged_sents_tostring(train_sents), file=f)
    with open("eval.tagged", "w") as f:
        print(tagged_sents_tostring(eval_sents), file=f)
