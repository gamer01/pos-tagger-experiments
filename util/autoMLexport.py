from itertools import chain

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from main import transform_to_dataset
from util.convert_dataset import load_dataset, singlesplit

docs = load_dataset("../dataset.json")
train_sents = []
eval_sents = []
for tagged_sents in docs:
    train, eval = singlesplit(tagged_sents, .8)
    train_sents.extend(train)
    eval_sents.extend(eval)

X, y = {}, {}
X["train"], y["train"] = transform_to_dataset(train_sents)
X["eval"], y["eval"] = transform_to_dataset(eval_sents)

vec = DictVectorizer(sparse=False)
vec.fit(chain.from_iterable(X.values()))
for k, v in X.items():
    X[k] = vec.transform(v)

if __name__ == '__main__':
    for split in ("train", "eval"):
        df = pd.DataFrame(X[split])
        df["groundtruth"] = y[split]
        df.to_csv(f"AutoML{split}.csv.gz", index=False, chunksize=1000, compression='gzip', encoding='utf-8')
