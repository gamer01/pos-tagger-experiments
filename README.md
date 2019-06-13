# What it does
This code trains a machine learning POS-tagger. The main dependency is sklearn.

currently word accuracies arround 88% are achieved.

# Dataset
A diacronic corpus of Middle Low German. The dataset is a from a research project and can currently not be published, but it will probably be published here in the future

The structure of the dataset json ist very simple:

    {
      "Document 1": {    <-- document name
        "s001": {        <-- sentence id
          "t01": [       <-- token id
            "et",        <-- token/word
            [
              "PPER"     <-- list of tags
            ]
      ...

The special thing is, that a token might have more than one POS tag, which is uncommon
