# WARNING:
`runner_asyncio.py` is extremely unsafe: running shell commands, without checking, read from a file. Only use this in a highly secure environment.

Zip of results are in the results folder.


# Setup:

1. Unzip datasets from datasets folder.
2. Set up conda environment from `env.yaml`
3. `python -m spacy download en_core_web_lg`
4. NLTK download stopwords: >> import nltk >> nltk.download("stopwords")

## CoreNLP
Make sure the CoreNLP server is running if you wish to include the stanford parser in your grid.

Guidance on how to do so is [here](https://stackoverflow.com/a/47627069/12275864) or [here](https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK).
If you CBA reading you can download version used for this repo with:
```
cd somewhere
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
unzip stanford-corenlp-full-2016-10-31.zip
cd stanford-corenlp-full-2016-10-31
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit \
-status_port 9000 -port 9000 -timeout 15000
```
Tested with 
openjdk 11.0.22 2024-01-16
OpenJDK Runtime Environment (build 11.0.22+7-post-Ubuntu-0ubuntu222.04.1)
OpenJDK 64-Bit Server VM (build 11.0.22+7-post-Ubuntu-0ubuntu222.04.1, mixed mode, sharing)

## Running a many experiments
To run all experiments start `runner_asyncio.py` script and paste in the commands for experiments into the queue file.

## TODO
- [ ] Add documentation on how original raw datasets were transformed into the data used here

# Exploring the code
Start at `pipeline.py`.