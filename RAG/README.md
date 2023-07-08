# RAG

Authors: @patrickvonplaten and @lhoestq

Aimed at tackling the knowledge-intensive NLP tasks (think tasks a human wouldn't be expected to solve without access to external knowledge sources), RAG models are seq2seq models with access to a retrieval mechanism providing relevant context documents at training and evaluation time.

A RAG model encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs are passed to the generator.

Read more about RAG at https://arxiv.org/abs/2005.11401.

# RagWithContext

Since the problem here we face is not simple open qa question, **question contexts are essential** in our problem.
here we treat the question contexts as the retrieved passsages, jointly used to generate answers.

# how to run
1. Build the Faiss Index
    1. wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
    2. gzip -d psgs_w100.tsv.gz
    3. ogranize the psgs_100.tsv.gz as 'title' and 'context' format -> use convert.py in preprocess
    4. python build_faiss.py
2. Build the dataset
    1. prepare_open_dataset.py
    2. prepare_feature.py
3. Run the Experiment
    1. cd script
    2. run experment with bash script
