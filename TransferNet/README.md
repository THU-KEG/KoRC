# TransferNet
Here we adopt TransferNet model adapted for WebQSP dataset method presented in EMNLP 2021 paper *TransferNet: An Effective and Transparent Framework for Multi-hop Question Answering over Relation Graph*

Note: there are a little modification made to fit our dataset

# Preprocess

Since this is a kind of KGQA methods, for each questions, one topic entity is needed, here we utilze the (ELQ (Entity Linking for Questions))[https://github.com/facebookresearch/BLINK/tree/main/elq] model to find the topic entity for each question.


# Data
Since TransferNet is a kind of KGQA model, one knowledge graph is needed.

Similar to embedkgqa baseline, we adopt one small subgraph which is extracted from the [Wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m)

- wyf50.ttl
  - contains 6565019 semantic triplets related to the high freq entitis in the wikidata5m
- acl_ess.ttl
  - contains 3932606 semantic triplets related to all the topic entities for every question in the dataset

# HOW TO RUN
```
cd script
bash {question_type}.sh
```