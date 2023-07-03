# EmbedKGQA
Here we adopt the RoBERTa based KGQA method presented in ACL 2020 paper *Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings*


# Preprocess

Since this is a kind of KGQA methods, for each questions, one topic entity is needed, here we utilze the (ELQ (Entity Linking for Questions))[https://github.com/facebookresearch/BLINK/tree/main/elq] model to find the topic entity for each question.
# Data
Since when training the embedkqga, you can choose the freeze or unfreeze the entity embeddings.

Here we present 2 knowledge graph which are both sample from [Wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m), and utilze the pretained models provided in [GraphVite](https://graphvite.io/docs/latest/pretrained_model.html)

- complex_wyf_big5m.pkl
  - contains about 480w entity embeddings
- wyf-500-wikidata5m.pkl
  - contains about 20w entity embeddings

# How to Run
```
cd script
bash complex_wyf_big5m.{question_type}.sh
bash wyf-500-wikidata5m.{question_type}.sh
```