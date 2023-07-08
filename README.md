# KQA Pro Baselines
[KoRC](https://arxiv.org/pdf/2307.03115.pdf) is a Knowledge oriented Reading Comprehension Benchmark for Deep Text Understanding


This repo implements several baselines for the benchmark:

- [chain-of-thought] (https://arxiv.org/abs/2201.11903) with [text-davinci-002] (https://platform.openai.com/docs/models/gpt-3-5) or [GLM-130B] (https://github.com/THUDM/GLM-130B) 
- Seq2Seq-MRC. It directly generates the answer based on the given document and question with Causal Language Model like BART, T5, Flan-T5, etc.
- [RAG](https://arxiv.org/abs/2005.11401) (Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)
- [EmbedKGQA](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)
- [TransferNet](https://aclanthology.org/2021.emnlp-main.341)

Instructions of how to run these models are described in their README files.
Before trying them, you need to first download the [dataset](https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1) and unzip it into the folder `./dataset`.
The file tree should be like
```
.
+-- dataset
|   +-- kb.json
|   +-- train.json
|   +-- val.json
|   +-- test.json
+-- Seq2Seq-MRC
|   +-- preprocess.py
|   +-- train.py
|   +-- ...
+-- KVMemNN
+-- RGCN
...
```