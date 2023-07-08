# Seq2Seq-MRC

This is the implementation of Seq2Seq-MRC model for KoRC dataset. It directly generates the answer based on the given document and question with Causal Language Model like BART, T5, Flan-T5, etc.


## Preprocess

Since there could be multiple answers for each question, we need to preprocess the dataset to make it suitable for Seq2Seq-MRC model, namely connect all the answers with `<ans>` into one string.

```
cd Seq2Seq-MRC
python preprocess.py

```

## HOW TO RUN

```
cd script
bash {model_name}.korc.{question_type}.sh

# for prediction
bash {model_name}.korc.{question_type}.pred.sh 
```
