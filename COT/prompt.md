# model
- glm-130b
- text-davinci-002
with temperature 0.7

# prompt
```
Instruction: you are given one document and one anonymized real-world entity with one or more mentions in the passage. Then we will ask your a question about this anonymized entity. The questions cannot be answered solely within the document or the background knowledge. Your task is to leverage world knowledge you have like Wikipedia or wikidata as background knowledge combined with the given document to answer the question related to the anonymized entity. You must output all answers in the end
Document: {context}
Question: {question}
Answer: {answer}
```