import os
from transformers import AutoTokenizer, RobertaTokenizer

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer


# Use tokenizer.add_special_tokens, and model.resize_token_embeddings to add special tokens
# This will return the tokenizer and the model
# If we get a brand new model (without fine-tuning), we add new_tokens to the tokenizer and the embedding layer of the model
# Else, we load the model from the given file_dir
# ** when we need to load from a ready model to avoid resizing the embedding matrix, resize_token_embeddings has not effect because the embedding matrix is already the size we want **
def get_model(model_type, model_name, tokenizer_type, tokenizer_name, new_tokens:list = []):
    if model_type in ['t5', 'mt5']:
        model_cls = T5ForConditionalGeneration
    elif model_type in ['bart']:
        model_cls = BartForConditionalGeneration
    else:
        raise ValueError("Unsupported Model")

    if tokenizer_type in ['t5', 'mt5']:
        tokenizer_cls = T5Tokenizer
    elif tokenizer_type in ['bart']:
        tokenizer_cls = BartTokenizer
    elif tokenizer_type in ['roberta']:
        tokenizer_cls = RobertaTokenizer
    else:
        raise ValueError("Unsupported Model")
    
    model = model_cls.from_pretrained(model_name)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config
