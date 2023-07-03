from datasets import load_dataset
from transformers import (
    RagTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)
from openqa_dataset import OpenQAMap
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-o','--output',type=str,required=True,help='the output dir')
parser.add_argument('--max_tgt_len',type=int,default=64,help='the max length of tgt string')
parser.add_argument('--max_source_length',type=int,default=512,help='the max length of quesiton string')
parser.add_argument('--max_context_length',type=int,default=200,help='the max length of quesiton context string')
parser.add_argument('--question_with_ctx',default=False,action='store_true',help='Whether question with ctx') # MUST BE TRUE

args = parser.parse_args()
output_dir = args.output
max_source_length = args.max_source_length
max_tgt_len = args.max_tgt_len
max_context_length = args.max_context_length
question_with_ctx = args.question_with_ctx

assert question_with_ctx, print('MUST BE TRUE')

special_tokens = ['<ans>']
print(f'add special tokens {special_tokens}')
rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
rag_tokenizer.generator.add_tokens(special_tokens,special_tokens=True)
openqamap = OpenQAMap(
    tokenizer=rag_tokenizer,
    ctx_tokenizer=DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base'),
    ctx_encoder=DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base'),
    max_source_length=max_source_length,
    max_target_length=max_tgt_len,
    max_context_length=max_context_length,
    question_with_ctx=question_with_ctx
)
data_path = output_dir
ds = load_dataset(
    'json',
    data_files = {
        'train':data_path+'train.jsonl',
        'val':data_path+'eval.jsonl',
        # 'test':data_path+'iid_test.jsonl'
    }
)
train_dataset = ds['train'].map(openqamap.prepare_features, batched=False, remove_columns=ds["train"].column_names)
val_dataset = ds['val'].map(openqamap.prepare_features, batched=False, remove_columns=ds["val"].column_names)
# test_dataset = ds['test'].map(openqamap.prepare_features, batched=False, remove_columns=ds["test"].column_names)

train_dataset.save_to_disk(os.path.join(output_dir,'train_dataset'))
# test_dataset.save_to_disk(os.path.join(output_dir,'test_dataset'))
val_dataset.save_to_disk(os.path.join(output_dir,'val_dataset'))

ds = load_dataset(
    'json',
    data_files = {
        # 'train':data_path+'train.jsonl',
        'val':data_path+'small_ood_test.jsonl',
        'test':data_path+'small_iid_test.jsonl'
    }
)
# train_dataset = ds['train'].map(openqamap.prepare_features, batched=False, remove_columns=ds["train"].column_names)
val_dataset = ds['val'].map(openqamap.prepare_features, batched=False, remove_columns=ds["val"].column_names)
test_dataset = ds['test'].map(openqamap.prepare_features, batched=False, remove_columns=ds["test"].column_names)

# train_dataset.save_to_disk(os.path.join(output_dir,'train_dataset'))
val_dataset.save_to_disk(os.path.join(output_dir,'small_ood_test_dataset'))
test_dataset.save_to_disk(os.path.join(output_dir,'small_iid_test_dataset'))