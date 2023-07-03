import json
import torch
from torch.utils import data
from transformers import AutoTokenizer

class Seq2SeqDataset(torch.utils.data.Dataset):
    #   Train, Eval  :
    ### input, output, xxx  ###
    #   Test        :
    ### input               ###
    def __init__(self, file_name, data_args, tokenizer : AutoTokenizer, mode = "train"):
        super().__init__()
        self.data = json.load(open(file_name))
        self.size = len(self.data)
        
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_args = data_args

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        data_slice = self.data[i]
        output_item = dict()

        input_seq = data_slice['input']
        output_item['input'] = input_seq

        # if self.mode != "test":
        if True:
            output_seq = data_slice['output']
            output_item['output'] = output_seq
        

        for k in data_slice.keys():
            if k not in ['input', 'output']:
                output_item[k] = data_slice[k]

        return output_item

    def collate_fn(self, data_batch):
        output_batch = dict()

        input_seqs = [x['input'] for x in data_batch]
        input_after_tokenizer = self.tokenizer(
            input_seqs, 
            return_tensors="pt", 
            padding='longest',
            max_length = self.data_args.max_input_length,
            truncation = True,
        )
        input_tokens = input_after_tokenizer.input_ids
        input_attens = input_after_tokenizer.attention_mask
        output_batch['input_ids'] = input_tokens
        output_batch['attention_mask'] = input_attens
        # output_batch['input_tokens'] = [self.tokenizer.tokenize(x) for x in input_seqs] 

        output_seqs = [x['output'] for x in data_batch]
        if self.mode == "train" and type(output_seqs[0]) != list:
            # if 'output' in data_batch[0]:
            # print(output_seqs)
            output_after_tokenizer = self.tokenizer(
                output_seqs, 
                return_tensors="pt", 
                padding='longest',
                max_length = self.data_args.max_output_length,
                truncation = True,
            )
            output_tokens = output_after_tokenizer.input_ids
            output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100
            output_batch['labels'] = output_tokens
            # output_attens = output_after_tokenizer.attention_mask
            # output_batch['decoder_input_ids'] = output_tokens
            # output_batch['decoder_attention_mask'] = output_attens
            # output_batch['output_tokens'] = [self.tokenizer.tokenize(x) for x in output_seqs] 
        else:
            # print(json.dumps(data_batch,indent=4,ensure_ascii=False))
            # output_seqs = [x['output'] for x in data_batch]
            output_tokens = [self.tokenizer(
                seqs_for_single_q, 
                return_tensors="pt", 
                padding='longest',
                max_length = self.data_args.max_output_length,
                truncation = True,
            ).input_ids for seqs_for_single_q in output_seqs]

            for label_ids in output_tokens:
                label_ids[label_ids == self.tokenizer.pad_token_id] = -100

            # output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100
            output_batch['labels'] = output_tokens

        for k in data_batch[0].keys():
            if k not in ['input', 'output']:
                output_batch[k] = [x[k] for x in data_batch]
        
        return output_batch




from torch.utils.data import DataLoader