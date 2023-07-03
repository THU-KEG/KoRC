# from use_own_knowledge_dataset import split_documents,embed,split_text
import re
import torch
from typing import List,Dict
from transformers import BartTokenizer, RagTokenizer, T5Tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def encode_ctx(tokenizer, context, title, question, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    tokenizer.padding_side = padding_side
    ctx_splits = split_text(context)
    return tokenizer(
        [title+' / '+line + ' // ' + question for line in ctx_splits],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
    )

def embed_ctx(ctx_encoder,tokenizer, context, title, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    tokenizer.padding_side = padding_side
    ctx_splits = split_text(context)
    input_ids = tokenizer(
        [[title,line] for line in ctx_splits],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
    )['input_ids']
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return embeddings

class OpenQAMap(object):
    def __init__(
        self,
        tokenizer,
        ctx_tokenizer,
        ctx_encoder,
        max_source_length,
        max_target_length,
        max_context_length,
        question_with_ctx,
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        # self.data = json.load(open( Path(data_dir).joinpath(type_path + ".json")))
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.ctx_encoder = ctx_encoder.to(device)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_context_length = max_context_length
        self.question_with_ctx = question_with_ctx
    
    def prepare_features(self,example):
        # print(example['answers'])
        source_line = example['question']
        tgt_line = ' <ans> '.join(example['answers'])
        ctx_line =  example['context']
        names = re.findall(r'(\[.*?\])',ctx_line)
        title = example['title'] if len(names) == 0 else names[0]
        

        # Need to add eos token manually for T5
        if isinstance(self.tokenizer, T5Tokenizer):
            source_line += self.tokenizer.eos_token
            tgt_line += self.tokenizer.eos_token

        # Pad source and target to the right
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer

        source_inputs = source_tokenizer(
            [[ctx_line,source_line]],
            max_length=self.max_source_length,
            padding="max_length",
            truncation="only_first",
            return_tensors='pt',
            add_special_tokens=True,
        ) if self.question_with_ctx else encode_line(source_tokenizer, source_line, self.max_source_length, "right")
        target_inputs = encode_line(target_tokenizer, tgt_line, self.max_target_length, "right")
        context_inputs = encode_ctx(target_tokenizer, ctx_line, title, source_line, self.max_context_length, 'right')
        context_embs = embed_ctx(self.ctx_encoder,self.ctx_tokenizer, ctx_line, title, self.max_context_length, "right")
        
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
            'question_context_input_ids':context_inputs['input_ids'],
            'question_context_attention_mask':context_inputs['attention_mask'],
            'question_context_embs':context_embs,
            # 'question_context_splits':split_text(ctx_line)
        }

class OpenQACollator(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.max_split = 5
        self.fp16 = False
        
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        int_cls = torch.int16 if self.fp16 else torch.long
        fp_cls = torch.float16 if self.fp16 else torch.float
        input_ids = torch.stack([torch.Tensor(x["input_ids"]) for x in batch]).to(int_cls)
        masks = torch.stack([torch.Tensor(x["attention_mask"]) for x in batch]).to(int_cls)
        target_ids = torch.stack([torch.Tensor(x["decoder_input_ids"]) for x in batch]).to(int_cls)
        
        question_context_input_ids = [torch.Tensor(x['question_context_input_ids']).to(int_cls) for x in batch]
        question_context_attention_mask = [torch.Tensor(x['question_context_attention_mask']).to(int_cls) for x in batch]
        question_context_embs = [torch.Tensor(x['question_context_embs']) for x in batch]

        # align to the same shape (max_split,max_len)
        # missed_rows = 
        question_context_input_ids=torch.stack([torch.cat((x[:self.max_split],torch.zeros((max(0,self.max_split-x.shape[0]),x.shape[1]))),dim=0) for x in question_context_input_ids]).to(int_cls)
        question_context_attention_mask=torch.stack([torch.cat((x[:self.max_split],torch.zeros((max(0,self.max_split-x.shape[0]),x.shape[1]))),dim=0) for x in question_context_attention_mask]).to(int_cls)
        question_context_embs=torch.stack([torch.cat((x[:self.max_split],torch.zeros((max(0,self.max_split-x.shape[0]),x.shape[1]))),dim=0) for x in question_context_embs]).to(fp_cls)
        
        
        
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        y = trim_batch(target_ids, tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
            'question_context_input_ids': question_context_input_ids,
            'question_context_attention_mask': question_context_attention_mask,
            'question_context_embs': question_context_embs,
        }
        return batch

# class OpenQADataset(Dataset):
#     def __init__(
#         self,
#         tokenizer,
#         ctx_tokenizer,
#         ctx_encoder,
#         data_dir,
#         max_source_length,
#         max_target_length,
#         max_context_length,
#         type_path="train",
#         n_obs=None,
#         src_lang=None,
#         tgt_lang=None,
#         prefix="",
#     ):
#         super().__init__()
#         self.data = json.load(open( Path(data_dir).joinpath(type_path + ".json")))
#         self.tokenizer = tokenizer
#         self.ctx_encoder = ctx_encoder.to(device)
#         self.ctx_tokenizer = ctx_tokenizer
#         self.max_source_length = max_source_length
#         self.max_target_length = max_target_length
#         self.max_context_length = max_context_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index) -> Dict[str, torch.Tensor]:

#         source_line = self.data[index]['question']
#         tgt_line = ' <ans> '.join(self.data[index]['answers'])
#         ctx_line =  self.data[index]['text']
#         title = self.data[index]['title']

        

#         # Need to add eos token manually for T5
#         if isinstance(self.tokenizer, T5Tokenizer):
#             source_line += self.tokenizer.eos_token
#             tgt_line += self.tokenizer.eos_token

#         # Pad source and target to the right
#         source_tokenizer = (
#             self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
#         )
#         target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer

#         source_inputs = encode_line(source_tokenizer, source_line, self.max_source_length, "right")
#         target_inputs = encode_line(target_tokenizer, tgt_line, self.max_target_length, "right")
#         context_inputs = encode_ctx(self.ctx_tokenizer, ctx_line, title, self.max_context_length, "right")
#         context_embs = embed_ctx(self.ctx_encoder,context_inputs['input_ids'])
#         source_ids = source_inputs["input_ids"].squeeze()
#         target_ids = target_inputs["input_ids"].squeeze()
#         src_mask = source_inputs["attention_mask"].squeeze()
#         return {
#             "input_ids": source_ids,
#             "attention_mask": src_mask,
#             "decoder_input_ids": target_ids,
#             'question_context_input_ids':context_inputs['input_ids'],
#             'question_context_attention_mask':context_inputs['attention_mask'],
#             'question_context_embs':context_embs,
#             'question_context_splits':split_text(ctx_line)
#         }

#     # @staticmethod
#     # def get_char_lens(data_file):
#     #     return [len(x) for x in Path(data_file).open().readlines()]

#     def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
#         input_ids = torch.stack([x["input_ids"] for x in batch])
#         masks = torch.stack([x["attention_mask"] for x in batch])
#         target_ids = torch.stack([x["decoder_input_ids"] for x in batch])

#         tgt_pad_token_id = (
#             self.tokenizer.generator.pad_token_id
#             if isinstance(self.tokenizer, RagTokenizer)
#             else self.tokenizer.pad_token_id
#         )
#         src_pad_token_id = (
#             self.tokenizer.question_encoder.pad_token_id
#             if isinstance(self.tokenizer, RagTokenizer)
#             else self.tokenizer.pad_token_id
#         )
#         y = trim_batch(target_ids, tgt_pad_token_id)
#         source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
#         batch = {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "decoder_input_ids": y,
#             'question_context_input_ids': torch.stack([x['question_context_input_ids'] for x in batch]),
#             'question_context_attention_mask': torch.stack([x['question_context_attention_mask'] for x in batch]),
#             'question_context_embs': torch.stack([x['question_context_embs'] for x in batch]),
#             'question_context_splits':[x['question_context_splits'] for x in batch]
#         }
#         return batch
