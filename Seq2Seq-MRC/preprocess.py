# new dataset
import json
import random
random.seed(0)
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List

def get_ans_str(ans:List[str],spt_token='<ans>')->str:
    return f' {spt_token} '.join(ans)

def truncate_input(truncatable_texts:List[str],key_texts:List[str],spt_token:str,tokenizer,max_len=512)->str:
    full_texts = truncatable_texts + key_texts
    ori_input = f' {spt_token} '.join(full_texts)
    ori_ids = tokenizer(ori_input)
    if len(ori_ids) < max_len:
        return ori_input
    else:
        trunct_ids = [tokenizer(text) for text in truncatable_texts]
        trunct_lens = [len(ids) for ids in trunct_ids]
        overlength = len(ori_ids) - max_len
        after_trunct_texts = []
        for i,text_len,ids in zip(range(len(trunct_lens)),trunct_lens,trunct_ids):
            if overlength < text_len:
                after_trunct_texts.append(tokenizer.decode(ids[:-overlength-3]))
                after_trunct_texts.extend(truncatable_texts[i+1:])
            else:
                overlength -= text_len
        
        full_texts = after_trunct_texts + key_texts
        return f' {spt_token} '.join(full_texts)

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
special_tokens = ['<spt>']
print(f'add special tokens {special_tokens}')
tokenizer.add_tokens(special_tokens,special_tokens=True)


def convert2mrc_format(dataset,question_type):
    new_dataset = []
    for unit in tqdm(dataset):
        question = unit['question']
        if question_type == 'gpt':
            question = unit['gpt_best_question'].replace('\n','').strip()
        elif question_type == 'template':
            question = unit['template_question']
        context = unit['context']
        assert len(unit['answer_ids']) == len(set(unit['answer_ids'])),print(unit)
        new_dataset.append(
            {
                "id":unit['id'],
                "input":truncate_input([context],[question],'<spt>',tokenizer),
                "output":get_ans_str(unit['answers'])
            }
        )
    return new_dataset

if __name__ == '__main__':

    for question_type in ['human','gpt','template']:
        input_dir = '../dataset/kgqa/'
        output_dir = f'../dataset/mrc/{question_type}/'
        with open(output_dir+'train.json','w') as f:
            train_dataset = convert2mrc_format(json.load(open(input_dir+'train.json')),question_type)
            random.shuffle(train_dataset)
            json.dump(train_dataset,f,indent=4,ensure_ascii=False)

        with open(output_dir+'eval.json','w') as f:
            json.dump(convert2mrc_format(json.load(open(input_dir+'valid.json')),question_type),f,indent=4,ensure_ascii=False)

        with open(output_dir+'small_iid_test.json','w') as f:
            json.dump(convert2mrc_format(json.load(open(input_dir+'small_iid_test.json')),question_type),f,indent=4,ensure_ascii=False)

        with open(output_dir+'small_ood_test.json','w') as f:
            json.dump(convert2mrc_format(json.load(open(input_dir+'small_ood_test.json')),question_type),f,indent=4,ensure_ascii=False)

