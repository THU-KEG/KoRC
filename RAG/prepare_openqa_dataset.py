import json
import argparse
import os
from tqdm import tqdm
from typing import List
import re

def convert_to_openqa_dataset(mrc_dataset:List[dict]):
    openqa_dataset = []
    for unit in tqdm(mrc_dataset):
        q = unit['question']
        text = unit['context']
        ans = unit['answers']
        names = re.findall(r'(\[.*?\])',text)
        title = unit['title'] if len(names) == 0 else names[0]
        openqa_dataset.append({
            'question':q,
            'context':text,
            'title':title,
            'answers':ans,
        })
    return openqa_dataset
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',type=str,help='Plz provide the input folder in mrc format')
    parser.add_argument('-o','--output',type=str,help='Plz provide the output folder in openqa format')
    parser.add_argument('-m','--model',type=str,choices=['human','gpt','template'],default='human')

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    mrc_train = json.load(open(os.path.join(input_path,'train.json')))
    mrc_eval = json.load(open(os.path.join(input_path,'valid.json')))
    mrc_iid_test = json.load(open(os.path.join(input_path,'small_iid_test.json')))
    mrc_ood_test = json.load(open(os.path.join(input_path,'small_ood_test.json')))
    if args.model == 'human':
        for dataset in [mrc_train,mrc_eval,mrc_iid_test,mrc_ood_test]:
            for data in dataset:
                data['question'] = data['question']
    elif args.model == 'gpt':
        for dataset in [mrc_train,mrc_eval,mrc_iid_test,mrc_ood_test]:
            for data in dataset:
                data['question'] = data['gpt_best_question'].replace('\n','').strip()
                # data['question'] = data['question'].str
    else:
        for dataset in [mrc_train,mrc_eval,mrc_iid_test,mrc_ood_test]:
            for data in dataset:
                data['question'] = data['template_question']
    

    for dataset,type_path in zip([mrc_train,mrc_eval,mrc_iid_test,mrc_ood_test],['train','eval','small_iid_test','small_ood_test']):
        with open(os.path.join(output_path,f'{type_path}.jsonl'),'w') as f:
            f.writelines([json.dumps(data,ensure_ascii=False)+'\n' for data in convert_to_openqa_dataset(dataset)])
            # json.dump(convert_to_openqa_dataset(dataset),f,indent=4,ensure_ascii=False)

