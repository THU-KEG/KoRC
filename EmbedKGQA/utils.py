import json
import torch
import os
from pathlib import Path
from tqdm import tqdm

print('loading wikidata qid2entity_name')
with open('../wikidata-5m/wikidata-5m-entity-en-label.json') as f:
    qid2entity_name_map = json.load(f)
print('loaded qid 2 entity name')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True    

def validate(dataset,model,device,writeCandidatesToFile=False,data_path=None,hops=None,output_path=None):
    total_acc = 0
    dataset.mode = 'eval'
    if writeCandidatesToFile and data_path is not None:
        with open(data_path) as f:
            ori_data_list = json.load(f)
    for idx,d in tqdm(zip(range(len(dataset)),dataset),total=len(dataset),mininterval=5.0):
        # try:
        question_tokenized = d[0].to(device)
        attention_mask = d[1].to(device)
        head = d[2].to(device)
        ans = d[3].to(device)
        not_in_kg = d[4]
        not_in_kg_num = len(not_in_kg)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        mask = torch.zeros(len(dataset.entity2idx)).to(device)
        mask[head] = 1
        new_scores = scores - (mask*99999)
        probs = torch.softmax(new_scores,dim=0)
        preds = (probs > 1e-2)
        ans = ans == 1
        matchs = torch.logical_and(preds,ans)

        ans_num = ans.sum().cpu().detach().item()
        correct_num = matchs.sum().cpu().detach().item()
        pred_num = preds.sum().cpu().detach().item()
        total_ans_num = ans_num + not_in_kg_num
        p1 = total_ans_num / pred_num if pred_num > total_ans_num else  pred_num / total_ans_num
        acc = correct_num / pred_num if pred_num != 0 else 0
        total_acc += (acc*p1)

        if writeCandidatesToFile:
            ori_data_list[idx]['not_in_kg'] = not_in_kg
            # print(probs[:100])
            # print(preds[:100])
            # print(ori_data_list[idx]['text'])
            # print(ori_data_list[idx]['question'])
            preds_ids = preds.nonzero().squeeze().cpu().detach().numpy()
            ori_data_list[idx]['pred_qids'] = [dataset.idx2entity[idx] for idx in preds_ids] if preds_ids.size != 1 else [dataset.idx2entity[preds_ids.item()]]
            ori_data_list[idx]['pred_names'] = [qid2entity_name_map[qid] for qid in ori_data_list[idx]['pred_qids']]
    if writeCandidatesToFile:
        data_path = Path(data_path)
        with open(os.path.join(output_path,f'{hops}_{data_path.parent.name}_{data_path.name}'),'w') as f:
            json.dump(ori_data_list,f,indent=4,ensure_ascii=False)

    return None, total_acc / len(dataset)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()





