import json
from single import Metric
datapath = 'checkpoints/roberta_finetune/'
# import pickle
kg_name = 'complex_wyf_big5m'
# wiki5m = pickle.load(open(kg_name+'.pkl','rb'))
# entity2idx = wiki5m.graph.entity2id

# print('loading wikidata qid2entity_name')
# with open('../wikidata-5m/wikidata-5m-entity-en-label.json') as f:
#     qid2entity_name_map = json.load(f)
# print('loaded qid 2 entity name')

for question_type in ['human']:
    print("="*75 + question_type + "="*75)
    for result_type in ['human']:
        print("*"*50 + result_type + "*"*50)
        for file in [f'{kg_name}_{result_type}_small_iid_test.json',f'{kg_name}_{result_type}_small_ood_test.json']:
            dataset = json.load(open(datapath+question_type+"/"+file))
            total_pred_name_list = [[pred['label'] for pred in data['pred_names']] for data in dataset]        
            total_ans_name_list = [data['answers'] for data in dataset]

            total_pred_id_list = [data['pred_qids'] for data in dataset]        
            total_ans_id_list = [data['answer_ids'] for data in dataset]

            em_acc = Metric.str_metric(total_pred_name_list,total_ans_name_list)['em_acc_with_penalty']
            token_f1 = Metric.str_metric(total_pred_id_list,total_ans_id_list)['token_f1_with_penalty']

            print({
                'em_acc':em_acc,
                'token_f1':token_f1,
            })