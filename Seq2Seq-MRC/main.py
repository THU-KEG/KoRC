import json
import os
import importlib
import torch
from torch.utils.data import dataset
from transformers import EarlyStoppingCallback
from transformers import HfArgumentParser
from transformers import set_seed
from arg_parser import ModelArguments, DataArguments, TrainingArguments
from pathlib import Path
# Define arguments for Model, Data, and Training
from model import get_model
from dataset import DataLoader, Seq2SeqDataset
from transformers import Seq2SeqTrainer
import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(filename)s:L%(lineno)d] [%(asctime)s] - %(levelname)-6s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel('INFO')

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()

# load wandb
if train_args.do_train:
    import wandb
    wandb.init(project=model_args.model_name.replace('/','_'),name=model_args.model_id)

if train_args.deepspeed and train_args.do_predict == train_args.do_train:
    print("You Cannot Train and Predict in one process when you are using deepspeed")
    exit()

set_seed(train_args.seed)
train_args.generation_max_length = data_args.max_output_length

# Disable Tqdm output for HW
train_args.disable_tqdm = True

# Refine the output dir with the model ID
train_args.output_dir = train_args.output_dir + model_args.model_id

# Import Task Configurations, mainly the metric, and the special tokens
task_config = importlib.import_module(model_args.conf_dir)
print(f'add labels')
train_args.label_names = task_config.label_names
print(train_args.label_names)



# Get the model, and the tokenizer
if model_args.add_special_token:
    special_tokens = task_config.special_tokens
    print("Add Special Tokens!")
    print(special_tokens)
else:
    special_tokens = []
    

if model_args.tokenizer_type is None:
    print("Assign Tokenizer!")
    model_args.tokenizer_type = model_args.model_type
model, tokenizer, model_config = get_model(model_args.model_type, model_args.model_name, model_args.tokenizer_type, model_args.tokenizer_name, special_tokens)

print(f'load metric')
metric_instance = task_config.Metric(tokenizer)
print(type(metric_instance.tokenizer))

# Disable use_cache when using deepspeed
if train_args.deepspeed:
    model.config.use_cache = False

print(train_args)

        
train_dataset = Seq2SeqDataset(data_args.train_dir, data_args, tokenizer, "train")
eval_dataset = Seq2SeqDataset(data_args.eval_dir, data_args, tokenizer, "eval")
test_dataset = Seq2SeqDataset(data_args.test_dir, data_args, tokenizer, "test")
if 'small' in data_args.test_dir and 'iid' in data_args.test_dir:
    iid_test_dataset = test_dataset
    ood_test_dataset = Seq2SeqDataset(data_args.test_dir.replace('small_iid_test','small_ood_test'), data_args, tokenizer, "test")


# Trainer
trainer = Seq2SeqTrainer(
    model = model,
    args = train_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = metric_instance.metric,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)


if train_args.do_train:
    # Model training
    print('waiwai 1')
    trainer.train(
        resume_from_checkpoint=train_args.resume_from_checkpoint
    )
    print('waiwai 2')

    print(trainer.state.best_model_checkpoint)


if train_args.do_predict:
    # Predict with the best model
    print('waiwai 3')
    for question_type in ['human','gpt','template']: # crossvalidation 
        now_type = Path(data_args.test_dir).parent.name
        test_dir = data_args.test_dir.replace(now_type,question_type)
        iid_test_dataset = Seq2SeqDataset(test_dir, data_args, tokenizer, "test")
        ood_test_dataset = Seq2SeqDataset(test_dir.replace('small_iid_test','small_ood_test'), data_args, tokenizer, "test")
        for phase, dataset in zip(['small_iid_test','small_ood_test'],[iid_test_dataset, ood_test_dataset]):
            test_dataloader = DataLoader(
                dataset = dataset,
                batch_size = train_args.per_device_eval_batch_size,
                collate_fn = dataset.collate_fn
            )
            trainer.model.eval()
            res = trainer.predict(test_dataset = dataset)
            
            prediction_result = tokenizer.batch_decode(res.predictions, skip_special_tokens = True)
            
            test_set = json.load(open(test_dir.replace('small_iid_test',phase),'r'))
            for q, pred in zip( test_set,prediction_result):
                q['pred'] = pred 
            print(f'prediction file is {os.path.join(train_args.output_dir, f"{question_type}_{phase}_prediction.json")}')
            predict_output_file = open(os.path.join(train_args.output_dir, f"{question_type}_{phase}_prediction.json") , "w")
            json.dump(test_set, predict_output_file,indent=4,ensure_ascii=False)