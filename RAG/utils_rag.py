import itertools
import json
import linecache
import os
import pickle
import re
import socket
import string
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import Levenshtein
import git
import torch
from torch.utils.data import Dataset
from config.single import Metric
from transformers import BartTokenizer, RagTokenizer, T5Tokenizer
from lightning_base import BaseTransformer

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




logger = getLogger(__name__)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "hostname": str(socket.gethostname()),
    }
    return repo_infos


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def find_max_sim(name, des2ent):
    new_dic = {}
    for index in des2ent:
        new_dic[index] = float(Levenshtein.distance(name, index))
    l = sorted(new_dic.items(), key=lambda x : x[1])
    return des2ent[l[0][0]]

def exact_match_score(prediction, ground_truth):
    ans = normalize_answer(prediction) == normalize_answer(ground_truth)
    return ans

def calculate_exact_match(output_lns: List[str], reference_lns: List[str],output_dir) -> Dict:
    assert len(output_lns) == len(reference_lns)
    # em = 0
    with open(os.path.join(output_dir,'output.txt'),'a+') as f:
        pred_str_list = []
        tgt_str_list = []
        em_cnt = 0
        for pred, tgt in zip(output_lns, reference_lns):
            em_cnt = em_cnt + (pred == tgt)
            pred_str_list.append(pred.split('  '))
            tgt_str_list.append(tgt.split('  '))
            f.write('{}\t{}\n'.format(pred,tgt))

    metric_dict = Metric.str_metric(pred_str_list,tgt_str_list)
    if len(output_lns) > 0:
        metric_dict['acc'] = float(em_cnt/len(output_lns))
    # with open(os.path.join(output_dir,'metric.txt'),'a+') as f: 
    #     f.write(json.dumps(metric_dict)+'\n')
    # metric_dict.update({"em":em})
    return metric_dict

def is_rag_model(model_prefix):
    return model_prefix.startswith("rag")


def set_extra_model_params(extra_params, hparams, config):
    equivalent_param = {p: p for p in extra_params}
    # T5 models don't have `dropout` param, they have `dropout_rate` instead
    equivalent_param["dropout"] = "dropout_rate"
    for p in extra_params:
        if getattr(hparams, p, None):
            if not hasattr(config, p) and not hasattr(config, equivalent_param[p]):
                logger.info("config doesn't have a `{}` attribute".format(p))
                delattr(hparams, p)
                continue
            set_p = p if hasattr(config, p) else equivalent_param[p]
            setattr(config, set_p, getattr(hparams, p))
            delattr(hparams, p)
    return hparams, config

def add_generic_args(parser, root_dir) -> None:
    #  To allow all pl args uncomment the following line
    #  parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )


def add_model_specific_args(parser, root_dir):
    BaseTransformer.add_model_specific_args(parser, root_dir)
    add_generic_args(parser, root_dir)
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="num_beams.",
    )
    parser.add_argument(
        "--save_step",
        default=-1,
        type=int,
        help="save steps.",
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=25,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--val_max_target_length",
        default=25,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--test_max_target_length",
        default=25,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix added at the beginning of each text, typically used with T5-based models.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
        help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So  will effect it.",
    )
    parser.add_argument(
        "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
    )
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart", "t5"],
        type=str,
        help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
    )
    return parser

def add_retriever_specific_args(parser):
    parser.add_argument(
        "--index_name",
        type=str,
        default=None,
        help="Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom' for a local index, or 'legacy' for the orignal one)",
    )
    parser.add_argument(
        "--passages_path",
        type=str,
        default=None,
        help="Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--max_combined_length",
        type=int,
        default=300,
        help="Max length of contextualized input returned by __call__()",
    )
    parser.add_argument(
        "--from_chkp_path",
        type=str,
        default=None,
        help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--dpr_encoder_path",
        type=str,
        default=None,
        help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--distributed_retriever",
        choices=["ray", "pytorch"],
        type=str,
        default="pytorch",
        help="What implementation to use for distributed retriever? If "
        "pytorch is selected, the index is loaded on training "
        "worker 0, and torch.distributed is used to handle "
        "communication between training worker 0, and the other "
        "training workers. If ray is selected, the Ray library is "
        "used to create load the index on separate processes, "
        "and Ray handles the communication between the training "
        "workers and the retrieval actors.",
    )
    parser.add_argument(
        "--use_dummy_dataset",
        type=bool,
        default=False,
        help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--no_eval",
        type=bool,
        default=False,
        help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--full_init",
        type=bool,
        default=False,
        help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    parser.add_argument(
        "--multi_retrieve",
        type=bool,
        default=False,
        help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
    )
    return parser

def add_ray_specific_args(parser):
    # Ray cluster address.
    parser.add_argument(
        "--ray-address",
        default="auto",
        type=str,
        help="The address of the Ray cluster to connect to. If not "
        "specified, Ray will attempt to automatically detect the "
        "cluster. Has no effect if pytorch is used as the distributed "
        "retriever.",
    )
    parser.add_argument(
        "--num_retrieval_workers",
        type=int,
        default=1,
        help="The number of retrieval actors to use when Ray is selected"
        "for the distributed retriever. Has no effect when "
        "distributed_retriever is set to pytorch.",
    )
    return parser

def add_openqa_specific_args(parser):
    parser.add_argument(
        "--dpr_ctx_encoder_model_name",
        type=str,
        default="facebook/dpr-ctx_encoder-multiset-base",
        help=(
            "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
            " 'facebook/dpr-ctx_encoder-multiset-base'"
        ),
    )
    return parser
