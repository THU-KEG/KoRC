from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field, fields
from importlib_metadata import metadata
import torch
from torch.utils.data import dataset

from transformers import Seq2SeqTrainingArguments as HfTrainingArguments

# Define arguments for Model, Data, and Training



@dataclass
class ModelArguments:
    conf_dir: str = field(
        metadata={"help": "directory of the configuration file, it should include the evaluation metric function, and the special token list (Optional)"}
    )
    model_type: str = field(
        default="t5",
        metadata={"help": "choose from t5 and bart"}
    )
    model_name: str = field(
        default="t5-base",
        metadata={"help": "Specify Which model we use, T5 or BART"}
    )
    tokenizer_type: str = field(
        default=None,
        metadata={"help": "choose from t5 and bart"}
    )
    tokenizer_name: str = field(
        default="t5-base",
        metadata={"help": "Specify Which tokenizer we use, T5 or BART"}
    )
    model_id: str = field(
        default="-1e-5",
        metadata={"help": "Model ID to avoid output races"}
    )
    add_special_token: bool = field(
        default=False,
        metadata={"help": "Whether Need to add special tokens importing from the configuration file"}
    )

@dataclass
class DataArguments:
    train_dir: str = field(
        metadata={"help": "training set directory"}
    )
    eval_dir: str = field(
        metadata={"help": "validation set directory"}
    )
    test_dir: str = field(
        metadata={"help": "test set directory"}
    )
    max_input_length: int = field(
        default=128, 
        metadata={"help": "max length of input sequence after tokenization"}
    )
    max_output_length: int = field(
        default=256, 
        metadata={"help": "max length of input sequence after tokenization"}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    # Turn off train/test
    seed: int = field(
        default=42,
        metadata={"help": "set seed for reproducibility"}
    )
    score_threshold: float = field(
        default=-0.2,
        metadata={"help": "set score_threshold for output ans"}
    )
    special_tokens: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "special tokens"}
    )