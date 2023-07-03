"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import logging
import faiss
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.distributed as torch_distrib
from pytorch_lightning.plugins.training_type import DDPPlugin
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagConfig,
    # RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
    DPRQuestionEncoder
)
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available


# if is_ray_available():
#     import ray
#     from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever

from callbacks_rag import (  # noqa: E402 # isort:skipq
    get_checkpoint_callback,
    get_early_stopping_callback,
    Seq2SeqLoggingCallback,
)

from distributed_pytorch_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip
from utils_rag import (  # noqa: E402 # isort:skip
    calculate_exact_match,
    flatten_list,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
    add_model_specific_args,
    add_retriever_specific_args,
    add_ray_specific_args,
    add_openqa_specific_args,
)

from openqa_dataset import OpenQACollator
from RagWithContext import RagSequenceWithContext as RagSequenceForGeneration

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, generic_train  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class CustomDDP(DDPPlugin):
    def init_ddp_connection(self, global_rank=None, world_size=None) -> None:
        module = self.model
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        if not torch.distributed.is_initialized():
            logger.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

        if module.is_rag_model:
            self.distributed_port = module.hparams.distributed_port
            if module.distributed_retriever == "pytorch":
                module.model.rag.retriever.init_retrieval(self.distributed_port)
                try:
                    module.model.rag.retriever_me.init_retrieval(self.distributed_port)
                except:
                    pass
            # elif module.distributed_retriever == "ray" and global_rank == 0:
            #     # For the Ray retriever, only initialize it once when global
            #     # rank is 0.
            #     module.model.rag.retriever.init_retrieval()
            #     try:
            #         module.model.rag.retriever_me.init_retrieval()
            #     except:
            #         pass

class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["loss"]
    metric_names = ["acc","em_acc_with_penalty","token_f1_with_penalty"]
    val_metric = "acc"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)
        self.multi_retrieve = hparams.multi_retrieve

        config_class = RagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(
            hparams.model_name_or_path, 
            local_files_only=False,
            # set retriever parameters
            index_name = hparams.index_name,
            passages_path = hparams.passages_path,
            index_path = hparams.index_path,
            use_dummy_dataset = hparams.use_dummy_dataset,
            max_combined_length = hparams.max_combined_length,
        )

        
        # add by yanto to set the max context length
        logger.info(f'-'*100)
        logger.info('here is the max combined length')
        logger.info(hparams.max_combined_length)
        logger.info(hparams)
        
        self.no_eval = hparams.no_eval

        if self.multi_retrieve:
            config.n_docs = 2 * int(config.n_docs // 2)

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            if hparams.distributed_retriever == "pytorch":
                retriever = RagPyTorchDistributedRetriever.from_pretrained(hparams.model_name_or_path, config=config)
            # elif hparams.distributed_retriever == "ray":
            #     # The Ray retriever needs the handles to the retriever actors.
            #     retriever = RagRayDistributedRetriever.from_pretrained(
            #         hparams.model_name_or_path, hparams.actor_handles, config=config
            #     )
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config, retriever=retriever)
            prefix = config.question_encoder.prefix

        tokenizer = (
            RagTokenizer.from_pretrained(hparams.model_name_or_path, local_files_only=False)
            if self.is_rag_model
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path, local_files_only=False)
        )
                        
        # add by yantao to include special tokens
        logger.info('add special tokens <ans>')
        tokenizer.generator.add_tokens('<ans>',special_tokens=True)
        model.generator.resize_token_embeddings(len(tokenizer.generator))

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)

        # save_git_info(self.hparams.output_dir)
        self.output_dir = Path(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        # self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "pytorch":
                self.model.retriever.init_retrieval(self.distributed_port)
            # elif hparams.distributed_retriever == "ray":
            #     self.model.retriever.init_retrieval()

        self.distributed_retriever = hparams.distributed_retriever
        
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int], question_encoder=False):
        if question_encoder:
            gen_text = self.tokenizer.question_encoder.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        else:
            gen_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        question_context_input_ids = batch['question_context_input_ids']
        question_context_attention_mask = batch['question_context_attention_mask']
        question_context_embs = batch['question_context_embs']
        # question_context_splits = batch['question_context_splits']
        

        rag_kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
        else:
            assert self.is_rag_model
            generator = self.model.rag.generator
            if isinstance(generator, T5ForConditionalGeneration):
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                        dim=1,
                    ) # this step is used to add start_token to at head of each tgt_ids
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = decoder_input_ids
            rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None
        assert question_context_embs is not None
        assert question_context_attention_mask is not None
        assert question_context_input_ids is not None

        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            question_context_attention_mask=question_context_attention_mask,
            question_context_input_ids=question_context_input_ids,
            question_context_embs=question_context_embs,
            use_cache=False,
            labels=lm_labels,
            **rag_kwargs,
        )

        loss = outputs["loss"]
        return (loss,)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:
        if batch_idx == 0:
            print('All training Step: ', self.trainer.num_training_batches)

        if self.hparams.save_step != -1 and batch_idx % self.hparams.save_step == self.hparams.save_step - 1:
            self.save_checkpoint_manual(batch_idx)
        if self.hparams.save_step == -1 and batch_idx == self.trainer.num_training_batches - 1:
            self.save_checkpoint_manual('last')
            
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
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
        logs["tpb"] = (
            batch["input_ids"].ne(src_pad_token_id).sum() + batch["decoder_input_ids"].ne(tgt_pad_token_id).sum()
        )

        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        logger.info('here is the validation_epoch_end')
        # logger.info(outputs.keys())
        logger.info(outputs[-1])
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metrics_tensor: torch.FloatTensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        # fix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2424
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor = metrics_tensor / dist.get_world_size()
            gen_metrics.update({self.val_metric: metrics_tensor.item()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": metrics_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target,self.output_dir)

    def _generative_step(self, batch: dict) -> dict:
        start_time = time.time()
        batch = BatchEncoding(batch).to(device=self.model.device)
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            question_context_input_ids=batch["question_context_input_ids"],
            question_context_attention_mask=batch["question_context_attention_mask"],
            question_context_embs=batch["question_context_embs"],
            do_deduplication=False,  # rag specific parameter
            use_cache=True,
            min_length=1,
            max_length=self.target_lens["val"],
        )

        gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        gen_metrics: Dict = self.calc_generative_metrics(preds, target)

        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **gen_metrics)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path):
        dataset = load_from_disk(Path(self.dataset_kwargs['data_dir']).joinpath(type_path+'_dataset'))
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=OpenQACollator(self.tokenizer).collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def transfer_batch_to_device(self, batch, device):
        for k in ['input_ids', 'attention_mask', 'decoder_input_ids', 'question_context_input_ids', 'question_context_attention_mask', 'question_context_embs']:
            batch[k] = batch[k].to(device)
        return batch

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("small_iid_test", batch_size=self.hparams.eval_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def save_checkpoint_manual(self, idx):
        if self.global_rank == 0 or self.global_rank == -1:
            save_path = self.output_dir.joinpath("checkpoint-{}-{}".format(self.step_count, idx))
            save_path_question_encoder = self.output_dir.joinpath("checkpoint-{}-{}/question_encoder_model".format(self.step_count, idx))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            if self.hparams.index_name == 'custom':
                self.model.question_encoder.save_pretrained(save_path_question_encoder)



def main(args=None, model=None) -> GenerativeQAModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    named_actors = []
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: GenerativeQAModule = GenerativeQAModule(args)

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        training_logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        training_logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        training_logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    es_callback = (
        get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        if args.early_stopping_patience >= 0
        else False
    )

    logger.info('Finished initialize the primary model')
    logger.info('prepare the DDP')
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        logger=training_logger,
        custom_ddp_plugin=CustomDDP(),
        profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
        val_check_interval=args.val_check_interval
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    logger.info('DDP finished!')

    if not args.do_predict:
        return model

    # trainer.validate(model)
    # test() without a model tests using the best checkpoint automatically
    orgin_output_dir = model.output_dir
    model.output_dir = os.path.join(orgin_output_dir,'small_iid_result')
    model.metrics_save_path = Path(model.output_dir) / "metrics.json"
    model.hparams_save_path = Path(model.output_dir) / "hparams.pkl"  
    trainer.test(model,model.get_dataloader(type_path = 'small_iid_test',batch_size = 64))

    model.output_dir = os.path.join(orgin_output_dir,'small_ood_result')
    model.metrics_save_path = Path(model.output_dir) / "metrics.json"
    model.hparams_save_path = Path(model.output_dir) / "hparams.pkl" 
    trainer.test(model,model.get_dataloader(type_path = 'small_ood_test',batch_size = 64))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_model_specific_args(parser, os.getcwd())
    parser = add_retriever_specific_args(parser)
    parser = add_ray_specific_args(parser)
    parser = add_openqa_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )
    logger.info(f'the Pid of this process is {os.getpid()}')
    args = parser.parse_args()

    main(args)