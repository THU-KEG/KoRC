from transformers import RagTokenizer, RagRetriever, RagModel
import torch
from torch import nn

from typing import Callable, List, Optional, Tuple, Union
from transformers.models.rag.modeling_rag import RetrievAugLMOutput,RetrievAugLMMarginOutput
from transformers.models.rag.configuration_rag import RagConfig
from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers import PretrainedConfig,PreTrainedModel
from transformers import RagSequenceForGeneration,RagTokenForGeneration
from transformers import RagRetriever


class RagWithContext(RagModel):
    # def __init__(
    #     self,
    #     config: Optional[PretrainedConfig] = None,
    #     question_encoder: Optional[PreTrainedModel] = None,
    #     generator: Optional[PreTrainedModel] = None,
    #     retriever: Optional[RagRetriever] = None,  # or maybe just use a `set_retriever(...)` method
    #     **kwargs,
    # ):
       
    #     super().__init__(config,question_encoder,generator,retriever,kwargs)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_context_input_ids:Optional[torch.LongTensor] = None,
        question_context_attention_mask:Optional[torch.LongTensor] = None,
        question_context_embs:Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        n_docs: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], RetrievAugLMOutput]:
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:

            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="pt",
                )
                if self.context_encoder_training:
                    raise NotImplementedError("You must implement this for your task")
                    # (
                    #     context_input_ids,
                    #     context_attention_mask,
                    #     retrieved_doc_embeds,
                    #     retrived_doc_input_ids,
                    #     retrived_doc_attention_mask,
                    #     retrieved_doc_ids,
                    # ) = (
                    #     retriever_outputs["context_input_ids"],
                    #     retriever_outputs["context_attention_mask"],
                    #     retriever_outputs["retrieved_doc_embeds"],
                    #     retriever_outputs["tokenized_doc_ids"],
                    #     retriever_outputs["tokenized_doc_attention_mask"],
                    #     retriever_outputs["doc_ids"],
                    # )

                    # context_input_ids = context_input_ids.to(input_ids)
                    # context_attention_mask = context_attention_mask.to(input_ids)

                    # retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    # retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    # retrieved_doc_embeds = self.ctx_encoder(
                    #     retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True
                    # ).pooler_output
                    # retrieved_doc_embeds = retrieved_doc_embeds.view(
                    #     -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    # )  # reshaping

                    # # compute doc_scores involving ctx_encoder
                    # doc_scores = torch.bmm(
                    #     question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    # ).squeeze(1

                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )
                    # azhe ?
                    # TODO: to(input_ids) may change the dtype
                    question_context_embs = question_context_embs.to(question_encoder_last_hidden_state)
                    question_context_input_ids = question_context_input_ids.to(input_ids)
                    question_context_attention_mask = question_context_attention_mask.to(input_ids)


                    # set to correct device
                    retrieved_doc_embeds = retrieved_doc_embeds.to(input_ids)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    # print(question_context_embs.dtype,retrieved_doc_embeds.dtype)
                    # print(question_context_embs.shape,retrieved_doc_embeds.shape)
                    # print(question_context_input_ids.shape,context_input_ids.shape)
                    # print(question_context_attention_mask.shape,context_attention_mask.shape)
                    shape = retrieved_doc_embeds.shape # shape = (bsz,n_doc,dim)

                    # (bsz,n_doc+n_split,dim)
                    all_doc_embeds = torch.cat((question_context_embs,retrieved_doc_embeds),dim=1)
                    
                    # (bsz*(n_doc+n_split),dim)
                    all_context_input_ids = torch.cat((question_context_input_ids,context_input_ids.reshape(shape[0],shape[1],-1)),dim=1).reshape(2*shape[0]*shape[1],-1)
                    all_context_attention_mask = torch.cat((question_context_attention_mask,context_attention_mask.reshape(shape[0],shape[1],-1)),dim=1).reshape(2*shape[0]*shape[1],-1)
                    assert all_context_input_ids.shape == all_context_attention_mask.shape, print(all_context_input_ids.shape,all_context_attention_mask.shape)
                    # compute doc_scores

                    # question_encoder_last_hidden_state (bsz,dim) -> question_encoder_last_hidden_state.unsqueeze(1) (bsz,1,dim)
                    # all_doc_embeds (bsz,n_doc, dim) -> all_doc_embeds.transpose(1, 2) (bsz,dim,n_docs)
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), all_doc_embeds.transpose(1, 2)
                    ).squeeze(1)
            else:
                assert all_context_input_ids is not None, (
                    "Make sure that `all_context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert all_context_attention_mask is not None, (
                    "Make sure that `all_context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert ( all_context_input_ids.shape[0] % doc_scores.shape[1] ) == 0, (
            f" The first dimension of `all_context_input_ids` should be a multiple of `n_docs`={len(all_doc_embeds)}, but is"
            f" {all_context_input_ids.shape[0]}."
        )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(all_doc_embeds.shape[1], dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(all_doc_embeds.shape[1], dim=0)

        gen_outputs = self.generator(
            input_ids=all_context_input_ids,
            attention_mask=all_context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            all_context_input_ids = (None,)
            all_context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )

class RagSequenceWithContext(RagSequenceForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)
        self.question_ctx_split=5
        # instantiate model
        self.rag = RagWithContext(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        question_context_input_ids=None,
        question_context_attention_mask=None,
        question_context_embs=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        exclude_bos_score=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        **kwargs  # needs kwargs for generation
    ):
        # print('here is inside of RagSequenceWithContext ')
        # print(kwargs)
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
        
        assert question_context_embs is not None
        assert question_context_attention_mask is not None
        assert question_context_input_ids is not None

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_context_input_ids=question_context_input_ids,
            question_context_attention_mask=question_context_attention_mask,
            question_context_embs=question_context_embs,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=(n_docs+self.question_ctx_split),
            )

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        question_context_input_ids:Optional[torch.LongTensor] = None,
        question_context_attention_mask:Optional[torch.LongTensor] = None,
        question_context_embs:Optional[torch.LongTensor] = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        do_deduplication=None,  # defaults to True
        num_return_sequences=None,  # defaults to 1
        num_beams=None,  # defaults to 1
        n_docs=None,
        **model_kwargs
    ):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
            input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )["context_input_ids"]

            # set to correct device
            context_input_ids = context_input_ids.to(input_ids)

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)
            generator_input_ids = torch.cat((generator_input_ids,question_context_input_ids[index])) # append question ctx
            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam

            # then, run model forwards to get nll scores:
            if input_ids is not None:
                new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
                outputs = self(
                    new_input_ids, 
                    question_context_input_ids=question_context_input_ids[index].repeat(num_candidates, 1, 1),
                    question_context_attention_mask=question_context_attention_mask[index].repeat(num_candidates, 1, 1),
                    question_context_embs=question_context_embs[index].repeat(num_candidates, 1, 1),
                    labels=output_sequences, 
                    exclude_bos_score=True
                )
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert (
                    context_attention_mask is not None
                ), "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    doc_scores is not None
                ), "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

                individual_input_ids = generator_input_ids.repeat(
                    num_candidates, 1
                )  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]

                outputs = self(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )

            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        # drop the bos tokens
        # target (bsz,tgt_lens)
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size

        # doc_scores = (bsz,n_docs) -> doc_logprobs (bsz,n_docs,1,1)
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :] # (bsz,n_docs,1,vocab_size)
        second_token_scores = seq_logprobs[:, :, 1:2, :] # (bsz,n_docs,1,vocab_size)
        remainder = seq_logprobs[:, :, 2:, :] # (bsz,n_docs,tgt_len-2,vocab_size)
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2) # (bsz,n_docs,tgt_len,vocab)
        # calculate loss
        # target (bsz,n_docs,tgt_len,1)
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        # ll is short for likelihood loss
        ll = rag_logprobs.gather(dim=-1, index=target) # (bsz,n_docs,tgt_len,1)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits (bsz,n_docs,tgt_len,1)

        ll, smooth_obj = _mask_pads(ll, smooth_obj) # (bsz,n_docs,tgt_len) (bsz,n_docs,tgt_len)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2) # (bsz,n_docs)
        smooth_obj = smooth_obj.sum(2) # (bsz,n_docs)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

class RagTokenWithContext(RagTokenForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)
        self.question_ctx_split=5
        # instantiate model
        self.rag = RagWithContext(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_context_input_ids:Optional[torch.LongTensor] = None,
        question_context_attention_mask:Optional[torch.LongTensor] = None,
        question_context_embs:Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        do_marginalize: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        **kwargs  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Legacy dictionary, which is required so that model can use *generate()* function.
        Returns:
        Example:
        ```python
        >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
        >>> import torch
        >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> # or use retriever separately
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
        >>> # 1. Encode
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        >>> doc_scores = torch.bmm(
        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ... ).squeeze(1)
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=labels,
        ... )
        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_context_input_ids=question_context_input_ids,
            question_context_attention_mask=question_context_attention_mask,
            question_context_embs=question_context_embs,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=(n_docs+self.question_ctx_split),
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, (n_docs+self.question_ctx_split))

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        question_context_input_ids:Optional[torch.LongTensor] = None,
        question_context_attention_mask:Optional[torch.LongTensor] = None,
        question_context_embs:Optional[torch.LongTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        n_docs: Optional[int] = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        """
        Implements RAG token decoding.
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.
                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.
                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.
                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether or not to stop the beam search when at least `num_beams` sentences are finished per batch or
                not.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.
                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[int]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation_utils.GenerationMixin.generate`] function,
                where we set `num_return_sequences` to `num_beams`. decoder_start_token_id (`int`, *optional*): If an
                encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        exponential_decay_length_penalty = (
            exponential_decay_length_penalty
            if exponential_decay_length_penalty is not None
            else self.config.exponential_decay_length_penalty
        )

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

        # set to correct device
        retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)

        question_context_embs = question_context_embs.to(question_hidden_states)
        question_context_input_ids = question_context_input_ids.to(input_ids)
        question_context_attention_mask = question_context_attention_mask.to(input_ids)

        shape = retrieved_doc_embeds.shape # shape = (bsz,n_doc,dim)
        # (bsz,n_doc+n_split,dim)
        all_doc_embeds = torch.cat((question_context_embs,retrieved_doc_embeds),dim=1)
        
        # (bsz*(n_doc+n_split),dim)
        all_context_input_ids = torch.cat((question_context_input_ids,context_input_ids.reshape(shape[0],shape[1],-1)),dim=1).reshape(2*shape[0]*shape[1],-1)
        all_context_attention_mask = torch.cat((question_context_attention_mask,context_attention_mask.reshape(shape[0],shape[1],-1)),dim=1).reshape(2*shape[0]*shape[1],-1)
        assert all_context_input_ids.shape == all_context_attention_mask.shape, print(all_context_input_ids.shape,all_context_attention_mask.shape)
        
        # compute doc_scores
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), all_doc_embeds.transpose(1, 2)).squeeze(
            1
        )

        assert (context_input_ids.shape[0] % (n_docs + self.question_ctx_split)) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs+self.question_ctx_split`={n_docs + self.question_ctx_split}, but is"
            f" {context_input_ids.shape[0]}."
        )

        batch_size = all_context_input_ids.shape[0] // (self.question_ctx_split + n_docs)

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=all_context_input_ids, attention_mask=all_context_attention_mask, return_dict=True)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, (self.question_ctx_split + n_docs)) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, (self.question_ctx_split + n_docs)) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * (self.question_ctx_split + n_docs),) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        all_context_attention_mask = extend_enc_output(all_context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = all_context_attention_mask
        model_kwargs["(self.question_ctx_split + n_docs)"] = (self.question_ctx_split + n_docs)

        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=all_context_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")
