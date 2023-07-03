# export CUDA_VISIBLE_DEVICES="4,5,6,7"
cd ..
# itself dataset
mkdir ../dataset/rag/human/seq1e-5/small_iid_result
mkdir ../dataset/rag/human/seq1e-5/small_ood_result
TRANSFORMERS_OFFLINE=1 python finetune_rag.py \
    --data_dir ../dataset/rag/human \
    --output_dir ../dataset/rag/human/seq1e-5 \
    --model_name_or_path /data0/lyt/model/rag-sequence-base \
    --index_name custom \
    --passages_path wiki_dpr/lvxin/my_knowledge_dataset \
    --index_path wiki_dpr/lvxin/my_knowledge_dataset_hnsw_index.faiss \
    --model_type rag_sequence \
    --gpus 8 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --num_beams 32 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --max_source_length 512 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --test_max_target_length 64 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 1e-05 \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --save_step -1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --logger_name wandb \
    --max_combined_length 200 \
