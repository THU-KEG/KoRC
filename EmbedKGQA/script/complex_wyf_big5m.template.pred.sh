# relation dim should be half of entity embedding
cd ..
TRANSFORMERS_OFFLINE=1 python3 main.py \
    --mode eval \
    --relation_dim 256 \
    --do_batch_norm 1 \
    --gpu 3 \
    --freeze 0 \
    --batch_size 8 \
    --validate_every 1 \
    --hops complex_wyf_big5m \
    --lr 0.00002 \
    --entdrop 0.0 \
    --reldrop 0.0 \
    --scoredrop 0.0 \
    --decay 1.0 \
    --model ComplEx \
    --patience 20 \
    --ls 0.05 \
    --l3_reg 0.001 \
    --nb_epochs 500 \
    --question_type template \
    --output_dir checkpoints/roberta_finetune/template \
    --outfile complex_wyf_big5m_template