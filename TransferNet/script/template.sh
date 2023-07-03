export CUDA_VISIBLE_DEVICES=2
python -m WebQSP.train \
    --input_dir data/rush/template \
    --save_dir data/rush/template/ckpt \
    --kg_name wyf50 \
    --num_epoch 32 \
    --val_epoch 1 \
    --batch_size 12