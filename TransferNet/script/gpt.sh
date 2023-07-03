export CUDA_VISIBLE_DEVICES=1
python -m WebQSP.train \
    --input_dir data/rush/gpt \
    --save_dir data/rush/gpt/ckpt \
    --kg_name wyf50 \
    --num_epoch 32 \
    --val_epoch 1 \
    --batch_size 12