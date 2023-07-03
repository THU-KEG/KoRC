export CUDA_VISIBLE_DEVICES=0
cd ..
python -m WebQSP.train \
    --input_dir data/rush/human \
    --save_dir data/rush/human/ckpt \
    --kg_name wyf50 \
    --num_epoch 32 \
    --val_epoch 1 \
    --batch_size 12