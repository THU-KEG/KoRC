export CUDA_VISIBLE_DEVICES=6
TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/human \
    --ckpt data/rush/human/ckpt/model-31-0.2312.pt \
    --output_dir data/rush/human \
    --kg_name wyf50 \
    --mode val 

TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/gpt \
    --ckpt data/rush/human/ckpt/model-31-0.2312.pt \
    --output_dir data/rush/human \
    --kg_name wyf50 \
    --mode val 

# TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
#     --input_dir data/rush/template \
#     --ckpt data/rush/human/ckpt/model-31-0.2312.pt \
#     --output_dir data/rush/human \
#     --kg_name wyf50 \
#     --mode val 

