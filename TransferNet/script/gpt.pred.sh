export CUDA_VISIBLE_DEVICES=5
TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/human \
    --ckpt data/rush/gpt/ckpt/model-30-0.2293.pt \
    --output_dir data/rush/gpt \
    --kg_name wyf50 \
    --mode val 

# TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
#     --input_dir data/rush/gpt \
#     --ckpt data/rush/gpt/ckpt/model-30-0.2293.pt \
#     --output_dir data/rush/gpt \
#     --kg_name wyf50 \
#     --mode val 

# TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
#     --input_dir data/rush/template \
#     --ckpt data/rush/gpt/ckpt/model-30-0.2293.pt \
#     --output_dir data/rush/gpt \
#     --kg_name wyf50 \
#     --mode val 


