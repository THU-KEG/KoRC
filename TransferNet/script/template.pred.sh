export CUDA_VISIBLE_DEVICES=7
TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/human \
    --ckpt data/rush/template/ckpt/model-31-0.2547.pt \
    --output_dir data/rush/template \
    --kg_name wyf50 \
    --mode val 

TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/gpt \
    --ckpt data/rush/template/ckpt/model-31-0.2547.pt \
    --output_dir data/rush/template \
    --kg_name wyf50 \
    --mode val 

TRANSFORMERS_OFFLINE=1 python -m WebQSP.predict \
    --input_dir data/rush/template \
    --ckpt data/rush/template/ckpt/model-31-0.2547.pt \
    --output_dir data/rush/template \
    --kg_name wyf50 \
    --mode val 


