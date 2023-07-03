cd ..
# export TRANSFORMERS_CACHE=/data0/lyt/transformers/
# echo $TRANSFORMERS_CACHE

export TRANSFORMERS_OFFLINE=1

MODELTYPE='t5'
MODEL='/data0/lyt/model/flan-t5-xxl'
DATASETPATH='../dataset/mrc/human'
CONF='config.single.single'
METRIC='acc'
MODELID='mrc'



deepspeed --include=localhost:4,5,6,7 main.py \
--deepspeed ds_config.json \
--model_type $MODELTYPE \
--model_name $MODEL \
--tokenizer_name $MODEL \
--model_id $MODELID \
--conf_dir $CONF \
--add_special_token \
--output_dir "${DATASETPATH}/${MODEL}" \
--save_total_limit 4 \
--train_dir "${DATASETPATH}/train.json" \
--eval_dir "${DATASETPATH}/eval.json" \
--test_dir "${DATASETPATH}/small_iid_test.json" \
--max_input_length 512 \
--max_output_length 128 \
--num_train_epochs 4 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 32 \
--generation_num_beams 1 \
--predict_with_generate \
--report_to wandb \
--metric_for_best_model $METRIC \
--greater_is_better True \
--load_best_model_at_end True \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 600 \
--save_steps 600 \
--logging_steps 10 \
--warmup_ratio 0.1 \
--learning_rate 1e-5 \
--do_train