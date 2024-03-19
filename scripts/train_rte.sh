FT_BERT_BASE_DIR="../data/checkpoints/bibert/fp/dynabert/RTE"
GENERAL_TINYBERT_DIR="../data/checkpoints/bibert/fp/dynabert/RTE"

TASK_DIR="../data/datasets/glue"
TASK_NAME="RTE"

OUTPUT_DIR="results/biBERT/rte/"
LOGFILE_DIR="./results/biBERT/logs/"

LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

log_filepath=$LOGFILE_DIR$LOG_FILENAME"-rte.log"

mkdir $OUTPUT_DIR
#SEED=42
SEED=631
#SEED=927

# 03/18 05:19:28 PM ***** Running evaluation *****
# 03/18 05:19:28 PM   Num examples = 277
# 03/18 05:19:28 PM   Batch size = 16
# 03/18 05:19:29 PM ***** Eval results *****
# 03/18 05:19:29 PM   acc = 0.5415162454873647
# 03/18 05:19:29 PM   eval_loss = 0.6952206724219852



CUDA_VISIBLE_DEVICES=0 python3 quant_task_glue.py \
            --data_dir $TASK_DIR \
            --teacher_model $FT_BERT_BASE_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --seed $SEED \
            --learning_rate 1e-5 \
            --weight_bits 1 \
            --embedding_bits 1 \
            --input_bits 1 \
            --batch_size 16 \
            --pred_distill \
            --intermediate_distill \
            --value_distill \
            --key_distill \
            --query_distill \
            --save_fp_model 2>&1 | tee ${log_filepath}
