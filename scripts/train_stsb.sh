FT_BERT_BASE_DIR="../data/checkpoints/bibert/fp/dynabert/STS-B"
GENERAL_TINYBERT_DIR="../data/checkpoints/bibert/fp/dynabert/STS-B"

TASK_DIR="../data/datasets/glue"
TASK_NAME="STS-B"

OUTPUT_DIR="./results/biBERT/sts-b/"
LOGFILE_DIR="./results/biBERT/logs/"

LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir $OUTPUT_DIR
SEED=42

CUDA_VISIBLE_DEVICES=0 python3 quant_task_glue.py \
            --data_dir $TASK_DIR \
            --teacher_model $FT_BERT_BASE_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --seed $SEED \
            --learning_rate 1e-4 \
            --weight_bits 1 \
            --embedding_bits 1 \
            --input_bits 1 \
            --batch_size 16 \
            --pred_distill \
            --intermediate_distill \
            --value_distill \
            --key_distill \
            --query_distill \
            --save_fp_model
