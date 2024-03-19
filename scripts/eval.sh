GENERAL_TINYBERT_DIR="../data/checkpoints/bibert/binarized/cola"
GENERAL_TINYBERT_DIR="../data/checkpoints/bibert/binarized/sst-2"
GENERAL_TINYBERT_DIR="./results/biBERT/rte/rte/"
GENERAL_TINYBERT_DIR="./results/biBERT/mrpc/mrpc/"

TASK_DIR="../data/datasets/glue"
TASK_NAME="RTE"
TASK_NAME="MRPC"

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
            --data_dir $TASK_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --weight_bits 1 \
            --embedding_bits 1 \
            --input_bits 1 \
            --batch_size 16 
