import argparse
import logging
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from transformer import BertConfig, BertTokenizer
from transformer.modeling_quant import (
    BertForSequenceClassification as QuantBertForSequenceClassification,
)
from utils_glue import *

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths
    )
    return tensor_data, all_label_ids


def do_eval(
    model, task_name, eval_dataloader, device, output_mode, eval_labels, num_labels
):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for _, batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _, _, _, _, _ = model(input_ids, segment_ids, input_mask)

        if output_mode == "classification":
            loss_fct = nn.CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = nn.MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result["eval_loss"] = eval_loss
    return result


# load argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default="data",
    type=str,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--model_dir", default="models/tinybert", type=str, help="The model dir."
)
parser.add_argument(
    "--student_model", default=None, type=str, help="The models directory."
)
parser.add_argument(
    "--task_name", default="sst-2", type=str, help="The name of the task to train."
)
parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
parser.add_argument(
    "--weight_bits", default=1, type=int, help="Quantization bits for weight."
)
parser.add_argument(
    "--embedding_bits", default=1, type=int, help="Quantization bits for embedding."
)
parser.add_argument(
    "--input_bits", default=1, type=int, help="Quantization bits for activation."
)
parser.add_argument("--clip_val", default=2.5, type=float, help="Initial clip value.")

args = parser.parse_args()


# load tokenizer
tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=True)

# load dataset
processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
}

default_params = {
    "cola": {
        "num_train_epochs": 50,
        "max_seq_length": 64,
        "batch_size": 16,
        "eval_step": 500,
    },
    "mnli": {
        "num_train_epochs": 5,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000,
    },
    "mrpc": {
        "num_train_epochs": 20,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 200,
    },
    "sst-2": {
        "num_train_epochs": 10,
        "max_seq_length": 64,
        "batch_size": 32,
        "eval_step": 200,
    },
    "sts-b": {
        "num_train_epochs": 20,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 200,
    },
    "qqp": {
        "num_train_epochs": 5,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000,
    },
    "qnli": {
        "num_train_epochs": 10,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000,
    },
    "rte": {
        "num_train_epochs": 20,
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 200,
    },
}

acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]

logger.info("The args: {}".format(args))
task_name = args.task_name.lower()
data_dir = os.path.join(args.data_dir, args.task_name)

if args.batch_size is None:
    args.batch_size = default_params[task_name]["batch_size"]

if task_name in default_params:
    args.max_seq_length = default_params[task_name]["max_seq_length"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

processor = processors[task_name]()
output_mode = output_modes[task_name]
label_list = processor.get_labels()
num_labels = len(label_list)

eval_examples = processor.get_dev_examples(data_dir)
eval_features = convert_examples_to_features(
    eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
)

eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(
    eval_data, sampler=eval_sampler, batch_size=args.batch_size
)

# load config and model
student_config = BertConfig.from_pretrained(
    args.student_model,
    quantize_act=True,
    weight_bits=args.weight_bits,
    embedding_bits=args.embedding_bits,
    input_bits=args.input_bits,
    clip_val=args.clip_val,
)
student_model = QuantBertForSequenceClassification.from_pretrained(
    args.student_model, config=student_config, num_labels=num_labels
)
student_model.to(device)

# run evaluation on dev set
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", args.batch_size)

student_model.eval()
result = do_eval(
    student_model,
    task_name,
    eval_dataloader,
    device,
    output_mode,
    eval_labels,
    num_labels,
)

logger.info("***** Eval results *****")
for key in sorted(result.keys()):
    logger.info("  %s = %s", key, str(result[key]))
