#coding=utf8
import argparse
import random
import numpy
import properties_loader
import sys
import torch
import collections
from utils import *

DIR="./data/"
parser = argparse.ArgumentParser(description="Experiemts\n")
parser.add_argument("-data",default = DIR, required=True,type=str, help="saved vectorized data")
parser.add_argument("-raw_data",default = "./data/zp_data/", type=str, help="raw_data")
parser.add_argument("-bert_dir",default = "/home/miaojingjing/data/Attention_bert/BertPretrainedModel/chinese_L-12_H-768_A-12/",type=str, help="saved BERT model")
parser.add_argument("-props",default = "./properties/prob", type=str, help="properties")
parser.add_argument("-reduced",default = 0, type=int, help="reduced")
parser.add_argument("-gpu",default = 0, type=int, help="GPU number")
parser.add_argument("-random_seed",default=0,type=int,help="random seed")

## Fine tune Required parameters
# parser.add_argument("--data_dir",
#                     default=None,
#                     type=str,
#                     required=True,
#                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
# parser.add_argument("--bert_model", default=None, type=str, required=True,
#                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                          "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
# parser.add_argument("--task_name",
#                     default=None,
#                     type=str,
#                     required=True,
#                     help="The name of the task to train.")
# parser.add_argument("--output_dir",
#                     default=None,
#                     type=str,
#                     required=True,
#                     help="The output directory where the model checkpoints will be written.")

## Other parameters
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=False,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
# parser.add_argument("--train_batch_size",
#                     default=32,
#                     type=int,
#                     help="Total batch size for training.")
# parser.add_argument("--eval_batch_size",
#                     default=8,
#                     type=int,
#                     help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=50,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument('--optimize_on_cpu',
                    default=False,
                    action='store_true',
                    help="Whether to perform optimization and keep the optimizer averages on CPU")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=128,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')


parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")


parser.add_argument("--layers", default="-1", type=str)
parser.add_argument("--batch_size", default=10, type=int, help="Batch size for predictions.")
parser.add_argument("--data_batch_size", default=10, type=int, help="Batch size for generating bert output.")
parser.add_argument("--max_sent_len", default=400, type=int, help="max sentence length.")
args = parser.parse_args()

# random.seed(0)
# numpy.random.seed(0)
# torch.manual_seed(args.random_seed)
# torch.cuda.manual_seed(args.random_seed)
nnargs = properties_loader.read_pros(args.props)#from prob
