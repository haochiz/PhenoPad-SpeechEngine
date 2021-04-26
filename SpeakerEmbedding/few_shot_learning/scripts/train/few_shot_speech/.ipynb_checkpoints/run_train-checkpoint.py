import argparse

from train import main

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'vctk'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
'''
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
'''
parser.add_argument('--data.way', type=int, default=15, metavar='WAY',
                    help="number of classes per episode (default: 15)")
parser.add_argument('--data.test_way', type=int, default=15, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 15)")
parser.add_argument('--data.duration', type=float, default=2, metavar='DURATION',
                    help="fix durations of audio segments (default: 2.0s)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=0, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 0)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
parser.add_argument('--data.min_len', type=int, default=100, metavar='MINLEN',
                    help="minimal length of input sequence (default: 100)")
parser.add_argument('--data.max_len', type=int, default=500, metavar='MAXLEN',
                    help="maximal length of input sequence (default: 500)")

# model args
default_model_name = 'few_shot_speech'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.in_dim', type=str, default='1,500,59', metavar='INDIM',
                    help="dimensionality of input audio segments (default (max: 5s): '1,500,59')")
parser.add_argument('--model.out_dim', type=int, default=16, metavar='OUTDIM',
                    help="dimensionality of embedding (default: 16)")
parser.add_argument('--model.n_rnn', type=int, default=1, metavar='NUMRNN',
                    help="number of stacked RNN layers (default: 1)")
parser.add_argument('--model.gpu_num', type=int, default=1, metavar='GPUNUM',
                    help='number of GPU used for training (default: 1)')

# train args
parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--train.checkpoint', type=str, default=None, metavar='CHECKPOINT',
                    help='checkpoint which to start with')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

args = vars(parser.parse_args())

main(args)
