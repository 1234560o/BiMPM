import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# file and requirement parameters
flags.DEFINE_string("task_name", "Snli", "The name of the task to train.")

flags.DEFINE_string(
    "model_config_file", "config.json",
    "The config json file corresponding to the trained BiMPM model which in 'output_dir'. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_dir", "data/Snli",
    "The dataset directory where train、dev、test set in")

flags.DEFINE_string("vec_file", "data/word2vec.txt", "Pre-training word2vec")

flags.DEFINE_string("vocab_file", "vocab.txt", "The path of vocab file which in the 'output_dir'")

flags.DEFINE_string("train_set", "train.tsv", "The train set path in the data_dir")

flags.DEFINE_string("dev_set", "dev.tsv", "The dev set path in the data_dir")

flags.DEFINE_string("test_set", "test.tsv", "The test set path in the data_dir")

flags.DEFINE_string(
    "output_dir", "output_dir/Snli",
    "The output directory where the model checkpoints and tensorboard will be written.")

# model parameters
flags.DEFINE_integer(
    "seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("num_classes", 3, "Numbers of Categories")

flags.DEFINE_integer("vocab_size", 30000, "Numbers of vocab")

flags.DEFINE_integer("embedding_size", 300, "Numbers of vocab")

flags.DEFINE_integer("context_hidden_size", 128, "Context layer hidden dimension")

flags.DEFINE_integer("agg_hidden_size", 128, "Aggregation layer hidden dimension")

flags.DEFINE_integer("num_perspective", 8, "Num of perspectives")

flags.DEFINE_integer("connection_size", 128, "Size of dense layer for last second output")

flags.DEFINE_float("dropout_prob", 0.1, "Dropout rate")

flags.DEFINE_float("initializer_range", 0.02, "Initializer_range")

# training parameters

flags.DEFINE_string("checkpoint_dir", "checkpoint", "The model save dir in the output_dir")

flags.DEFINE_string("tensorboard_dir", "tensorboard", "The tensorboard save dir in the output_dir")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_dev", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", False, "Whether to run eval on the test set.")

flags.DEFINE_bool("use_pre_vec", True, "Whether to use pre vec.")

flags.DEFINE_bool("is_pre_vec_tuning", True, "Whether to fine-tuning pre_word_embedding.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 0.005, "The initial learning rate for Adam.")

tf.flags.DEFINE_integer("decay_step", 100, "Decay step")

tf.flags.DEFINE_float("decay_rate", 0.95, "Decay_rate of learning_rate")

flags.DEFINE_integer("num_epochs", 10, "Number of epoch.")

flags.DEFINE_integer("print_train_steps", 100, "Print loss and acc of train steps.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_integer("save_per_batch", 10, "How many steps write to tensorboard")

flags.DEFINE_integer("require_improvement", 50000, "How many steps that not improved on dev set to early stop.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
