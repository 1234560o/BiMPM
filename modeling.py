import tensorflow as tf
import json
import six
import copy


class BiMPMConfig(object):
    """Configuration for `BiMPMModel`."""
    def __init__(self,
                 seq_length=128,
                 num_classes=2,
                 context_hidden_size=256,
                 agg_hidden_size=128,
                 num_perspective=16,
                 connection_size=128,
                 vocab_size=5000,
                 embedding_dim=300,
                 use_pre_vec=True,
                 is_pre_vec_tuning=True,
                 learning_rate=0.01,
                 decay_step=100,
                 decay_rate=0.97,
                 initializer_range=0.02,
                 scope="BiMPM"):
        """
        Constructs BiMPMConfig.
        Args:
            context_hidden_size: Size of the context representation layers.
            agg_hidden_size:
            num_perspective:
            connection_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the model.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.context_hidden_size = context_hidden_size
        self.agg_hidden_size = agg_hidden_size
        self.num_perspective = num_perspective
        self.connection_size = connection_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_pre_vec = use_pre_vec
        self.is_pre_vec_tuning = is_pre_vec_tuning
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.initializer_range = initializer_range
        self.scope = scope

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BiMPMConfig` from a Python dictionary of parameters."""
        config = BiMPMConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BiMPMConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with open(json_file, 'w', encoding='utf-8') as fout:
            json.dump(self.to_dict(), fout, indent=2, sort_keys=True)


class BiMPMModel(object):
    """BiMPM model ("Bilateral Multi-Perspective Matching for Natural Language Sentences")"""

    def __init__(self, config, embed_vec):

        self.input_a = tf.placeholder(tf.int32, [None, config.seq_length], name="input_a")
        self.input_b = tf.placeholder(tf.int32, [None, config.seq_length], name="input_b")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        config = copy.deepcopy(config)

        # Scalar dimensions referenced here:
        #   B = batch size (number of one batch)
        #   S = seq_length
        #   E = embedding_size(word2vec dimension)
        #   L =  num_perspective
        #   H = context_hidden_size
        #   A = agg_hidden_size

        with tf.variable_scope(config.scope):

            # -----Word Representation Layer-----
            with tf.variable_scope("embedding"):
                if config.use_pre_vec:
                    assert embed_vec is not None
                    self.word_embedding = tf.get_variable(initializer=embed_vec,
                                                          name='word_embedding',
                                                          dtype=tf.float32,
                                                          trainable=config.is_pre_vec_tuning)

                else:
                    self.word_embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_dim],
                                                          name='word_embedding',
                                                          initializer=create_initializer(config.initializer_range),
                                                          trainable=True)

                self.a_embedding = tf.nn.embedding_lookup(self.word_embedding, self.input_a)  # [B, S, E]
                self.b_embedding = tf.nn.embedding_lookup(self.word_embedding, self.input_b)  # [B, S, E]

                a_embedding = dropout(self.a_embedding, self.dropout_prob)    # [B, S, E]
                b_embedding = dropout(self.b_embedding, self.dropout_prob)    # [B, S, E]

            # ----- Context Representation Layer -----
            # 论文中是取context，tf不会输出所有时刻的ctx，这里用输出值代替
            with tf.variable_scope("context_representation_layer"):
                with tf.variable_scope("bilstm_context", reuse=False):
                    (self.a_fw, self.a_bw), _ = BiLSTM(a_embedding, config.context_hidden_size)
                    self.a_fw = dropout(self.a_fw, self.dropout_prob)    # [B, S, H]
                    self.a_bw = dropout(self.a_bw, self.dropout_prob)    # [B, S, H]

                with tf.variable_scope("bilstm_context", reuse=True):
                    (self.b_fw, self.b_bw), _ = BiLSTM(b_embedding, config.context_hidden_size)
                    self.a_fw = dropout(self.b_fw, self.dropout_prob)    # [B, S, H]
                    self.b_bw = dropout(self.b_bw, self.dropout_prob)    # [B, S, H]

            # ----- Matching Layer -----
            with tf.variable_scope("matching_layer"):
                for i in range(1, 9):
                    setattr(self, f'w{i}', tf.get_variable(name=f'w{i}',
                                                           shape=(config.num_perspective, config.context_hidden_size),
                                                           dtype=tf.float32))

                # 1、Full-Matching
                # all shape are[B, S, 1, L]
                a_full_fw = full_matching(self.a_fw,
                                          tf.expand_dims(self.b_fw[:, -1, :], 1),
                                          self.w1,
                                          config.num_perspective)
                a_full_bw = full_matching(self.a_bw,
                                          tf.expand_dims(self.b_bw[:, 0, :], 1),
                                          self.w2,
                                          config.num_perspective)
                b_full_fw = full_matching(self.b_fw,
                                          tf.expand_dims(self.a_fw[:, -1, :], 1),
                                          self.w1,
                                          config.num_perspective)
                b_full_bw = full_matching(self.b_bw,
                                          tf.expand_dims(self.a_bw[:, 0, :], 1),
                                          self.w2,
                                          config.num_perspective)

                # 2、Maxpooling-Matching
                a_max_fw = maxpool_full_matching(self.a_fw, self.b_fw, self.w3, config.num_perspective)
                a_max_bw = maxpool_full_matching(self.a_bw, self.b_bw, self.w4, config.num_perspective)
                b_max_fw = maxpool_full_matching(self.b_fw, self.a_fw, self.w3, config.num_perspective)
                b_max_bw = maxpool_full_matching(self.b_bw, self.a_bw, self.w4, config.num_perspective)

                # 3、Attentive-Matching
                # 计算权重即相似度
                fw_cos = matrix_cos(self.a_fw, self.b_fw)   # [B, S, S]
                bw_cos = matrix_cos(self.a_bw, self.b_bw)   # [B, S, S]

                # 计算attentive vector
                a_att_fw = tf.matmul(fw_cos, self.b_fw)   # [B, S, S] 乘 [B, S, H] = [B, S, H] 下七同
                a_att_bw = tf.matmul(bw_cos, self.b_bw)
                b_att_fw = tf.matmul(tf.transpose(fw_cos, [0, 2, 1]), self.a_fw)
                b_att_bw = tf.matmul(tf.transpose(bw_cos, [0, 2, 1]), self.a_bw)

                # [B, S, H]
                a_mean_fw = tf.math.divide(a_att_fw, tf.reduce_sum(fw_cos, axis=2, keepdims=True))
                a_mean_bw = tf.math.divide(a_att_bw, tf.reduce_sum(bw_cos, axis=2, keepdims=True))
                b_mean_fw = tf.math.divide(b_att_fw,
                                           tf.reduce_sum(tf.transpose(fw_cos, [0, 2, 1]), axis=2, keepdims=True))
                b_mean_bw = tf.math.divide(b_att_bw,
                                           tf.reduce_sum(tf.transpose(bw_cos, [0, 2, 1]), axis=2, keepdims=True))

                # 对于a的每个time_step所对应的对b注意力平均向量不同，所以维度应是[B, S, H]
                # [a1, a2, ..., a_s]与[mean_1, mean_2, ..., mean_s]对应向量作matching
                a_att_mean_fw = full_matching(self.a_fw, a_mean_fw, self.w5, config.num_perspective, mean=True)
                a_att_mean_bw = full_matching(self.a_bw, a_mean_bw, self.w6, config.num_perspective, mean=True)
                b_att_mean_fw = full_matching(self.b_fw, b_mean_fw, self.w5, config.num_perspective, mean=True)
                b_att_mean_bw = full_matching(self.b_bw, b_mean_bw, self.w6, config.num_perspective, mean=True)

                # 4、Max-Attentive-Matching
                # a_att_max_fw = tf.reduce_max(a_att_fw, axis=1, keep_dims=True)   # [B, 1, H]
                # a_att_max_bw = tf.reduce_max(a_att_bw, axis=1, keep_dims=True)
                # b_att_max_fw = tf.reduce_max(b_att_fw, axis=1, keep_dims=True)
                # b_att_max_bw = tf.reduce_max(b_att_bw, axis=1, keep_dims=True)

                # [B, S, S] * [B, S, H] = [B, S, H] 求出每个a的time_step所对应b的余弦最大向量
                a_att_max_fw = tf.matmul(
                    tf.one_hot(tf.argmax(fw_cos, axis=2), depth=config.seq_length),
                    self.b_fw)
                a_att_max_bw = tf.matmul(
                    tf.one_hot(tf.argmax(bw_cos, axis=2), depth=config.seq_length),
                    self.b_bw)
                b_att_max_fw = tf.matmul(
                    tf.one_hot(tf.argmax(tf.transpose(fw_cos, [0, 2, 1]), axis=2), depth=config.seq_length),
                    self.a_fw)
                b_att_max_bw = tf.matmul(
                    tf.one_hot(tf.argmax(tf.transpose(bw_cos, [0, 2, 1]), axis=2), depth=config.seq_length),
                    self.a_bw)

                a_att_max_fw = full_matching(self.a_fw, a_att_max_fw, self.w7, config.num_perspective, mean=True)
                a_att_max_bw = full_matching(self.a_bw, a_att_max_bw, self.w8, config.num_perspective, mean=True)
                b_att_max_fw = full_matching(self.b_fw, b_att_max_fw, self.w7, config.num_perspective, mean=True)
                b_att_max_bw = full_matching(self.b_bw, b_att_max_bw, self.w8, config.num_perspective, mean=True)

                mv_a = tf.concat(
                    [a_full_fw, a_max_fw, a_att_mean_fw, a_att_max_fw,
                     a_full_bw, a_max_bw, a_att_mean_bw, a_att_max_bw],
                    axis=2)

                mv_b = tf.concat(
                    [b_full_fw, b_max_fw, b_att_mean_fw, b_att_max_fw,
                     b_full_bw, b_max_bw, b_att_mean_bw, b_att_max_bw],
                    axis=2)

                mv_a = dropout(mv_a, self.dropout_prob)
                mv_b = dropout(mv_b, self.dropout_prob)

                self.mv_a = tf.reshape(mv_a, [-1, mv_a.shape[1], mv_a.shape[2] * mv_a.shape[3]])
                self.mv_b = tf.reshape(mv_b, [-1, mv_b.shape[1], mv_b.shape[2] * mv_b.shape[3]])   # [B, S, 8 * L]

            # ----- Aggregation Layer -----
            with tf.variable_scope("aggregation_layer"):
                with tf.variable_scope("bilstm_agg", reuse=False):
                    (a_f_agg, a_b_agg), _ = BiLSTM(self.mv_a, config.agg_hidden_size)
                with tf.variable_scope("bilstm_agg", reuse=True):
                    (b_f_agg, b_b_agg), _ = BiLSTM(self.mv_b, config.agg_hidden_size)

                a_f_last = tf.expand_dims(a_f_agg[:, -1, :], axis=1)   # [B, 1, A]
                a_b_last = tf.expand_dims(a_b_agg[:, 0, :], axis=1)
                b_f_last = tf.expand_dims(b_f_agg[:, -1, :], axis=1)
                b_b_last = tf.expand_dims(b_b_agg[:, 0, :], axis=1)

                agg_output = tf.concat([a_f_last, a_b_last, b_f_last, b_b_last], axis=2)   # [B, 1, 32 * L]
                agg_output = tf.reshape(agg_output, shape=[-1, agg_output.shape[1] * agg_output.shape[2]])
                self.agg_output = dropout(agg_output, self.dropout_prob)

            # ----- Output Layer -----
            with tf.variable_scope("output_layer"):
                self.pooled_output = tf.layers.dense(
                    self.agg_output,
                    config.connection_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))
                self.logits = tf.layers.dense(
                    self.pooled_output,
                    config.num_classes,
                    activation=None,
                    kernel_initializer=create_initializer(config.initializer_range)
                )
                self.probabilities = tf.nn.softmax(self.logits, axis=-1)
                self.prediction = tf.argmax(self.probabilities, axis=-1)

            # model train
            with tf.variable_scope("train_op"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(
                    config.learning_rate,
                    self.global_step,
                    config.decay_step,
                    config.decay_rate)
                # 优化器
                self.optim = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)

            with tf.name_scope("accuracy"):
                # 准确率
                correct_pred = tf.equal(self.input_y, self.prediction)
                self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def BiLSTM(x, hidden_size):
    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)


def full_matching(metric, vec, w, num_perspective, mean=False):
    """
    m = f(vec*w, metric*w), len(m)=L, m_k = f(vec, metric, w_k)
    for one situation; Maxpooling_Matching
    :param metric: sequence output(LSTM)
    :param vec: other sentence one_step vec
    :param w: W矩阵，shape=[L, H]
    :param num_perspective: value=L
    :param mean: 当求Attentive-Matching时用到
    :return:
    """
    w = tf.expand_dims(tf.expand_dims(w, 0), 2)  # [1, L, 1, H]
    metric = w * tf.stack([metric] * num_perspective, axis=1)  # [1, L, 1, H] * [B, L, S, H] = [B, L, S, H]
    vec = w * tf.stack([vec] * num_perspective, axis=1)  # [1, L, 1, H] * [B, L, 1, H] = [B, L, 1, H]

    if mean:
        # [B, L, S, 1] while input metric and vec all [B, S, H]
        m = tf.reduce_sum(tf.multiply(metric, vec), axis=3, keepdims=True)
    else:
        m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))   # [B, L, S, 1]
    n = tf.norm(metric, axis=3, keepdims=True) * tf.norm(vec, axis=3, keepdims=True)  # [B, L, S, 1]
    cosine = tf.transpose(tf.math.divide(m, n), [0, 2, 3, 1])   # [B, S, 1, L]

    return cosine


def maxpool_full_matching(v1, v2, w, num_perspective):
    cosine = full_matching(v1, v2, w, num_perspective)  # [B, S, S, L]
    max_value = tf.reduce_max(cosine, axis=2, keepdims=True)   # [B, S, 1, L] 取与v2各列作用的最大值
    return max_value


def matrix_cos(v1, v2):
    m = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))  # [B, S, H] 乘 [B, H, S] = [B, S, S]
    n = tf.norm(v1, axis=2, keepdims=True) * tf.norm(v2, axis=2, keepdims=True)
    # cosine = m / n
    cosine = tf.math.divide(m, n)  # [B, S, S]
    return cosine


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)
