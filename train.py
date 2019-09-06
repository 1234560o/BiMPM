import tensorflow as tf
import time
from datetime import timedelta
import modeling
from data_loader import BQLoader, SNLILoader
from data_loader import process_data, build_vocab, batch_iter, load_word2vec, read_vocab
from flags import FLAGS, flags
import os


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(model, a_batch, b_batch, y_batch, dropout_prob):
    feed_dict = {
        model.input_a: a_batch,
        model.input_b: b_batch,
        model.input_y: y_batch,
        model.dropout_prob: dropout_prob
    }
    return feed_dict


def load_train_and_dev_dataset(processor, do_dev=True):
    dev_dataset = None

    train_text_a, train_text_b, train_labels = processor.read_data(os.path.join(FLAGS.data_dir, FLAGS.train_set))
    tf.logging.info(" build vocab file... ")
    build_vocab(
        train_text_a + train_text_b,
        os.path.join(FLAGS.output_dir, FLAGS.vocab_file),
        FLAGS.vocab_size)
    tf.logging.info(" build vocab file done! ")

    tf.logging.info(" load vocab file... ")
    word2index, words = read_vocab(os.path.join(FLAGS.output_dir, FLAGS.vocab_file))
    tf.logging.info(" load vocab file done! ")

    tf.logging.info(" load train set... ")
    train_dataset = process_data(
        train_text_a, train_text_b, train_labels, word2index, FLAGS.seq_length)
    tf.logging.info(" train set total: {}".format(len(train_labels)))
    epoch_steps = len(train_labels) // FLAGS.train_batch_size
    tf.logging.info(" On train, one epoch = {} steps".format(epoch_steps))
    tf.logging.info(" load train set done! ")
    tf.logging.info("---------------------------------------\n")

    if do_dev:
        tf.logging.info(" load dev set... ")
        dev_text_a, dev_text_b, dev_labels = processor.read_data(os.path.join(FLAGS.data_dir, FLAGS.dev_set))
        dev_dataset = process_data(
            dev_text_a, dev_text_b, dev_labels, word2index, FLAGS.seq_length)
        tf.logging.info(" dev set total: {}".format(len(dev_labels)))
        dev_epoch_steps = (len(dev_labels) - 1) // FLAGS.eval_batch_size + 1
        tf.logging.info(" On dev, one epoch = {} steps".format(dev_epoch_steps))
        tf.logging.info(" load dev set done! ")
        tf.logging.info("---------------------------------------\n")

    return train_dataset, dev_dataset, words


def load_test_dataset(processor):
    tf.logging.info(" load vocab file... ")
    word2index, words = read_vocab(os.path.join(FLAGS.output_dir, FLAGS.vocab_file))
    tf.logging.info(" load vocab file done! ")

    tf.logging.info(" load test set... ")
    test_text_a, test_text_b, test_labels = processor.read_data(os.path.join(FLAGS.data_dir, FLAGS.test_set))
    test_dataset = process_data(
        test_text_a, test_text_b, test_labels, word2index, FLAGS.seq_length)
    tf.logging.info(" test set total: {}".format(len(test_labels)))
    test_epoch_steps = (len(test_labels) - 1) // FLAGS.eval_batch_size + 1
    tf.logging.info(" On test, one epoch = {} steps".format(test_epoch_steps))
    tf.logging.info(" load test set done! ")
    tf.logging.info("---------------------------------------\n")

    return test_dataset, words


def evaluate(sess, model, dev_dataset, batch_size, step, summary_op=None, writer=None):
    """评估在某一数据上的准确率和损失"""
    total_loss = 0.0
    total_acc = 0.0
    total_len = 0
    dev_a, dev_b, dev_labels = dev_dataset
    dev_batch = batch_iter(dev_a, dev_b, dev_labels, batch_size, is_training=False)
    for dev_batch_a, dev_batch_b, dev_batch_y in dev_batch:
        batch_len = len(dev_batch_y)   # 每一个batch验证集大小
        total_len += batch_len
        feed_dict = feed_data(model, dev_batch_a, dev_batch_b, dev_batch_y, 0.0)
        if summary_op is not None and writer is not None:
            loss, acc, summaries = sess.run([model.loss, model.acc, summary_op], feed_dict=feed_dict)
            writer.add_summary(summaries, step)
        else:
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / total_len, total_acc / total_len


def train(model, train_dataset, dev_dataset, do_dev=True):
    tf.logging.info(" Build saver and tensorboard... ")
    save_dir = os.path.join(FLAGS.output_dir, FLAGS.checkpoint_dir)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "best_validation")
    tf.logging.info("-------------------------------------------")
    tf.logging.info("  Write checkpoint to {}  ".format(save_path))
    tf.logging.info("-------------------------------------------")

    tensorboard_dir = os.path.join(FLAGS.output_dir, FLAGS.tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.logging.info(' ***** Training and evaluating ... *****')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_dev = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    flag = False

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)) as sess:
        sess.run(tf.global_variables_initializer())

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.acc)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(tensorboard_dir, "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(tensorboard_dir, "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 断点续训功能
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            tf.logging.info("***** Restore parameters from {} *****".format(save_path))

        train_a, train_b, train_labels = train_dataset

        for epoch in range(FLAGS.num_epochs):
            tf.logging.info("Epoch {} start ... ".format(epoch + 1))
            train_batch = batch_iter(
                train_a, train_b, train_labels, FLAGS.train_batch_size)
            for train_batch_a, train_batch_b, train_batch_y in train_batch:
                feed_dict = feed_data(model, train_batch_a, train_batch_b,
                                      train_batch_y, FLAGS.dropout_prob)
                if total_batch % FLAGS.print_train_steps == 0:
                    feed_dict[model.dropout_prob] = 0.0
                    loss_train, acc_train, current_learning_rate = sess.run(
                        [model.loss, model.acc, model.learning_rate], feed_dict=feed_dict)
                    time_dif = get_time_dif(start_time)
                    train_msg = "Epoch: {0:>2}, Iter: {1:>5} Train Loss: {2:>6.3}, Train Acc: {3:>6.2%}, " \
                                "Current Learning_rate: {4:>7.5} Time: {5} "
                    tf.logging.info(
                        train_msg.format(epoch + 1, total_batch, loss_train, acc_train, current_learning_rate, time_dif))

                if do_dev and total_batch % FLAGS.max_eval_steps == 0:
                    # 每多少轮次输出验证集上的性能
                    loss_dev, acc_dev = evaluate(
                        sess, model, dev_dataset, FLAGS.eval_batch_size, total_batch, dev_summary_op, dev_summary_writer)
                    if acc_dev > best_acc_dev:
                        # 保存最好结果
                        best_acc_dev = acc_dev
                        last_improved = total_batch
                        saver.save(sess=sess, save_path=save_path)
                        improved_str = '*****'
                    else:
                        improved_str = ''
                    time_dif = get_time_dif(start_time)
                    msg = "Epoch: {0:>2}, Iter: {1:>5} Dev Loss: {2:>6.3}, Dev Acc: {3:>6.2%}, Time: {4} {5}"
                    tf.logging.info(" -----------------------------------------------------------------------")
                    tf.logging.info(
                        msg.format(epoch + 1, total_batch, loss_dev, acc_dev, time_dif, improved_str))
                    tf.logging.info(" -----------------------------------------------------------------------")

                if total_batch - last_improved > FLAGS.require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    tf.logging.info(" $$$$$ No optimization for a long time, auto-stopping... $$$$$ ")
                    flag = True
                    break  # 跳出循环

                feed_dict[model.dropout_prob] = FLAGS.dropout_prob
                _, summaries = sess.run([model.optim, train_summary_op], feed_dict=feed_dict)  # 运行优化
                train_summary_writer.add_summary(summaries, total_batch)

                total_batch += 1

            if flag:  # 同上
                break

        train_summary_writer.close()
        dev_summary_writer.close()


def eval_test(model, test_dataset):
    """
    Evalution on best model
    """
    tf.logging.info(" Restore saver... ")
    save_dir = os.path.join(FLAGS.output_dir, "checkpoint")
    if not os.path.exists(save_dir):
        raise IOError("The model checkpoint dir is not exists")
    save_path = os.path.join(save_dir, "best_validation")
    tf.logging.info("----------------------------------------------")
    tf.logging.info("  Restore checkpoint from {}  ".format(save_path))
    tf.logging.info("----------------------------------------------")

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    loss_test, acc_test = evaluate(session, model, test_dataset, FLAGS.eval_batch_size, 0)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    tf.logging.info(msg.format(loss_test, acc_test))
    session.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and FLAGS.do_dev:
        raise ValueError("'do_dev' is true while 'do_true' must be True!")

    if not FLAGS.do_train and not FLAGS.do_test:
        raise ValueError("At least one of 'do_train' and 'do_test' must be True!")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    processors = {
        "bq": BQLoader,
        "snli": SNLILoader,
    }
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()

    # 初始化
    word_vec = None
    train_dataset = None
    dev_dataset = None
    test_dataset = None
    words = None

    start_time = time.time()

    tf.logging.info("***** Load dataset *****")
    if FLAGS.do_train:
        train_dataset, dev_dataset, words = load_train_and_dev_dataset(processor, FLAGS.do_dev)
    if FLAGS.do_test:
        test_dataset, words = load_test_dataset(processor)
    tf.logging.info("===== Load dataset done! =====")
    time_dif = get_time_dif(start_time)
    tf.logging.info(" ----- Load dataset time usage: {} ----- ".format(time_dif))

    # 如果使用预训练词向量
    if FLAGS.use_pre_vec:
        start_time1 = time.time()
        tf.logging.info(" load pre vec... ")
        word_vec = load_word2vec(words, FLAGS.vec_file)
        tf.logging.info(" load pre vec done! ")
        time_dif1 = get_time_dif(start_time1)
        tf.logging.info(" 加载词向量用时{}s. ".format(time_dif1))
        tf.logging.info("\n")

    model_config = modeling.BiMPMConfig(seq_length=FLAGS.seq_length,
                                        num_classes=FLAGS.num_classes,
                                        context_hidden_size=FLAGS.context_hidden_size,
                                        agg_hidden_size=FLAGS.agg_hidden_size,
                                        num_perspective=FLAGS.num_perspective,
                                        connection_size=FLAGS.connection_size,
                                        initializer_range=FLAGS.initializer_range,
                                        vocab_size=FLAGS.vocab_size,
                                        embedding_dim=FLAGS.embedding_size,
                                        use_pre_vec=FLAGS.use_pre_vec,
                                        is_pre_vec_tuning=FLAGS.is_pre_vec_tuning,
                                        learning_rate=FLAGS.learning_rate,
                                        decay_step=FLAGS.decay_step,
                                        decay_rate=FLAGS.decay_rate)
    model_config.to_json_file(os.path.join(FLAGS.output_dir, FLAGS.model_config_file))
    config = model_config.from_json_file(os.path.join(FLAGS.output_dir, FLAGS.model_config_file))
    tf.logging.info(" ***** Model Hyper-Parameters *****")
    for key, value in config.__dict__.items():
        tf.logging.info("{}: {}".format(key, value))

    model = modeling.BiMPMModel(config, word_vec)

    if FLAGS.do_train:
        tf.logging.info("****** Enter into training... ***** ")
        train(model, train_dataset, dev_dataset, FLAGS.do_dev)
        tf.logging.info("===== Training done! ===== ")

    if FLAGS.do_test:
        tf.logging.info("****** Enter into testing... ***** ")
        eval_test(model, test_dataset)
        tf.logging.info("===== Testing done! ===== ")


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
