python3 train.py \
	--task_name=BQ \
	--data_dir=data/BQ_corpus \
	--train_set=BQ_train.json \
	--dev_set=BQ_dev.json \
	--test_set=BQ_test.json \
	--output_dir=output_dir/BQ_corpus \
	--vec_file=data/word2vec.txt \
	--model_config_file=config.json \
	--vocab_file=vocab.txt \
	--seq_length=128 \
	--vocab_size=5000 \
	--embedding_size=300 \
	--context_hidden_size=256 \
	--agg_hidden_size=128 \
	--num_perspective=4 \
	--connection_size=64 \
	--dropout_prob=0.1 \
	--use_pre_vec=True \
	--train_batch_size=128 \
	--eval_batch_size=128 \
	--learning_rate=0.01 \
	--decay_rate=0.97 \
	--decay_steps=300 \
	--print_train_steps=20 \
	--max_eval_steps=300 \
	--num_epochs=10 \
	--require_improvement=5000
