export SEMEVAL=gs://jmomarty/semeval/stacking/first_level
export BERT_BASE_DIR=gs://jmomarty/uncased_L-24_H-1024_A-16
for i in `seq 0 10`;
do
    python run_classifier.py   --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/uncased_L-24_H-1024_A-16_standard/$i --use_tpu=True --tpu_name=$TPU_NAME
done
export BERT_BASE_DIR=gs://jmomarty/cased_L-12_H-768_A-12
for i in `seq 0 10`;
do
    python run_classifier.py   --do_lower=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-12_H-768_A-12_standard/$i --use_tpu=True --tpu_name=$TPU_NAME
done
export BERT_BASE_DIR=gs://jmomarty/cased_L-24_H-1024_A-16
for i in `seq 0 10`;
do
    python run_classifier.py   --do_lower=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-24_H-1024_A-16_standard/$i --use_tpu=True --tpu_name=$TPU_NAME
done



