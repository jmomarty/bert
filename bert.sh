#!/usr/bin/env bash
# where to put the data
export SEMEVAL=gs://jmomarty/semeval/stacking/
# the path to the pretrained model
# model bert.L-12_H-768_A-12_standard.nopreprocessing.f1.0.835
export BERT_BASE_DIR=gs://jmomarty/uncased_L-12_H-768_A-12
# training
python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL   --do_train=true \
        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   \
           --learning_rate=2e-5   --num_train_epochs=4.0   \
           --output_dir=gs://jmomarty/semeval/bert_models/L-12_H-768_A-12_standard/full \
           --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask A
#python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL \
#--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
#--init_checkpoint=gs://jmomarty/semeval/bert_models/L-12_H-768_A-12_standard/full/model.ckpt.1062   --max_seq_length=128   \
#--train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=4.0   \
#--output_dir=gs://jmomarty/bert_output_dir/prediction --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask B
# predicting on subtask Bdev
#export BERT_BASE_DIR=gs://jmomarty/uncased_L-12_H-768_A-12
#python run_classifier.py   --do_lower_case=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=16   --learning_rate=5e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-12_H-768_A-12_lr_5e5_bs_16/$i --use_tpu=True --tpu_name=$TPU_NAME
#export BERT_BASE_DIR=gs://jmomarty/uncased_L-12_H-768_A-12
#python run_classifier.py   --do_lower_case=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=16   --learning_rate=5e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-12_H-768_A-12_lr_5e5_bs_16/$i --use_tpu=True --tpu_name=$TPU_NAME
#export BERT_BASE_DIR=gs://jmomarty/uncased_L-12_H-768_A-12
#python run_classifier.py   --do_lower_case=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=16   --learning_rate=5e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-12_H-768_A-12_lr_5e5_bs_16/$i --use_tpu=True --tpu_name=$TPU_NAME
#export BERT_BASE_DIR=gs://jmomarty/uncased_L-24_H-1024_A-16
#python run_classifier.py   --do_lower_case=False --fold_number=$i --task_name=SEMEVAL   --do_train=true  --do_predict=true --data_dir=$SEMEVAL --fold_number=$i --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=16   --learning_rate=5e-5   --num_train_epochs=4.0   --output_dir=gs://jmomarty/semeval/bert_models/cased_L-24_H-1024_A-16_lr_5e5_bs_16/$i --use_tpu=True --tpu_name=$TPU_NAME
|