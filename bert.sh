#!/usr/bin/env bash
# where to put the data
export SEMEVAL=gs://jmomarty2/semeval/stacking
# the path to the pretrained model

# model bert.uncased.L-12_H-768_A-12_standard.nopreprocessing.f1.0.835
#export BERT_BASE_DIR=gs://jmomarty2/uncased_L-12_H-768_A-12
#export MODEL_NAME=L-12_H-768_A-12_standard
## training
#python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL   --do_train=true \
#        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   \
#           --learning_rate=2e-5   --num_train_epochs=4.0   \
#           --output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full \
#           --use_tpu=True --tpu_name=$TPU_NAME
## predicting on subtask A
#python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskA \
#--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
#--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
#--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskA --use_tpu=True --tpu_name=$TPU_NAME
## predicting on subtask B
#python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskB \
#--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
#--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
#--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskB --use_tpu=True --tpu_name=$TPU_NAME
## predicting on subtask Bdev
#python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskBdev \
#--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
#--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
#--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskBdev --use_tpu=True --tpu_name=$TPU_NAME

# model bert.cased_L-12_H-768_A-12_lr_5e5_bs_16.f1.0.825
export BERT_BASE_DIR=gs://jmomarty2/cased_L-12_H-768_A-12
export MODEL_NAME=cased_L-12_H-768_A-12_lr_5e5_bs_16/
# training
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL   --do_train=true \
        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=16   \
           --learning_rate=5e-5   --num_train_epochs=4.0   \
           --output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full \
           --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask A
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskA \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskA --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask B
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskB \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskB --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask Bdev
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskBdev \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskBdev --use_tpu=True --tpu_name=$TPU_NAME

# model bert_cased_L-12_H-768_A-12_standard.f1.0.833
export BERT_BASE_DIR=gs://jmomarty2/cased_L-12_H-768_A-12
export MODEL_NAME=cased_L-12_H-768_A-12_standard/
# training
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL   --do_train=true \
        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   \
           --learning_rate=2e-5   --num_train_epochs=4.0   \
           --output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full \
           --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask A
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskA \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskA --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask B
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskB \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskB --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask Bdev
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskBdev \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskBdev --use_tpu=True --tpu_name=$TPU_NAME

# model bert_cased_L-24_H-1024_A-16_standard.f1.0.750
export BERT_BASE_DIR=gs://jmomarty2/cased_L-24_H-1024_A-16
export MODEL_NAME=cased_L-24_H-1024_A-16_standard
# training
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL   --do_train=true \
        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   \
           --learning_rate=2e-5   --num_train_epochs=4.0   \
           --output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full \
           --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask A
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskA \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskA --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask B
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskB \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskB --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask Bdev
python run_classifier.py   --do_lower_case=False --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskBdev \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskBdev --use_tpu=True --tpu_name=$TPU_NAME

# model bert_uncased_L-24_H-1024_A-16_standard.f1.0.838
export BERT_BASE_DIR=gs://jmomarty2/uncased_L-24_H-1024_A-16
export MODEL_NAME=uncased_L-24_H-1024_A-16_standard/
# training
python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL   --do_train=true \
        --data_dir=$SEMEVAL --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
           --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   \
           --learning_rate=2e-5   --num_train_epochs=4.0   \
           --output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full \
           --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask A
python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskA \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskA --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask B
python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskB \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskB --use_tpu=True --tpu_name=$TPU_NAME
# predicting on subtask Bdev
python run_classifier.py   --do_lower_case=True --fold_number=0 --task_name=SEMEVAL  --do_predict=true --data_dir=$SEMEVAL/subtaskBdev \
--vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
--init_checkpoint=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/model.ckpt-1054   --max_seq_length=128   \
--output_dir=gs://jmomarty2/semeval/bert_models/$MODEL_NAME/full/subtaskBdev --use_tpu=True --tpu_name=$TPU_NAME
