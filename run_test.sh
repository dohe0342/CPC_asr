#!/bin/bash
stage="$1" # parse first argument 

if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech
    CUDA_VISIBLE_DEVICES=0 python main.py \
	--train-raw LibriSpeech/train-Librispeech.h5 \
	--validation-raw LibriSpeech/dev-Librispeech.h5 \
	--eval-raw LibriSpeech/test-Librispeech.h5 \
	--train-list LibriSpeech/list/train.txt \
        --validation-list LibriSpeech/list/validation.txt \
        --eval-list LibriSpeech/list/eval.txt \
        --logging-dir snapshot/cdc/ \
	--log-interval 50 \
	--audio-window 20480 \
	--timestep 12 \
	--masked-frames 10 \
	--n-warmup-steps 1000 \
	| tee CPC_train.log
fi

if [ $stage -eq 1 ]; then
    # call spk_class.py
    CUDA_VISIBLE_DEVICES=0 python spk_class.py \
		--raw-hdf5 LibriSpeech/train-clean-100.h5 \
		--train-list LibriSpeech/list/spk_train.txt \
		--validation-list LibriSpeech/list/spk_val.txt \
		--eval-list LibriSpeech/list/spk_test.txt \
		--index-file LibriSpeech/spk2idx \
		--logging-dir snapshot/cdc/ \
		--log-interval 5 \
		--model-path snapshot/cdc/CPC_pretrained.pth 
fi


