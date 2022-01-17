CUDA_VISIBLE_DEVICES=0 python asr_train.py \
	--raw-hdf5 LibriSpeech/train-clean-100.h5 \
	--train-list LibriSpeech/list/train_gt.txt \
	--validation-list LibriSpeech/list/spk_val.txt \
	--eval-list LibriSpeech/list/spk_test.txt \
	--index-file LibriSpeech/spk2idx \
	--logging-dir snapshot/cdc/ \
	--log-interval 5 \
	--model-path snapshot/cdc/CPC_pretrained.pth #| tee vanilla_GRU.log
