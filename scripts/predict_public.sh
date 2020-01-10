#!/bin/bash

export SEED=15270
export PYTORCH_SEED=`expr $SEED / 10`
export NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export TEST_PATH=data/PaperAbstract/public_test.jsonl
export cuda_device=0

# export SERIALIZATION_DIR=output/background
# export PREDS_FILE=preds/preds_background.txt

label_type=(all background objectives methods results conclusions others)
# label_type=(all)
for label in "${label_type[@]}"
do
  python -m allennlp.run predict --silent --cuda-device $cuda_device --predictor SeqClassificationPredictor --output-file preds/preds_public_$label.txt  --include-package abstract_labeling output/$label $TEST_PATH
done

