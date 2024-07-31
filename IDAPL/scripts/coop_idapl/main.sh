#!/bin/bash

# custom config
# DATA=/path/to/datasets
# GPT_DATA=/path/to/GPT4_data
# TOKENCLASSIFIER_PRETRAIN_PATH=/path/to/classifier

TRAINER=CoOp_IDAPL
CFG=vit_b16_ep50_ctxv1 
SHOTS=16   # number of shots (1, 2, 4, 8, 16)
ASSOCIATIVE_LEARNING=True
SCORE_LC=0.1
SCORE_CLF=5.0
DATASET=$1

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/SCORE_LC_${SCORE_LC}_SCORE_CLF_${SCORE_CLF}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
	   --seed ${SEED} \
	   --trainer ${TRAINER} \
	   --dataset-config-file configs/datasets/${DATASET}.yaml \
	   --config-file configs/trainers/CoOp/${CFG}.yaml \
	   --output-dir ${DIR} \
	   TRAINER.COOP.ASSOCIATIVE_LEARNING ${ASSOCIATIVE_LEARNING} \
	   TRAINER.COOP.SCORE_LC ${SCORE_LC} \
	   TRAINER.COOP.SCORE_CLF ${SCORE_CLF} \
	   TRAINER.COOP.GPT_DATA ${GPT_DATA} \
	   TRAINER.COOP.TOKENCLASSIFIER_PRETRAIN_PATH ${TOKENCLASSIFIER_PRETRAIN_PATH} \
	   DATASET.SUBSAMPLE_CLASSES base \
	   DATASET.NUM_SHOTS ${SHOTS}
    fi
done