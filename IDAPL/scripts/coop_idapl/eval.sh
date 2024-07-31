#!/bin/bash
#cd ../..
# custom config
# DATA=/path/to/datasets
# GPT_DATA=/path/to/GPT4_data
# TOKENCLASSIFIER_PRETRAIN_PATH=/path/to/classifier

TRAINER=CoOp_IDAPL
CFG=vit_b16_ep50_ctxv1  # config file
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
LDEP=50
SUB=new
ASSOCIATIVE_LEARNING=False
DATASET=$1
SCORE_LC=$2
SCORE_CLF=$3

for SEED in 1 2 3
do
    MODEL_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/SCORE_LC_${SCORE_LC}_SCORE_CLF_${SCORE_CLF}/seed${SEED}
    DIR=output/test_base2${SUB}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/SCORE_LC_${SCORE_LC}_SCORE_CLF_${SCORE_CLF}/seed${SEED}
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
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LDEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.ASSOCIATIVE_LEARNING ${ASSOCIATIVE_LEARNING} \
	   TRAINER.COOP.SCORE_LC ${SCORE_LC} \
	   TRAINER.COOP.SCORE_CLF ${SCORE_CLF} \
	   TRAINER.COOP.GPT_DATA ${GPT_DATA} \
	   TRAINER.COOP.TOKENCLASSIFIER_PRETRAIN_PATH ${TOKENCLASSIFIER_PRETRAIN_PATH} \
	   DATASET.SUBSAMPLE_CLASSES ${SUB} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done