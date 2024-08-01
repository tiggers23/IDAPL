#!/bin/bash
#cd ../..
# custom config
#DATA=/path/to/datasets
DATA=/data/hdd/ncr/CoOp/datasets
GPT_DATA=/data/hdd/ncr/CoOp/
TOKENCLASSIFIER_PRETRAIN_PATH=/data/hdd/ncr/2024CoOp/CoOp/classnames
TRAINER=CoOp_IDAPL
DATASET=$1
#CFG=vit_b16_ep50_ctxv1 
CFG=$2
#SHOTS=16
SHOTS=$3
#NCTX=16
N_CTX=$4
#ASSOCIATIVE_LEARNING=False
ASSOCIATIVE_LEARNING=$5
#SCORE_LC=0.1
SCORE_LC=$6
#SCORE_CLF=5.0
SCORE_CLF=$7
LDEP=50
SUB=new

for SEED in 1 2 3
do
    MODEL_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${N_CTX}_ctpend/SCORE_LC_${SCORE_LC}_SCORE_CLF_${SCORE_CLF}/seed${SEED}
    DIR=output_test/base2${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${N_CTX}_ctpend/SCORE_LC_${SCORE_LC}_SCORE_CLF_${SCORE_CLF}/seed${SEED}
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
        TRAINER.COOP.N_CTX ${N_CTX} \
        TRAINER.COOP.ASSOCIATIVE_LEARNING ${ASSOCIATIVE_LEARNING} \
	   TRAINER.COOP.SCORE_LC ${SCORE_LC} \
	   TRAINER.COOP.SCORE_CLF ${SCORE_CLF} \
	   TRAINER.COOP.GPT_DATA ${GPT_DATA} \
	   TRAINER.COOP.TOKENCLASSIFIER_PRETRAIN_PATH ${TOKENCLASSIFIER_PRETRAIN_PATH} \
	   DATASET.SUBSAMPLE_CLASSES ${SUB} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done