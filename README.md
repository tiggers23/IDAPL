# Rethinking the Effect of Uninformative Class Name in Prompt Learning
This repo contains the code for the IDAPL research project, which focuses on investigating the effect of uninformative classname in prompt Learning. This code is based on the [CoOp](https://github.com/KaiyangZhou/CoOp) implementation.
# How to install
This code is built on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). so you need to install the dassl environment first. All installation details for the environment can refer to [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch#installation).
Follow [DATASETS](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.
If you want to use the GPT descriptions, please thank and download the corpora provided by [CLIP-A-Self](https://github.com/mayug/VDT-Adapter/tree/main).
# Few-shot setting on 11 datasets.
## How to run!
We have chosen hyperparameters 0.1 and 5.0 for the weights of learnable class vectors and classification loss weights, but they are generally robust in most cases. You can choose based on your downstream tasks
```Bash
bash scripts/coop_idapl/main.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} ${SCORE_LC} ${SCORE_CLF}
```
For Example:
```Bash
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_idapl/main.sh stanford_cars vit_b16_ep50_ctxv1 16 16 0.1 5.0
```
## How to test!
When you migrate to a novel class, please set ASSOCIATIVE_LEARNING False, and the learnable vectors are only used for the base class!
```Bash
bash scripts/coop_idapl/eval.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} ${SCORE_LC} ${SCORE_CLF}
```
This means that when you load the model of the previous run example, you do not use the learnable vectors of categories
For Example:
```Bash
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_idapl/eval.sh stanford_cars vit_b16_ep50_ctxv1 16 16 0.1 5.0
```
