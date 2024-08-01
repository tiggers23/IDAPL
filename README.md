# Rethinking the Effect of Uninformative Class Name in Prompt Learning
This repo contains the code for the IDAPL research project, which focuses on improving the generalizability of prompt learning by improving the semantic richness of class name embedding. This code is based on the [CoOp](https://github.com/KaiyangZhou/CoOp) implementation.
# How to install
This code is built on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). so you need to install the dassl environment first. All installation details for the environment can be referred to [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch#installation).
Follow [DATASETS](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.
If you want to use the GPT descriptions, please thank and download the corpora provided by [CLIP-A-Self](https://github.com/mayug/VDT-Adapter/tree/main).
# Few-shot setting on 11 datasets.
## How to run!
We have chosen hyperparameters 0.1 and 5.0 for the weights of learnable class vectors and classification loss weights, but they are generally robust in most cases, therefore, you can choose them based on your downstream tasks. 
```Bash
bash scripts/coop_idapl/main.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} **${ASSOCIATIVE_LEARNING}** ${SCORE_LC} ${SCORE_CLF}
```
### For example:
When training base classes, please remember to set the **ASSOCIATIVE_LEARNING** to **True**
```Bash
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_idapl/main.sh stanford_cars vit_b16_ep50_ctxv1 16 16 True 0.1 5.0
```
## How to test!
```Bash
bash scripts/coop_idapl/eval.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} **${ASSOCIATIVE_LEARNING}** ${SCORE_LC} ${SCORE_CLF} ${SUB}
```
### For example:
When testing basic classes, please set **ASSOCIATIVE_LEARNING** to **True** and **SUB** to **base**:
```Bash
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_idapl/eval.sh stanford_cars vit_b16_ep50_ctxv1 16 16 True 0.1 5.0 base
```
When testing novel classes, only the learned prompt contexts are used, therefore, please set **ASSOCIATIVE_LEARNING** to **False** and **SUB** to **new**:
```Bash
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_idapl/eval.sh stanford_cars vit_b16_ep50_ctxv1 16 16 False 0.1 5.0 new
```
