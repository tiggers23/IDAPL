# Rethinking the Effect of Uninformative Class Name in Prompt Learning
This repo contains the code for the IDAPL research project, which focuses on investigating the effect of uninformative classname in prompt Learning. This code is based on the [CoOp](https://github.com/KaiyangZhou/CoOp) implementation.
# How to install
This code is built on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). so you need to install the dassl environment first. After that, run `pip install -r requirements.txt` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). 
```
# Create a conda environment
conda create -n dassl python=3.8
# Activate the environment
conda activate dassl
# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# Install dependencies
pip install -r requirements.txt
# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
Follow [DATASETS.md](DATASETS.md) to install the datasets.
And if you want to use the description generated by [gpt4_data](https://github.com/mayug/VDT-Adapter/tree/acece943215deb73545bfa3d005e5de5f5cfec9b/gpt4_data) to standardize the direction of prompt learning, please thank and download the data support provided by [CLIP-A-Self](https://github.com/mayug/VDT-Adapter/tree/main).
# Few-shot setting on 11 datasets.
## How to run!
#We have chosen hyperparameters 0.1 and 5.0 for the weights of learnable class vectors and classification loss weights, but they are generally robust in most cases. You can choose based on your downstream tasks
```Bash
bash scripts/coop_idapl/main.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} ${ASSOCIATIVE_LEARNING} ${SCORE_LC} ${SCORE_CLF}
```
For Example:
```Bash
bash scripts/coop_idapl/main.sh standford_cars vit_b16_ep50_ctxv1 16 16 True 0.1 0.5
```
## How to test!
When you migrate to a novel class, please set ASSOCIATIVE_LEARNING False, and the learnable vectors are only used for the base class!
```Bash
bash scripts/coop_idapl/eval.sh ${DATSET_NAME} ${CFG} ${SHOTS} ${N_CTX} ${ASSOCIATIVE_LEARNING} ${SCORE_LC} ${SCORE_CLF}
```
This means that when you load the model of the previous run example, you do not use the learnable vectors of categories
For Example:
```Bash
bash scripts/coop_idapl/eval.sh standford_cars vit_b16_ep50_ctxv1 16 16 False 0.1 0.5
```
# Citation
If you use this code in your research, please kindly cite the following papers
【缺引用】
