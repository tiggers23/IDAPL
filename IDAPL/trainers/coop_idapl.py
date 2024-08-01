import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.utils.data import Dataset, DataLoader
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np

_tokenizer = _Tokenizer()

DATASET_NAME={
    "OxfordPets": "oxford_pets",
    "Caltech101": "caltech101",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "DescribableTextures",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
}
GPT4_FILENAME = {
    "OxfordPets": 'oxford_pets.pt',
    "CUB": 'cub.pt',
    "OxfordFlowers": 'oxford_flowers.pt',
    "FGVCAircraft": 'fgvc_aircraft.pt',
    "DescribableTextures":  'dtd.pt',
    "EuroSAT":  "eurosat.pt",
    "StanfordCars": "stanford_cars.pt",
    "Food101": "food-101.pt",
    "SUN397": "sun397.pt",
    "Caltech101": "caltech-101.pt",
    "UCF101": "ucf101.pt",
    "ImageNet": "imagenet.pt",
    "ImageNetSketch": "imagenet.pt",
    "ImageNetR": "imagenet.pt",
    "ImageNetA": "imagenet.pt",
    "ImageNetV2": "imagenet.pt",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def extract_gpt4_features_from_sentences(cfg, clip_model):
    # Path to GPT-4 data
    text_path = cfg.TRAINER.COOP.GPT_DATA 
    # Load GPT-4 sentences
    gpt4_sentences = torch.load(f'{text_path}/gpt4_data/{GPT4_FILENAME[cfg.DATASET.NAME]}')
    all_categories = list(gpt4_sentences.keys())
    encoded_gpt4_features_dict = {}
    clip_model = clip_model.to('cuda')
    # Process each category
    for category in all_categories:
        if cfg.DATASET.NAME not in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
            category = '_'.join(category.split(' '))
        current_sentences = gpt4_sentences[category.lower()]
        tokenized_prompts = torch.cat([clip.tokenize(c) for c in current_sentences]).to('cuda')
        with torch.no_grad():
            current_text_features = clip_model.encode_text(tokenized_prompts)
        # Store encoded features in dictionary
        encoded_gpt4_features_dict[category] = current_text_features
    return encoded_gpt4_features_dict
    

class TwoLayerNN(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1) 
        self.sigmoid = nn.Sigmoid()           

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = self.sigmoid(self.fc2(x))
        return x
token_classifier = TwoLayerNN(512).cuda()

class TextDataset(Dataset):
    def __init__(self, attributes_dict, classnames):
        super().__init__()
        self.sample_list = []
        self.label_list = []
        k = 0
        for class_id, c in enumerate(classnames):
            for _ in attributes_dict[c]:
                self.sample_list.append(k)
                self.label_list.append(class_id)
                k += 1

    def __getitem__(self, index):
        return self.sample_list[index], self.label_list[index]

    def __len__(self):
        return len(self.sample_list)

    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.cfg = cfg
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        if self.cfg.TRAINER.COOP.ASSOCIATIVE_LEARNING:
            learnable_class_tokens = []
            for name_len in name_lens:
                lc_vectors = torch.empty(1, name_len, ctx_dim, dtype=dtype)
                nn.init.normal_(lc_vectors, std=0.02)
                learnable_class_tokens.append(nn.Parameter(lc_vectors))
            print(f"Use Learnable_Class_Tokens")
            self.lc_tokens = nn.ParameterList(learnable_class_tokens)
            self.learnable_classname_weights = cfg.TRAINER.COOP.SCORE_LC

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts=[]
        average_class_features =[] 
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i:i + 1, :, :]
            class_i = suffix[i:i + 1, :name_len, :]     
            if self.cfg.TRAINER.COOP.ASSOCIATIVE_LEARNING:
                class_i = self.learnable_classname_weights * self.lc_tokens[i] + (1 - self.learnable_classname_weights) * class_i
            suffix_i = suffix[i: i + 1, name_len:, :]
            ctx_i = ctx[i: i + 1, :, :]
            prompt = torch.cat([prefix_i, ctx_i, class_i, suffix_i], dim=1)
            prompts.append(prompt)
            average_class_features.append(class_i.mean(dim=1))
        prompts = torch.cat(prompts, dim=0)
        average_class_features = torch.cat(average_class_features,dim=0)
        return prompts,ctx,average_class_features

def compute_embeddings_and_labels(ctx,average_class_features):
    gene_e = ctx.mean(dim=1)
    class_e = average_class_features
    
    ctx_labels = np.zeros(len(gene_e))
    cnx_labels = np.ones(len(class_e))
    
    merged_embeddings = torch.cat((gene_e, class_e), dim=0)
    merged_labels = np.concatenate((ctx_labels, cnx_labels), axis=0)
    indices = np.arange(len(merged_embeddings))
    np.random.shuffle(indices)
    merged_embeddings = merged_embeddings[indices]
    merged_labels = merged_labels[indices]
    
    labels_tensor = torch.tensor(merged_labels, dtype=torch.float32)
    
    return merged_embeddings, labels_tensor

def load_gpt4_text_features(cfg, clip_model, classnames):
    encoded_gpt4_features_dict = extract_gpt4_features_from_sentences(cfg, clip_model)
    if cfg.DATASET.NAME in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
        for i in range(len(classnames)):
            classnames[i] = classnames[i].lower()
    else:
        for i in range(len(classnames)):
            classnames[i] = '_'.join(classnames[i].lower().split(' '))
    gpt4_text_features = [encoded_gpt4_features_dict[c] for c in classnames]
    gpt4_text_features = torch.stack(gpt4_text_features, dim=0)
    base_dataset = TextDataset(encoded_gpt4_features_dict, classnames)
    return gpt4_text_features,base_dataset
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.len_c = len(classnames)  
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        gpt4_text_features,base_dataset = load_gpt4_text_features(cfg, clip_model, classnames)
        self.gpt4_text_features = gpt4_text_features.type(self.dtype)
        #Choose your GPT data loading capacity
        self.base_loader = DataLoader(base_dataset, batch_size=base_dataset.__len__(), shuffle=True, drop_last=False)
        self.criterion = nn.BCELoss()

    def forward(self, image,base_sample=None):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts,ctx,average_class_features = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        #compute cls_loss
        merged_embeddings, labels_tensor = compute_embeddings_and_labels(ctx, average_class_features)
        tcf_outputs = token_classifier(merged_embeddings)
        clf_loss = self.criterion(tcf_outputs, labels_tensor.view(-1, 1).to('cuda'))

        if self.prompt_learner.training:
            base_gpt_text = self.gpt4_text_features.view(self.len_c*self.gpt4_text_features.shape[1],-1)
            base_gpt_index, base_class_index = base_sample
            base_gpt_prompts = base_gpt_text[base_gpt_index]
            base_gpt_prompts  = base_gpt_prompts  / base_gpt_prompts.norm(dim=-1, keepdim=True)
            base_one_hot_label = torch.nn.functional.one_hot(base_class_index,num_classes=self.len_c).type(self.dtype).to('cuda')
            base_l2_loss = torch.mean(torch.sum((base_one_hot_label @ text_features-base_gpt_prompts)**2, dim=1))
            return logits, base_l2_loss, clf_loss
        return logits


@TRAINER_REGISTRY.register()
class CoOp_IDAPL(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        TOKENCLASSIFIER_PRETRAIN_PATH = cfg.TRAINER.COOP.TOKENCLASSIFIER_PRETRAIN_PATH
        if TOKENCLASSIFIER_PRETRAIN_PATH:
            TOKENCLASSIFIER_INIT_WEIGHTS = f'{TOKENCLASSIFIER_PRETRAIN_PATH}/{DATASET_NAME[cfg.DATASET.NAME]}_model.pth'
            load_pretrained_weights(token_classifier, TOKENCLASSIFIER_INIT_WEIGHTS)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.base_loader = self.model.base_loader
        self.base_iter = iter(self.base_loader)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                
        #Open Classifier
        for param in token_classifier.parameters():
            param.requires_grad = True

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.tc_optim = build_optimizer(token_classifier, cfg.OPTIM)
        self.tc_sched = build_lr_scheduler(self.tc_optim, cfg.OPTIM)
        self.register_model("token_classifier", token_classifier, self.tc_optim, self.tc_sched)


        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        try:
            base_sample = self.base_iter.__next__()
        except:
            self.base_iter = iter(self.base_loader)
            base_sample = self.base_iter.__next__()
        
        logits, base_l2_loss,clf_loss = self.model(image,base_sample)
        ITC_loss = F.cross_entropy(logits, label)
        loss =  ITC_loss + base_l2_loss + self.cfg.TRAINER.COOP.SCORE_CLF * clf_loss
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            # NOTE: When you test the new category, please remove lc_token
            if self.cfg.DATASET.SUBSAMPLE_CLASSES == "new":
                keys_to_remove=[]
                for key in state_dict:
                    if "lc_tokens" in key:
                       keys_to_remove.append(key)
                for key in keys_to_remove:
                    del state_dict[key]
                print("Load model to test novel categories, removing lc_tokens")

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)