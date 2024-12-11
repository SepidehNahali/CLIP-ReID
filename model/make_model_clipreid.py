import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.dataset_utils import load_vehicle_features
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        
        # Load vehicle features
        label_file = cfg.DATASETS.LABEL_FILE
        color_file = cfg.DATASETS.COLOR_FILE
        type_file = cfg.DATASETS.TYPE_FILE
        camera_file = cfg.DATASETS.CAMERA_FILE

        vehicle_features = load_vehicle_features(label_file, color_file, type_file, camera_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!build_transformer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print('vehicle_features',vehicle_features)
        print('vehicle_features_keys',vehicle_features.keys())
        self.prompt_learner = PromptLearner(num_classes, 'veri', clip_model.dtype, clip_model.token_embedding, vehicle_features)
        # Assuming clip_model is the full CLIP model object, not just token_embedding

        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, clip_model, vehicle_features):
        super().__init__()
        
        # Store the vehicle features dictionary
        self.vehicle_features = vehicle_features

        # Set dynamic template depending on the dataset
        if dataset_name in ["VehicleID", "veri"]:
            self.ctx_init = "A photo of a {color} {type} vehicle."
        else:
            self.ctx_init = "A photo of a {description} person."

        self.ctx_dim = 512
        self.num_class = num_class
        self.n_cls_ctx = 4

        # Token embedding from the clip_model
        self.token_embedding = clip_model.token_embedding

        # Initialize a default tokenized prompt for all vehicles
        default_prompt = self.ctx_init.format(color="unknown", type="unknown")
        self.tokenized_prompts = clip.tokenize(default_prompt).cuda()

        with torch.no_grad():
            embedding = self.token_embedding(self.tokenized_prompts).type(dtype)

        # Save the prefix and suffix based on the default embedding
        self.register_buffer("token_prefix", embedding[:, :self.n_cls_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, self.n_cls_ctx + 1 + self.n_cls_ctx:, :])

        # Initialize class-specific context vectors
        cls_vectors = torch.empty(num_class, self.n_cls_ctx, self.ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

    def forward(self, vehicle_ids):
        """
        vehicle_ids: a batch of vehicle IDs used to retrieve corresponding features.
        """
        batch_size = len(vehicle_ids)
        dynamic_prompts = []

        for vehicle_id in vehicle_ids:
            # Access the features for each vehicle ID, defaulting to 'unknown' if not found
            features = self.vehicle_features.get(vehicle_id, {'color': 'unknown', 'type': 'unknown'})
            prompt_text = self.ctx_init.format(**features)
            tokenized_prompt = clip.tokenize(prompt_text).cuda()
            dynamic_prompts.append(tokenized_prompt)

        # Stack the tokenized prompts for the batch
        tokenized_prompts = torch.cat(dynamic_prompts, dim=0)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.cls_ctx.dtype)

        # Split the embedding into prefix and suffix parts
        prefix = embedding[:, :self.n_cls_ctx + 1, :]
        suffix = embedding[:, self.n_cls_ctx + 1 + self.n_cls_ctx:, :]

        # Retrieve class-specific context vectors for each vehicle ID
        cls_ctx = self.cls_ctx[vehicle_ids]  # Assuming vehicle_ids are mapped to class indices

        # Concatenate prefix, class-specific context, and suffix to form the complete prompt
        prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)

        return prompts


# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding, vehicle_features):
#         super().__init__()
#         self.token_embedding = token_embedding
#         self.vehicle_features = vehicle_features

#         # Define prompt template
#         if dataset_name.lower() in ["vehicleid", "veri"]:
#             self.ctx_init = "Photo of {color} {type} X X X X vehicle."
#         else:
#             self.ctx_init = "A photo of a X X X X person."
            
#         # Define possible colors and types
#         self.possible_colors = [
#             'yellow', 'orange', 'green', 'gray', 'red',
#             'blue', 'white', 'golden', 'brown', 'black'
#         ]
        
#         self.possible_types = [
#             'sedan', 'suv', 'van', 'hatchback', 'mpv',
#             'pickup', 'bus', 'truck', 'estate'
#         ]
#         # Precompute all color-type combinations
#         self.color_type_combinations = {}
#         for color, vehicle_type in product(self.possible_colors, self.possible_types):
#             prompt_str = f"{color} {vehicle_type}"
#             tokenized = clip.tokenize([prompt_str]).cuda()
#             with torch.no_grad():
#                 self.color_type_combinations[(color, vehicle_type)] = self.token_embedding(tokenized).type(dtype)

#         print(f"Number of precomputed color-type combinations: {len(self.color_type_combinations)}")

#         # Context initialization
#         self.ctx_dim = 512
#         self.n_ctx = 4

#         n_cls_ctx = 4
#         cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
#         nn.init.normal_(cls_vectors, std=0.02)
#         self.cls_ctx = nn.Parameter(cls_vectors) 
#         self.num_class = num_class
#         self.n_cls_ctx = n_cls_ctx
        
#     def forward(self, labels):
#         batch_size = labels.shape[0]

#         # Generate dynamic prefixes for each label
#         dynamic_prefix_embeddings = []
#         for label in labels:
#             vehicle_id = self.vehicle_ids[label.item()]
#             features = self.vehicle_features.get(vehicle_id, {'color': 'unknown', 'type': 'unknown'})
#             color = features.get('color', 'unknown')
#             vehicle_type = features.get('type', 'unknown')
#             prefix_color_type = self.color_type_combinations.get((color, vehicle_type))
#             if prefix_color_type is None:
#                 raise ValueError(f"Color-type combination '{color} {vehicle_type}' not found.")
#             dynamic_prefix = "Photo of " + prefix_color_type
#             ctx_init = dynamic_prefix + " X X X X vehicle."
             
#             tokenized_prompts = clip.tokenize(ctx_init).cuda() 
#             with torch.no_grad():
#                 embedding = token_embedding(tokenized_prompts).type(dtype) 
#             self.tokenized_prompts = tokenized_prompts  # torch.Tensor
            
#             self.register_buffer("token_prefix", self.tokenized_prompts[:, :self.n_ctx + 1, :])  
#             self.register_buffer("token_suffix", self.tokenized_prompts[:, self.n_ctx + 1 + self.n_cls_ctx: , :]) 
            
#             # Expand suffix to match batch size
#             suffix = self.token_suffix.expand(batch_size, -1, -1)  # Shape: (batch_size, 4, 512)
#             prefix = self.token_prefix.expand(batch_size, -1, -1) 

#             dynamic_prefix_embeddings.append(self.tokenized_prompts)
#             prompts = torch.cat([dynamic_prefix_embeddings, cls_ctx, suffix,  # (n_cls, *, dim)],dim=1,) 

#         return prompts
