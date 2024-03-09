import torch
import torch.nn.functional as F
from fairscale.nn.misc import checkpoint_wrapper

from transformers import BertModel, BertTokenizer

# import os
import json
# import stanza
# from gensim.models import KeyedVectors
import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# nltk.data.path.append('./models/nltk_data')
# from nltk.corpus import wordnet as wn

from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0))) / d_model
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # batch first
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FramePositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_frames=500):
        super(FramePositionalEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.1)

        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_frames, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码矩阵注册为模型参数
        self.register_buffer("pe", pe)

    def forward(self, x, frame_indices):
        # 对每个特征根据其帧序号添加位置编码
        pe = self.pe[frame_indices]  # 根据帧序号选择对应的位置编码
        x = x + pe
        return self.dropout(x)
    
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()
        self.hidden1=torch.nn.Linear(input_size,hidden_size)
        self.hidden2=torch.nn.Linear(hidden_size,hidden_size)
        self.output=torch.nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=self.output(x)
        return x


class VideoFeatureExtractor:
    def __init__(self, resnet, resnet_mlp, frame_pos_emb, device='cuda'):
        self.resnet = resnet
        self.resnet_mlp = resnet_mlp
        self.frame_pos_emb = frame_pos_emb
        self.device = torch.device(device)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.resnet.eval()

    # def preprocess_and_extract_features(self, frame, bboxes):
    #     processed_images = []
    #     bbox_features_list = []

    #     # Preprocess all image regions and collect bbox features
    #     for bbox in bboxes:
    #         try:
    #             xmin, ymin, xmax, ymax = bbox["bbox"]
    #             frame_image = Image.fromarray(frame.cpu().numpy().astype("uint8"), "RGB").crop((xmin, ymin, xmax, ymax))
    #             processed_image = self.preprocess(frame_image).to(self.device)
    #             processed_images.append(processed_image.unsqueeze(0).half())  # Add batch dimension
    #             bbox_features = torch.tensor(bbox["bbox"], dtype=torch.half, device=self.device)
    #             bbox_features_list.append(bbox_features.unsqueeze(0))
    #         except Exception as e:
    #             print(f"Error preprocessing image region: {e}")
    #             continue

    #     if not processed_images:  # If no valid images, return empty list
    #         return []

    #     # Batch process images
    #     batch_images = torch.cat(processed_images, dim=0)
    #     batch_bbox_features = torch.cat(bbox_features_list, dim=0)

    #     with torch.no_grad():
    #         batch_features = self.resnet(batch_images).view(batch_images.size(0), -1)
    #         combined_features = torch.cat((batch_features, batch_bbox_features), dim=1)
    #         region_tokens = self.resnet_mlp(combined_features)

    #     return region_tokens
        
    # def preprocess_and_extract_features(self, frames, bboxes_list):
    #     processed_images = []
    #     bbox_features_list = []

    #     for frame, bboxes in zip(frames, bboxes_list):
    #         try:
    #             for bbox in bboxes:
    #                 xmin, ymin, xmax, ymax = bbox["bbox"]
    #                 frame_image = Image.fromarray(frame.cpu().numpy().astype("uint8"), "RGB").crop((xmin, ymin, xmax, ymax))
    #                 processed_image = self.preprocess(frame_image).to(self.device)
    #                 processed_images.append(processed_image.unsqueeze(0).half())  # Add batch dimension
    #                 bbox_features = torch.tensor(bbox["bbox"], dtype=torch.half, device=self.device)
    #                 bbox_features_list.append(bbox_features.unsqueeze(0))
    #         except Exception as e:
    #             print(f"Error preprocessing image regions: {e}")
    #             continue

    #     if not processed_images:  # If no valid images, return empty list
    #         return []

    #     batch_images = torch.cat(processed_images, dim=0)
    #     batch_bbox_features = torch.cat(bbox_features_list, dim=0)

    #     with torch.no_grad():
    #         batch_features = self.resnet(batch_images).view(batch_images.size(0), -1)
    #         combined_features = torch.cat((batch_features, batch_bbox_features), dim=1)
    #         region_tokens = self.resnet_mlp(combined_features)

    #     return region_tokens
        
    def preprocess_and_extract_features(self, frames, bboxes_list):
        processed_images = []
        bbox_features_list = []
        indices_mapping = []
        for idx, (frame, bboxes) in enumerate(zip(frames, bboxes_list)):
            try:
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox["bbox"]
                    frame_image = Image.fromarray(frame.cpu().numpy().astype("uint8"), "RGB").crop((xmin, ymin, xmax, ymax))
                    processed_image = self.preprocess(frame_image).to(self.device)
                    processed_images.append(processed_image.unsqueeze(0).half())  # Add batch dimension
                    bbox_features = torch.tensor(bbox["bbox"], dtype=torch.half, device=self.device)
                    bbox_features_list.append(bbox_features.unsqueeze(0))
                    indices_mapping.append(idx)#记录被处理的region的索引，异常的不被记录
            except Exception as e:
                print(f"Error preprocessing image regions: {e}")
                continue
        if not processed_images:  # If no valid images, return empty list
            return [], []
        batch_images = torch.cat(processed_images, dim=0)
        batch_bbox_features = torch.cat(bbox_features_list, dim=0)
        with torch.no_grad():
            batch_features = self.resnet(batch_images).view(batch_images.size(0), -1)
            combined_features = torch.cat((batch_features, batch_bbox_features), dim=1)
            region_tokens = self.resnet_mlp(combined_features)

        return region_tokens, indices_mapping
    
    def get_region_feats(self, video_id, raw_frames):
        with open(f'datasets/MSRVTT-v2/objects/32frames/filtered/{video_id}.json', 'r') as f:
            obj_dic_filtered = json.load(f)

        frame_indices = []
        frames = []

        for frame_idx, bboxes in obj_dic_filtered.items():
            # try:
            frame = raw_frames[int(frame_idx) - 1]
            frame = torch.permute(frame, (1, 2, 0))  # Adjust frame dimensions if necessary
            frames.append(frame)
            frame_indices.extend([int(frame_idx) - 1] * len(bboxes))
            # except Exception as e:
            #     print(f"Error processing frame {frame_idx}: {e}")
            #     continue
        tokens, indices_mapping = self.preprocess_and_extract_features(frames, obj_dic_filtered.values())
        mapped_frame_indices = [frame_indices[i] for i in indices_mapping]
        return self.prepare_output(tokens, mapped_frame_indices)
    # def get_region_feats(self, video_id, raw_frames):
    #     with open(f'datasets/MSRVTT-v2/objects/32frames/filtered/{video_id}.json', 'r') as f:
    #         obj_dic_filtered = json.load(f)

    #     region_tokens = []
    #     frame_indices = []

    #     for frame_idx, bboxes in obj_dic_filtered.items():
    #         try:
    #             frame = raw_frames[int(frame_idx) - 1]
    #             frame = torch.permute(frame, (1, 2, 0))  # Adjust frame dimensions if necessary
    #         except Exception as e:
    #             print(f"Error processing frame {frame_idx}: {e}")
    #             continue

    #         tokens = self.preprocess_and_extract_features(frame, bboxes)
    #         region_tokens.extend(tokens)
    #         frame_indices.extend([int(frame_idx) - 1] * len(tokens))

    #     return self.prepare_output(region_tokens, frame_indices)

    def get_region_feats(self, video_id, raw_frames):
        with open(f'datasets/MSRVTT-v2/objects/32frames/filtered/{video_id}.json', 'r') as f:
            obj_dic_filtered = json.load(f)
        frame_indices = []
        frames = []
        for frame_idx, bboxes in obj_dic_filtered.items():
            try:
                frame = raw_frames[int(frame_idx) - 1]
                frame = torch.permute(frame, (1, 2, 0))  # Adjust frame dimensions if necessary
                frames.append(frame)
                frame_indices.extend([int(frame_idx) - 1] * len(bboxes))
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        tokens = self.preprocess_and_extract_features(frames, obj_dic_filtered.values())
        return self.prepare_output(tokens, frame_indices)


    def pad_sequence(self, tensor, max_len=220):
        current_size = tensor.size(0)
        padding_size = max(max_len - current_size, 0)
        if current_size > max_len:
            tensor = tensor[:max_len]
            attention_mask = torch.zeros(max_len, dtype=torch.bool, device=self.device)
        else:
            padding = torch.zeros((padding_size, tensor.size(1)), device=self.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding), dim=0)
            attention_mask = torch.cat((torch.zeros(current_size, dtype=torch.bool, device=self.device),
                                        torch.ones(padding_size, dtype=torch.bool, device=self.device)))
        return tensor, attention_mask

    def prepare_output(self, region_tokens, frame_indices):
        feature_dim = 768  # Assuming feature_dim is known or obtained from resnet_mlp output
        if not isinstance(region_tokens,torch.Tensor):#[176,768] tensor
            resized_tensor = torch.zeros((220, feature_dim), device=self.device, dtype=torch.float16)
            attention_mask = torch.full((220,), True, dtype=torch.bool, device=self.device)
        else:
            region_tokens = self.frame_pos_emb(region_tokens, frame_indices)
            resized_tensor, attention_mask = self.pad_sequence(region_tokens)

        return resized_tensor.half(), attention_mask

class MultimodalTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(MultimodalTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)

        self.bert = BertModel.from_pretrained(
            "./models/bert_pretrained/bert-base-uncased"
        )
        for param in self.bert.parameters():
            param.requires_grad = False
            # 采用pre_ln试试效果，与post_ln做对比，在大部分指标上（除了SPICE）没有post_ln好。
        self.frames_cap_temporal_enc = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=768, nhead=8, activation="gelu", batch_first=True
            ),
            num_layers=6,
        )
        self.obj_region_enc = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=768, nhead=12, activation="gelu",batch_first=True
            ),
            num_layers=10,
        )

        self.pos_emb = PositionalEmbedding(d_model=768)
        self.frame_pos_emb = FramePositionalEmbedding(d_model=768)

        # self.nlp = stanza.Pipeline(
        #     lang="en",
        #     processors="tokenize,pos",
        #     dir="./models/stanfordnlp/stanza/",
        #     download_method=None,
        # )
        # self.word2vec = KeyedVectors.load_word2vec_format(
        #     "./models/word2vec/GoogleNews-vectors-negative300.bin.gz", binary=True
        # )

        self.resnet = models.resnet101(pretrained=True)
        # 移除avgpool和fc层
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.resnet_linear = torch.nn.Linear(2048 * 7 * 7+4, 768)#感觉这里一个linear层不太行，要融合信息最好还是用一个MLP
        self.resnet_mlp = MLP(input_size=2048 * 7 * 7+4,hidden_size=768*2,output_size=768)

        self.video_feature_extractor = VideoFeatureExtractor(resnet=self.resnet,resnet_mlp=self.resnet_mlp,frame_pos_emb=self.frame_pos_emb)

        self.compute_mask_on_the_fly = False  # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        self.max_frames_cap_num = args.max_frames_cap_num

        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, "learn_mask_enabled", False)
        self.sparse_mask_soft2hard = getattr(args, "sparse_mask_soft2hard", False)

        if self.learn_mask_enabled == True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)#Embedding[784*784,1]

            # self.learn_vid_att = torch.nn.Embedding((args.max_img_seq_length + args.max_frames_cap_num)*(args.max_img_seq_length + args.max_frames_cap_num),1)
            #不对obj_region使用mask
            # self.learn_vid_att = torch.nn.Embedding(
            #     (args.max_img_seq_length + args.max_frames_cap_num)
            #     * (args.max_img_seq_length + args.max_frames_cap_num),
            #     1,
            # )
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs["img_feats"]
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(
            0, 2, 1, 3, 4
        )  # shape=[6,3,32,224,224]#6个视频，3通道，每个视频采样32帧

        frames_cap_ids = kwargs["frames_cap_ids"]  # shape=[6,32,50]
        frames_cap_att_masks = kwargs["frames_cap_attention_masks"]  # shape=[6,32,50]
        # frames_cap_pad_masked_feats = []
        frames_sentence_feats = []
        with torch.no_grad():
            for i in range(frames_cap_ids.shape[0]):
                frames_cap_feat = self.bert(
                    input_ids=frames_cap_ids[i], attention_mask=frames_cap_att_masks[i]
                ).last_hidden_state  # shape=[32,50,768]
                # 取每个句子首位CLS特征来表征句子特征
                frames_sentence_feat = frames_cap_feat[:, 0, :]
                frames_sentence_feats.append(frames_sentence_feat)

                # # 创建一个与frames_cap_feat相同维度的全零张量
                # zero_tensor = torch.zeros_like(frames_cap_feat)
                # # 将frames_cap_att_masks[i]为零的位置对应的特征置为全零
                # frames_cap_feat_masked = torch.where(frames_cap_att_masks[i].unsqueeze(-1) == 0, zero_tensor, frames_cap_feat)
                # frames_cap_pad_masked_feats.append(frames_cap_feat_masked)

            # frames_cap_pad_masked_feats = torch.stack(frames_cap_pad_masked_feats)#shape=[6,32,50,768]
        frames_sentence_feats = torch.stack(
            frames_sentence_feats
        )  # shape=[6,32,768],得到每一帧描述级别的特征

        # 下面这行应该是之前实现的一个bug，原始特征被加了两次
        # frames_sentence_feats_enc = self.frames_cap_temporal_enc(frames_sentence_feats + self.pos_emb(frames_sentence_feats))#shape=[6,32,768]

        frames_sentence_feats_enc = self.frames_cap_temporal_enc(
            self.pos_emb(frames_sentence_feats)
        )  # shape=[6,32,768]
        kwargs["frames_sentence_feats_enc"] = frames_sentence_feats_enc

        # frames_caps = kwargs["frames_cap"]  # shape=[32,n]
        video_ids = kwargs["video_id"]
        raw_frames = kwargs["raw_frames"]
        region_feats = []
        region_attention_mask = []
        for i in range(0, len(video_ids)):
            f,m = self.video_feature_extractor.get_region_feats(video_id=video_ids[i],raw_frames=raw_frames[i])
            # f = torch.zeros((220,768)).to(torch.device("cuda")).half()
            # m = torch.full((220,),True,dtype=torch.bool).to(torch.device("cuda"))
            region_feats.append(f)
            region_attention_mask.append(m)
        region_attention_mask = torch.stack(region_attention_mask)
        region_feats = torch.stack(region_feats)

        for i in range(0,region_attention_mask.shape[0]):
            if region_attention_mask[i][0]==True:
                region_attention_mask[i][0]=False#将region的第一个token设置为始终可见，不然可能会报错，希望这样的更改对结果的影响不会太大
                
        region_feats = self.obj_region_enc(region_feats,src_key_padding_mask=region_attention_mask)
        kwargs["region_feats"] = region_feats

        vid_feats = self.swin(images)  # shape=[6,1024,16,7,7]，输出为[B,d,T/2,H/32.W/32]

        if self.use_grid_feat == True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)  # shape=[6,16,7,7,1024]
        vid_feats = vid_feats.view(
            B, -1, self.latent_feat_size
        )  # 改变张量的形状而不改变其数据，self.latent_feat_size=1024。[6,16,7,7,1024]->[6,784,1024]
        vid_feats = self.fc(vid_feats)  # shape=[6,784,512]
        # prepare VL transformer inputs
        kwargs["img_feats"] = vid_feats
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)

        # 原始SwinBERT稀疏注意力mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)#shape=[784,784]
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()#shape=[784,784],初始化一个对角矩阵，对角线为1，其余部分为0
            video_attention = (1. - diag_mask)*learn_att#这一操作移除了自注意力
            learn_att = diag_mask + video_attention#这一操作又将自注意力设置为了1
            #learn_att 成为了一个融合了原始自注意力（对角线上的元素）和调整后的元素之间的注意力（非对角线元素）的矩阵。这样的设计可以让模型在保留对自身信息的关注的同时，也考虑到了其他元素之间的关系。这种方法在处理序列数据时，特别是在视觉和语言任务中，通常能够提高模型的性能。
            if self.sparse_mask_soft2hard:#论文给的数据使用soft的mask更多的时候能取得更好的效果
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att#attention mask shape = [batch_size,784+50,784+50]

            #串联上frames_cap(32)和obj_regions(32)的mask依照以下原则:
            #1.caption token, img token可以看到所有frame token, 有效的region token
            #2.frame token，有效的region token 能看到img token，有效的 rgion token，frame token
            expanded_attention_mask=torch.zeros((kwargs['attention_mask'].shape[0],kwargs['attention_mask'].shape[1]+32+220,kwargs['attention_mask'].shape[2]+32+220)).to(torch.device("cuda"))#初始化一个全mask的mask矩阵
            expanded_attention_mask[:,:kwargs['attention_mask'].shape[1],:kwargs['attention_mask'].shape[2]]=kwargs['attention_mask']

            expanded_attention_mask[:,:kwargs['attention_mask'].shape[1],kwargs['attention_mask'].shape[2]:]=1#1
            expanded_attention_mask[:,kwargs['attention_mask'].shape[1]:,50:]=1#2
            #处理region的padding
            for i in range(0,region_attention_mask.shape[0]):
                expanded_attention_mask[i,:,kwargs['attention_mask'].shape[1]+32:]=torch.logical_not(region_attention_mask[i])
            kwargs['attention_mask']=expanded_attention_mask



        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)
            outputs = outputs + (loss_sparsity,)
        return outputs
    
    # def get_region_feats(self,video_id,raw_frames):
    #     self.resnet.eval()
    #     with open('datasets/MSRVTT-v2/objects/32frames/filtered/' + video_id + '.json','r') as f:
    #         obj_dic_filtered = json.load(f)

    #     # 跟据obj_dic_filtered提取对应帧的对应区域，以帧顺序作为时序编码信息，obj的两个对点坐标作为空间位置信息，使用resnet提取区域特征，一块区域作为一个token
    #     # 定义图像预处理步骤
    #     preprocess = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),
    #         ]
    #     )
    #     frame_features = {}
    #     device = torch.device("cuda")
    #     for frame_idx, bboxes in obj_dic_filtered.items():
    #         frame_feature_list = []
    #         for bbox in bboxes:
    #             try:
    #                 # 裁剪图像区域
    #                 xmin, ymin, xmax, ymax = bbox["bbox"]

    #                 frame = raw_frames[int(frame_idx) - 1]
    #                 frame = torch.permute(frame, (1, 2, 0))
    #                 frame_image = Image.fromarray(
    #                     frame.numpy().astype("uint8"), "RGB"
    #                 ).crop((xmin, ymin, xmax, ymax))

    #                 # 图像预处理并提取特征
    #                 input_tensor = preprocess(frame_image).half()
    #                 input_batch = input_tensor.unsqueeze(0)  # 创建一个batch
    #                 input_batch = input_batch.to(device)
    #                 with torch.no_grad():
    #                     features = self.resnet(input_batch)
    #                     # 展平特征张量
    #                     features = features.view(features.size(0), -1)
    #                     # 特征与bbox特征拼接
    #                     bbox_features = torch.tensor(bbox["bbox"], dtype=torch.float16).to(
    #                         device
    #                     )
    #                     combined_features = torch.cat((features.squeeze(0), bbox_features), 0)

    #                 #从这断开，前面的结果可以保存，从这开始就需要进行代码优化了。

    #                 features = self.resnet_mlp(combined_features)#先与坐标串联，再mlp，再加位置编码

    #                 frame_feature_list.append(features)
    #             except Exception:
    #                 continue


    #         frame_features[frame_idx] = frame_feature_list
    #     region_tokens = []
    #     frame_indices = []
    #     for frame_idx, features in frame_features.items():
    #         for feature in features:
    #             region_tokens.append(feature)
    #             frame_indices.append(int(frame_idx) - 1)  # 帧序号减1，因为位置编码是从0开始的
    #     # 将tokens转换为Tensor
    #     if not region_tokens:
    #         feature_dim=768
    #         resized_tensor = torch.zeros((220,feature_dim)).to(device)
    #         attention_mask=torch.full((220,),True,dtype=torch.bool).to(device)
    #     else:
    #         region_tokens_tensor = torch.stack(region_tokens).to(
    #             device
    #         )  # shape: [num_tokens, feature_dim]
    #         region_tokens_tensor = self.frame_pos_emb(region_tokens_tensor, frame_indices)
        
    #         # 固定序列长度
    #         current_size = region_tokens_tensor.shape[0]
    #         padding_size = max(220 - current_size, 0)  # 固定长度到220
    #         if current_size > 220:
    #             resized_tensor = region_tokens_tensor[:220, :]
    #             attention_mask =torch.full((220,),False,dtype=torch.bool).to(torch.device("cuda"))#没有mask的部分用0填充
    #         else:
    #             padding = torch.zeros(
    #                 (padding_size, region_tokens_tensor.shape[1]),
    #                 dtype=region_tokens_tensor.dtype,
    #             ).to(torch.device("cuda"))
    #             resized_tensor = torch.cat((region_tokens_tensor, padding), dim=0)
    #             attention_mask = torch.cat((torch.full((current_size,),False,dtype=torch.bool),torch.full((220-current_size,),True,dtype=torch.bool))).to(torch.device("cuda"))

    #     resized_tensor = resized_tensor.half()
    #     return resized_tensor,attention_mask#[220,d_model]
        
        



    # def get_reigion_feats(self, frames_cap, index, video_id, raw_frames):
    #     frames_cap_str = ""
    #     for i in range(0, len(frames_cap)):
    #         frames_cap_str += " " + frames_cap[i][index]
    #     # 处理文本
    #     doc = self.nlp(frames_cap_str)

    #     # 提取名词
    #     nouns = [
    #         word.text
    #         for sent in doc.sentences
    #         for word in sent.words
    #         if word.upos == "NOUN"
    #     ]
    #     nouns_num_dic = dict()
    #     for noun in nouns:
    #         nouns_num_dic.setdefault(noun, 0)
    #         nouns_num_dic[noun] += 1
    #     nouns_unit = list(nouns_num_dic.keys())  # 去除重复单词

    #     ##过滤出wordnet中属于object的单词
    #     def is_word_under_object(word):
    #         # 查找word的所有同义词集
    #         synsets = wn.synsets(word)

    #         # 对于每个同义词集
    #         for synset in synsets:
    #             # 获取该同义词集到'entity.n.01'（最顶层实体）的hypernym_paths
    #             hypernym_paths = synset.hypernym_paths()

    #             # 检查每条路径
    #             for path in hypernym_paths:
    #                 # 检查路径中是否有'object'这个上位词
    #                 for hypernym in path:
    #                     if hypernym.name().split(".")[0] == "object":
    #                         return True
    #         return False

    #     nouns_unit = [noun for noun in nouns_unit if is_word_under_object(noun)]
    #     nouns_unit_vec = np.array([self.word2vec[word] for word in nouns_unit if word in self.word2vec])
    #     # 计算余弦相似度矩阵
    #     similarity_matrix = cosine_similarity(nouns_unit_vec)
    #     similarity_matrix = np.clip(similarity_matrix, -1, 1)
    #     # 将相似度转换为距离（DBSCAN需要距离矩阵）
    #     distance_matrix = 1 - similarity_matrix
    #     # 应用DBSCAN聚类
    #     # eps参数设置为0.5，表示如果两个点的距离小于0.5（即相似度大于0.5），则它们属于同一个聚类
    #     # min_samples设置为1，因为我们不需要考虑最小样本数的限制
    #     dbscan = DBSCAN(eps=0.5, min_samples=1, metric="precomputed")
    #     clusters = dbscan.fit_predict(distance_matrix)
    #     # 初始化聚类计数字典，用于存储每个聚类的样本数量
    #     cluster_counts = {}

    #     # 遍历每个单词及其对应的聚类标签
    #     for word, cluster_label in zip(nouns_unit, clusters):
    #         # 如果当前聚类标签还未在字典中，则初始化计数为0
    #         if cluster_label not in cluster_counts:
    #             cluster_counts[cluster_label] = 0
    #         # 累加当前单词的数量到对应的聚类计数上
    #         cluster_counts[cluster_label] += nouns_num_dic[word]
    #     # 筛选出样本数量前10的聚类
    #     cluster_counts_list = sorted(
    #         cluster_counts.items(), key=lambda x: x[1], reverse=True
    #     )
    #     top_clusters = [item[0] for item in cluster_counts_list[:10]]
    #     filtered_nouns_unit = []
    #     # 遍历每个名词及其对应的聚类标签
    #     for word, cluster_label in zip(nouns_unit, clusters):
    #         if cluster_label in top_clusters:
    #             filtered_nouns_unit.append(word)
    #     # 遍历每个名词向量及其对应的聚类标签
    #     filtered_nouns_unit_vec = []
    #     for word_vec, cluster_label in zip(nouns_unit_vec, clusters):
    #         if cluster_label in top_clusters:
    #             filtered_nouns_unit_vec.append(word_vec)

    #     # 获取faster_rcnn输出结果
    #     obj_file_path = (
    #         "datasets/MSRVTT-v2/objects/32frames/faster_rcnn/"
    #         + video_id
    #         + ".json"
    #     )
    #     with open(obj_file_path, "r") as f:
    #         obj_dic = json.load(f)

    #     def is_in_top_clusters(label, filtered_nouns_unit_vec, eps=0.5):
    #         labels = label.split(" ")
    #         for l in labels:
    #             l_vec = self.word2vec[l].reshape(1, -1)
    #             # 计算新单词与已有单词列表中每个单词的余弦相似度
    #             sim = cosine_similarity(l_vec, filtered_nouns_unit_vec)
    #             dis = 1 - sim
    #             min_dis = np.min(dis)
    #             if min_dis < eps:
    #                 return True
    #             else:
    #                 return False

    #     def filter_obj_dic(obj_dic, filtered_nouns_unit_vec, eps=0.5):
    #         obj_dic_filtered = dict()
    #         for key in obj_dic.keys():
    #             frame_obj_arr = []
    #             if obj_dic[key] != None:
    #                 for obj in obj_dic[key]:
    #                     if is_in_top_clusters(obj["cls"], filtered_nouns_unit_vec, eps):
    #                         frame_obj_arr.append(obj)
    #             obj_dic_filtered[key] = frame_obj_arr
    #         return obj_dic_filtered

    #     obj_dic_filtered = filter_obj_dic(obj_dic, filtered_nouns_unit_vec, dbscan.eps)
    #     # 跟据obj_dic_filtered提取对应帧的对应区域，以帧顺序作为时序编码信息，obj的两个对点坐标作为空间位置信息，使用resnet提取区域特征，一块区域作为一个token

    #     self.resnet.eval()
    #     # 定义图像预处理步骤
    #     preprocess = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),
    #         ]
    #     )
    #     frame_features = {}
    #     device = torch.device("cuda")
    #     for frame_idx, bboxes in obj_dic_filtered.items():
    #         frame_feature_list = []
    #         for bbox in bboxes:
    #             try:
    #                 # 裁剪图像区域
    #                 xmin, ymin, xmax, ymax = bbox["bbox"]

    #                 frame = raw_frames[int(frame_idx) - 1]
    #                 frame = torch.permute(frame, (1, 2, 0))
    #                 frame_image = Image.fromarray(
    #                     frame.numpy().astype("uint8"), "RGB"
    #                 ).crop((xmin, ymin, xmax, ymax))

    #                 # 图像预处理并提取特征
    #                 input_tensor = preprocess(frame_image).half()
    #                 input_batch = input_tensor.unsqueeze(0)  # 创建一个batch
    #                 input_batch = input_batch.to(device)
    #                 with torch.no_grad():
    #                     features = self.resnet(input_batch)
    #                     # 展平特征张量
    #                     features = features.view(features.size(0), -1)
    #                     # 特征与bbox特征拼接
    #                     bbox_features = torch.tensor(bbox["bbox"], dtype=torch.float16).to(
    #                         device
    #                     )
    #                     combined_features = torch.cat((features.squeeze(0), bbox_features), 0)

    #                 #从这断开，前面的结果可以保存，从这开始就需要进行代码优化了。

    #                 features = self.resnet_mlp(combined_features)#先与坐标串联，再mlp，再加位置编码

    #                 frame_feature_list.append(features)
    #             except Exception:
    #                 continue


    #         frame_features[frame_idx] = frame_feature_list
    #     region_tokens = []
    #     frame_indices = []
    #     for frame_idx, features in frame_features.items():
    #         for feature in features:
    #             region_tokens.append(feature)
    #             frame_indices.append(int(frame_idx) - 1)  # 帧序号减1，因为位置编码是从0开始的
    #     # 将tokens转换为Tensor
    #     if not region_tokens:
    #         feature_dim=768
    #         resized_tensor = torch.zeros((220,feature_dim)).to(device)
    #         attention_mask=torch.full((220,),True,dtype=torch.bool).to(device)
    #     else:
    #         region_tokens_tensor = torch.stack(region_tokens).to(
    #             device
    #         )  # shape: [num_tokens, feature_dim]
    #         region_tokens_tensor = self.frame_pos_emb(region_tokens_tensor, frame_indices)
        
    #         # 固定序列长度
    #         current_size = region_tokens_tensor.shape[0]
    #         padding_size = max(220 - current_size, 0)  # 固定长度到220
    #         if current_size > 220:
    #             resized_tensor = region_tokens_tensor[:220, :]
    #             attention_mask =torch.full((220,),False,dtype=torch.bool).to(torch.device("cuda"))#没有mask的部分用0填充
    #         else:
    #             padding = torch.zeros(
    #                 (padding_size, region_tokens_tensor.shape[1]),
    #                 dtype=region_tokens_tensor.dtype,
    #             ).to(torch.device("cuda"))
    #             resized_tensor = torch.cat((region_tokens_tensor, padding), dim=0)
    #             attention_mask = torch.cat((torch.full((current_size,),False,dtype=torch.bool),torch.full((220-current_size,),True,dtype=torch.bool))).to(torch.device("cuda"))

    #     resized_tensor = resized_tensor.half()
    #     return resized_tensor,attention_mask#[220,d_model]

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += torch.mean(torch.abs(video_attention))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy

        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens
        )
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length / pretrained_num_tokens

        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[
                    pretrained_num_tokens * i : pretrained_num_tokens * (i + 1),
                    pretrained_num_tokens * i : pretrained_num_tokens * (i + 1),
                ] = pretrained_learn_att

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print("init attn mask with bilinear interpolation")
        import numpy

        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens
        )
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        scale_factor = int(self.max_img_seq_length / pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None, None, :, :].double())[
                0, 0, :, :
            ].half()

    def random_init_attn_mask(self):
        print("random init attn mask")
        self.learn_vid_att = torch.nn.Embedding(
            self.max_img_seq_length * self.max_img_seq_length, 1
        )

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy

        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens
        )
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[
                    pretrained_num_tokens * i : pretrained_num_tokens * (i + 1),
                    pretrained_num_tokens * i : pretrained_num_tokens * (i + 1),
                ] = pretrained_learn_att

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad = not freeze
