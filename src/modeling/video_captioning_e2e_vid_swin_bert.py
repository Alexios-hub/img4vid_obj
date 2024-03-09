import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
from transformers import BertModel, BertTokenizer


class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
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

        # self.bert = BertModel.from_pretrained('./models/bert_pretrained/bert-base-uncased')
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)#shape=[6,3,32,224,224]#6个视频，3通道，每个视频采样32帧

        # frames_cap_ids = kwargs['frames_cap_ids'] #shape=[6,32,50]
        # frames_cap_att_masks = kwargs['frames_cap_attention_masks']#shape=[6,32,50]
        # frames_cap_pad_masked_feats = []
        # for i in range(frames_cap_ids.shape[0]):
        #     frames_cap_feat = self.bert(input_ids = frames_cap_ids[i],attention_mask = frames_cap_att_masks[i] ).last_hidden_state
        #     # 创建一个与frames_cap_feat相同维度的全零张量
        #     zero_tensor = torch.zeros_like(frames_cap_feat)
        #     # 将frames_cap_att_masks[i]为零的位置对应的特征置为全零
        #     frames_cap_feat_masked = torch.where(frames_cap_att_masks[i].unsqueeze(-1) == 0, zero_tensor, frames_cap_feat)
        #     frames_cap_pad_masked_feats.append(frames_cap_feat_masked)
        # frames_cap_pad_masked_feats = torch.stack(frames_cap_pad_masked_feats)

        vid_feats = self.swin(images)#shape=[6,1024,16,7,7]，输出为[B,d,T/2,H/32.W/32]
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)#shape=[6,16,7,7,1024]
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)#改变张量的形状而不改变其数据，self.latent_feat_size=1024。[6,16,7,7,1024]->[6,784,1024]
        vid_feats = self.fc(vid_feats)#shape=[6,784,512]
        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        # learn soft attention mask
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
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs = outputs + (loss_sparsity, )          
        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 


    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)


    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 