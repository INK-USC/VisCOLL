import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from nets.EXPERT.masker import Masker
import nets.EXPERT.models

from .VLBERT.common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertForPretraining
from .VLBERT.pretrain.function.config import config as vlbert_cfg
from .VLBERT.pretrain.function.config import update_config
from .LXMERT.lxrt.tokenization import BertTokenizer
from .LXMERT.lxrt.modeling import LXRTPretraining, BertConfig
from .LXMERT.lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG
import random

device = torch.device("cuda")

SOS_token = 1
EOS_token = 3
MAX_LENGTH = 50


class MLMModel(nn.Module):
    def __init__(self, cfg, init, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = nets.EXPERT.models.VLBertTokenizer.from_pretrained(cfg.EXTERNAL.MLM.TOKENIZER)
        self.model = nets.EXPERT.models.load_arch(
            cfg.EXTERNAL.MLM.CONFIG_PATH,
            pretrained=False,
            tok=self.tokenizer
        )
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.shown = False

    def show_sample(self,images, captions):
        if not self.shown:
            self.shown = True
            print(captions[:5])

    def forward(self, images, captions, caption_lens, labels, decode_strategy=None, reduce=True, img_emb=None):
        #print("batch")
        device = images.device
        text, text_len, lm_labels = captions, caption_lens, labels

        psuedo_pos_ids = torch.zeros(*list(images.size()), 2).long().to(device)
        psuedo_img_attn_mask = torch.ones(images.size(0),1).to(device)
        psuedo_imgs_len = torch.ones(images.size(0)).long().to(device)
        text_attn_mask = torch.arange(self.cfg.EXTERNAL.MLM.MAX_TXT_SEQ_LEN, device=device)[None, :] < text_len[:, None]
        text_attn_mask, psuedo_img_attn_mask = text_attn_mask.long(), psuedo_img_attn_mask.long()
        attn_mask = torch.cat((text_attn_mask[:, :1], psuedo_img_attn_mask, text_attn_mask[:, 1:]), dim=1)

        lm_preds, _, img_embedding, *_ = self.model(
            images, text, psuedo_pos_ids, 
            attention_mask=attn_mask, 
            img_lens=psuedo_imgs_len,
            txt_lens=text_len,
            img_emb=img_emb
        )


        if reduce:
            loss = self.lm_criterion(lm_preds.view(-1, lm_preds.size(-1)), lm_labels.view(-1))
            return {
                'loss': loss,
                'score': lm_preds,
                'feat': img_embedding
            }
        else:
            loss = F.cross_entropy(lm_preds.view(-1, lm_preds.size(-1)), lm_labels.view(-1), ignore_index=-1,
                                   reduction='none')
            mask_cnts = []
            for b in range(labels.size(0)):
                cnt = 0
                for t in range(labels.size(1)):
                    if labels[b,t].item() != -1:
                        cnt += 1
                mask_cnts.append(cnt)
            return {
                'loss': loss,
                'score': lm_preds,
                'feat': img_embedding,
                'mask_cnts': mask_cnts,
            }

    def init(self):
        # if hasattr(self.cfg.EXTERNAL, 'FREEZE_RESNET') and self.cfg.EXTERNAL.FREEZE_RESNET:
        #     for param in self.resnet.parameters():
        #         param.requires_grad = False
        pass

    def forward_from_feat(self, feat, y, reduce=True, **kwargs):
        captions, caption_lens, labels = y
        return self.forward(kwargs['images'], captions, caption_lens, labels, reduce=reduce, img_emb=feat)


class VLBERTModel(MLMModel):
    """
    MLM model with pretrained spatial features and bbox features as inputs
    """
    def __init__(self, cfg, init, spatial_feat_dim=2048, object_feat_dim=2048, *args, **kwargs):
        super().__init__(cfg, init, *args, **kwargs)
        self.cfg = cfg
        update_config('nets/VLBERT/cfgs/pretrain/base_e2e_16x16G_fp16.yaml')
        self.vlbert_cfg = vlbert_cfg
        self.tokenizer = nets.EXPERT.models.VLBertTokenizer.from_pretrained(cfg.EXTERNAL.MLM.TOKENIZER)
        self.object_linguistic_embeddings = nn.Embedding(1, vlbert_cfg.NETWORK.VLBERT.hidden_size)

        self.model = VisualLinguisticBertForPretraining(self.vlbert_cfg.NETWORK.VLBERT, with_rel_head=False, with_mvrc_head=False,
                                                        with_mlm_head=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_object_feats = nn.Linear(object_feat_dim, vlbert_cfg.NETWORK.VLBERT.visual_size)

        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, bbox_feats, bboxes, captions, caption_lens, labels, reduce=True, **kwargs):
        device = bbox_feats.device
        batch_size = bbox_feats.size(0)
        text, text_len, lm_labels = captions, caption_lens, labels

        bbox_feats = self.linear_object_feats(bbox_feats)

        # handle vlbert
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)
        text_visual_embeddings = self._collect_obj_reps(text_tags, bbox_feats)

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            bboxes.new_zeros((bboxes.shape[0], bboxes.shape[1])).long()
        )
        #if self.config.NETWORK.WITH_MVRC_LOSS:
        #    object_linguistic_embeddings[mvrc_ops == 1] = self.object_mask_word_embedding.weight[0]
        object_vl_embeddings = torch.cat((bbox_feats, object_linguistic_embeddings), -1)
        box_mask = self._build_bbox_masks(bboxes)

        relationship_logits, mlm_logits, mvrc_logits = self.model(
            text_input_ids,
            text_token_type_ids,
            text_visual_embeddings,
            text_mask,
            object_vl_embeddings,
            box_mask
        )
        if mlm_logits.size(1) < lm_labels.size(1):
            lm_labels = lm_labels[:, :mlm_logits.size(1)]
        else:
            mlm_logits = mlm_logits[:, :lm_labels.size(1)]
        if reduce:
            loss = self.lm_criterion(mlm_logits.contiguous().view(-1, mlm_logits.size(-1)),
                                     lm_labels.contiguous().view(-1))
            return {
                'loss': loss,
                'score': mlm_logits,
            }
        else:
            loss = F.cross_entropy(mlm_logits.contiguous().view(-1, mlm_logits.size(-1)),
                                   lm_labels.contiguous().view(-1), ignore_index=-1,
                                   reduction='none')
            mask_cnts = []
            labels_np = labels.cpu().numpy()
            for b in range(labels_np.shape[0]):
                cnt = 0
                for t in range(labels_np.shape[1]):
                    if labels[b,t] != -1:
                        cnt += 1
                mask_cnts.append(cnt)
            return {
                'loss': loss,
                'score': mlm_logits,
                'mask_cnts': mask_cnts,
            }


    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def _build_bbox_masks(self, bboxes, max_box=100):
        #masks = torch.zeros(box_nums.size(0), max_box).bool()
        #for b in range(box_nums.size(0)):
        #    masks[b, :box_nums[b]] = False
        #return masks
        return bboxes.sum(-1) > 1e-6



class LXMERTModel(MLMModel):
    """
    LXMERT
    """
    def __init__(self, cfg, init, spatial_feat_dim=2048, object_feat_dim=2048, *args, **kwargs):
        super().__init__(cfg, init, *args, **kwargs)
        self.cfg = cfg

        config = BertConfig(
            30600, # not 30522. need add +2
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=3,
            initializer_range=0.02
        )

        #self.tokenizer = BertTokenizer.from_pretrained(
        #    cfg.EXTERNAL.MLM.TOKENIZER,
        #    do_lower_case=True
        #)

        self.tokenizer = nets.EXPERT.models.VLBertTokenizer.from_pretrained(cfg.EXTERNAL.MLM.TOKENIZER)

        VISUAL_CONFIG.l_layers = 9
        VISUAL_CONFIG.x_layers = 5
        VISUAL_CONFIG.r_layers = 5
        # num_answers is only for task_qa. default is 2 so that I just set num_answers to 2
        self.model = LXRTPretraining(
            config,
            task_mask_lm=True,
            task_obj_predict=False,
            task_matched=False,
            task_qa=False,
            visual_losses='obj,attr,feat',
            num_answers=2
        )

        self.model.apply(self.model.init_bert_weights)

    def forward(self, bbox_feats, bboxes, captions, caption_lens, labels, reduce=True, **kwargs):

        # language_inputs
        text, text_len, lm_labels = captions, caption_lens, labels
        text_input_ids = text
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        loss, losses, mlm_logits = self.model(
            text_input_ids, text_token_type_ids, text_mask, lm_labels, bbox_feats, bboxes, None, None, None,
            reduce=reduce
        )

        if reduce:
            return {
                'loss': loss,
                'score': mlm_logits,
            }
        else:
            mask_cnts = []
            for b in range(labels.size(0)):
                cnt = 0
                for t in range(labels.size(1)):
                    if labels[b,t].item() != -1:
                        cnt += 1
                mask_cnts.append(cnt)
            return {
                'loss': loss,
                'score': mlm_logits,
                'mask_cnts': mask_cnts,
            }
