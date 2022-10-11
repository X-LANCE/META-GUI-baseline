import torch
import torch.nn as nn
from config import Config
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertConfig, BertAttention
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from queue import Queue
import math
from torchvision.ops import roi_pool
from torchvision.models.resnet import ResNet
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ActionTypeClassifier(nn.Module):
    def __init__(self, config: Config):
        super(ActionTypeClassifier, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.action_size)

    def forward(self, pooled_hidden_states):
        """
        :param pooled_hidden_states: [batch_size, hidden_size]
        :return:
        """
        outputs = self.dropout(pooled_hidden_states)
        outputs = self.linear(outputs)

        return outputs


class TypingClassifier(nn.Module):
    def __init__(self, config: Config):
        super(TypingClassifier, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = nn.Tanh()
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, dialog_encoder_hidden_states):
        """
        :param dialog_encoder_hidden_states: [batch_size, dialog_seq_length, hidden_size]
        :return:
        """
        dialog_encoder_hidden_states = self.act_fn(self.linear(dialog_encoder_hidden_states))
        logits = self.qa_outputs(dialog_encoder_hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits


class ItemScoreClassifier(nn.Module):
    def __init__(self, config: Config):
        super(ItemScoreClassifier, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, item_encoder_hidden_states):
        """
        :param item_encoder_hidden_states: [batch_size, item_encoding_length, hidden_size]
        :return:
        """
        batch_size = item_encoder_hidden_states.shape[0]
        item_encoder_hidden_states = self.act_fn(self.linear(item_encoder_hidden_states))
        item_encoder_hidden_states = item_encoder_hidden_states.view(batch_size, -1, self.config.item_embedding_length,
                                                                     self.config.hidden_size)
        embedding_size = item_encoder_hidden_states.shape[2]
        item_encoder_hidden_states = item_encoder_hidden_states.sum(dim=2) / embedding_size
        outputs = self.classifier(self.dropout(item_encoder_hidden_states)).view(batch_size, -1)

        return outputs


class DirectionClassifier(nn.Module):
    def __init__(self, config):
        super(DirectionClassifier, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.scroll_direction)
        self.config = config

    def forward(self, encoder_hidden_states):
        encoder_hidden_states = self.dropout(self.act_fn(self.linear(encoder_hidden_states)))
        encoder_hidden_states = self.classifier(encoder_hidden_states)

        return encoder_hidden_states


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class ReplyTextDecoder(nn.Module):
    def __init__(self, config: Config):
        super(ReplyTextDecoder, self).__init__()
        self.config = config
        bert_config = AutoConfig.from_pretrained(config.encoder_model_type)
        self.embedding = BertEmbeddings(bert_config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.decode = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, target, memory, target_mask, memory_padding_mask):
        """
        :param target: [batch_size, target_seq_length, hidden_size]
        :param memory: [batch_size, encoder_seq_length, hidden_size]
        :param target_mask: [encoder_seq_length, encoder_seq_length]
        :param memory_padding_mask: [batch_size, encoder_seq_length]
        :return:
        """
        target = self.embedding(target)
        out = self.transformer_decoder(tgt=target, memory=memory, tgt_mask=target_mask,
                                       memory_key_padding_mask=memory_padding_mask)
        out = self.decode(out)

        return out


class Node(object):
    def __init__(self, words, prob, avg_prob):
        self.words = words
        self.prob = prob
        self.avg_prob = avg_prob


class ResponseModel(nn.Module):
    def __init__(self, config: Config):
        super(ResponseModel, self).__init__()
        self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type)
        self.encoder_model = AutoModel.from_config(self.encoder_model_config)
        self.reply_text_decoder = ReplyTextDecoder(config)
        self.config = config

    def forward(self, input_ids, page_bbox, attention_mask, token_type_ids, reply_text):
        device = input_ids.device
        reply_seq_length = reply_text.shape[1]
        if "layoutlm" in self.config.encoder_model_type:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                       attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        decoder_input_ids = shift_tokens_right(reply_text, 0)
        tgt_mask = torch.triu(torch.ones((reply_seq_length, reply_seq_length), device=device)).transpose(0, 1)
        tgt_mask = (1.0 - tgt_mask) * -10000.0
        reply_pred = self.reply_text_decoder(decoder_input_ids.transpose(0, 1),
                                             encoder_hidden_states.transpose(0, 1),
                                             tgt_mask, attention_mask).transpose(0, 1).contiguous()

        reply_loss = F.cross_entropy(reply_pred.view(-1, self.config.vocab_size), reply_text.view(-1), ignore_index=0)

        return reply_loss

    def generate(self, input_ids, page_bbox, attention_mask, token_type_ids):
        device = input_ids.device
        if "layoutlm" in self.config.encoder_model_type:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                       attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state

        if not self.config.beam_search:

            decoder_input = torch.tensor([[self.config.cls_token_id]], dtype=torch.long)
            decoder_input = decoder_input.to(device)
            current_len = 1
            decoded_words = []
            for i in range(self.config.reply_seq_length):
                tgt_mask = torch.triu(torch.ones((current_len, current_len), device=device)).transpose(0, 1)
                tgt_mask = (1.0 - tgt_mask) * -10000.0
                outs = self.reply_text_decoder(decoder_input.transpose(0, 1), encoder_hidden_states.transpose(0, 1),
                                               tgt_mask, attention_mask).transpose(0, 1).contiguous()

                pred_text_id = torch.argmax(outs.view(-1, self.config.vocab_size)[-1, :], dim=-1)
                decoded_words.append(pred_text_id)
                if pred_text_id.item() == self.config.sep_token_id:
                    break
                decoder_input = torch.cat((decoder_input, pred_text_id.view(1, 1)), dim=1)
                current_len += 1

            return decoded_words

        else:
            decoded_candidates = []
            candidates = Queue()
            candidates.put(Node([self.config.cls_token_id], 0, 0))
            while not candidates.empty():
                current_candidates = []
                for _ in range(candidates.qsize()):
                    node = candidates.get()
                    input_words = node.words
                    prob = node.prob
                    if input_words[-1] == self.config.sep_token_id or len(input_words) >= self.config.reply_seq_length:
                        decoded_candidates.append(node)
                        continue
                    decoder_input = torch.tensor([input_words], dtype=torch.long).view(1, -1)
                    decoder_input = decoder_input.to(device)
                    current_len = len(input_words)
                    tgt_mask = torch.triu(torch.ones((current_len, current_len), device=device)).transpose(0, 1)
                    tgt_mask = (1.0 - tgt_mask) * -10000.0
                    outs = self.reply_text_decoder(decoder_input.transpose(0, 1), encoder_hidden_states.transpose(0, 1),
                                                   tgt_mask, attention_mask).transpose(0, 1).contiguous()
                    outs_prob = F.softmax(outs.view(-1, self.config.vocab_size)[-1, :], dim=-1)
                    probs, indices = outs_prob.topk(k=self.config.beam_width)
                    probs = probs.squeeze()
                    indices = indices.squeeze()
                    for i in range(self.config.beam_width):
                        new_prob = prob + math.log2(probs[i].item())
                        new_words = input_words + [indices[i].item()]
                        current_candidates.append(Node(new_words, new_prob, new_prob / len(new_words)))

                current_candidates = sorted(current_candidates, key=lambda x: x.avg_prob, reverse=True)
                length = min(len(current_candidates), self.config.beam_width)
                for i in range(length):
                    candidates.put(current_candidates[i])
            decoded_candidates = sorted(decoded_candidates, key=lambda x: x.avg_prob, reverse=True)
            decoded_words = decoded_candidates[0].words

            return decoded_words


class ActionModel(nn.Module):
    def __init__(self, config: Config):
        super(ActionModel, self).__init__()
        self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type)

        self.encoder_model = AutoModel.from_config(self.encoder_model_config)
        if config.history == "all" or config.history == "action":
            self.encoder_model.resize_token_embeddings(self.encoder_model_config.vocab_size + 7)
        self.action_type_classifier = ActionTypeClassifier(config)
        self.typing_classifier = TypingClassifier(config)
        self.item_score_classifier = ItemScoreClassifier(config)
        self.direction_classifier = DirectionClassifier(config)

        if config.weight_loss:
            self.action_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.type_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.item_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.direction_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
        self.config = config

    def forward(self, input_ids, page_bbox, attention_mask, token_type_ids, item_matrix, actions=None,
                starts=None, ends=None, target_items=None, directions=None):
        device = input_ids.device
        if "layoutlm" in self.config.encoder_model_type:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                       attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)
        encoder_outputs = encoder_hidden_states.last_hidden_state
        pooled_outputs = encoder_hidden_states.pooler_output
        action_loss = 0
        action_pred = self.action_type_classifier(pooled_outputs)
        if actions is not None:
            action_loss = F.cross_entropy(action_pred, actions)

        action_index = torch.argmax(action_pred, dim=-1).squeeze()

        starts_loss, ends_loss = 0, 0

        dialog_encoder_hidden_states = encoder_outputs[:, :self.config.dialog_seq_length, :]
        dialog_decoder_inputs = dialog_encoder_hidden_states
        starts_pred, ends_pred = self.typing_classifier(dialog_decoder_inputs)
        if starts is not None and ends is not None:
            if len(starts.size()) > 1:
                starts = starts.squeeze(-1)
            if len(ends.size()) > 1:
                ends = ends.squeeze(-1)
            starts_loss = F.cross_entropy(starts_pred, starts, ignore_index=-100)
            ends_loss = F.cross_entropy(ends_pred, ends, ignore_index=-100)

        start_index = torch.argmax(starts_pred, dim=-1).squeeze()
        end_index = torch.argmax(ends_pred, dim=-1).squeeze()

        items_loss = 0
        item_encoder_hidden_states = torch.matmul(item_matrix, encoder_outputs)
        item_decoder_inputs = item_encoder_hidden_states
        items_pred = self.item_score_classifier(item_decoder_inputs)

        if target_items is not None:
            items_loss = F.cross_entropy(items_pred, target_items, ignore_index=-100)

        item_index = torch.argmax(items_pred, dim=-1).squeeze()

        direction_loss = 0
        direction_encoder_hidden_states = pooled_outputs
        direction_pred = self.direction_classifier(direction_encoder_hidden_states)

        if directions is not None:
            direction_loss = F.cross_entropy(direction_pred, directions, ignore_index=-100)

        direction_index = torch.argmax(direction_pred, dim=-1).squeeze()

        if self.config.weight_loss:
            self.action_loss_weight = self.action_loss_weight.to(device)
            self.type_loss_weight = self.type_loss_weight.to(device)
            self.item_loss_weight = self.item_loss_weight.to(device)
            self.direction_loss_weight = self.direction_loss_weight.to(device)

            loss = torch.exp(-self.action_loss_weight) * action_loss + self.action_loss_weight + \
                   torch.exp(-self.type_loss_weight) * (starts_loss + ends_loss) / 2 + self.type_loss_weight + \
                   torch.exp(-self.item_loss_weight) * items_loss + self.item_loss_weight + \
                   torch.exp(-self.direction_loss_weight) * direction_loss + self.direction_loss_weight
        else:
            loss = action_loss + (starts_loss + ends_loss) / 2 + items_loss + direction_loss

        return loss, action_loss, (starts_loss + ends_loss) / 2, items_loss, direction_loss, action_index, start_index,\
               end_index, item_index, direction_index


class AttentionBlock(nn.Module):
    r"""
    the visual information enhanced self-attention block.
    """
    def __init__(self, config):
        super().__init__()
        self.bert_config = BertConfig()
        self.bert_config.hidden_size = config.hidden_size
        self.attention = BertAttention(self.bert_config)
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            inputs,
            visual_feature,
            attention_mask=None,
    ):
        output = torch.cat([inputs, visual_feature], dim=2)
        output = self.dense(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + inputs)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dense.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        output = self.attention(output, attention_mask=extended_attention_mask)[0]

        return output


class MultiModalActionModelWithHistory(nn.Module):
    def __init__(self, config):
        super(MultiModalActionModelWithHistory, self).__init__()
        self.config = config
        if "layoutlmv2" in config.encoder_model_type:
            self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type, revision="no_ocr")
        else:
            self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type)
        self.encoder_model = AutoModel.from_config(self.encoder_model_config)
        if config.history == "all" or config.history == "action":
            self.encoder_model.resize_token_embeddings(self.encoder_model_config.vocab_size + 7)
        self.action_type_classifier = ActionTypeClassifier(config)
        self.typing_classifier = TypingClassifier(config)
        self.item_score_classifier = ItemScoreClassifier(config)
        self.direction_classifier = DirectionClassifier(config)
        if "layoutlm-" in config.encoder_model_type or "bert" in config.encoder_model_type:
            self.init_resnet_fpn()
            self.projection = nn.Linear(2048, config.hidden_size)
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation1 = nn.Tanh()
            self.struc = nn.ModuleList([AttentionBlock(config) for _ in range(3)])

        if config.history == "all" or config.history == "screen":
            self.q_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.k_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.compose = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation2 = nn.Tanh()
            self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        if config.weight_loss:
            self.action_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.type_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.item_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
            self.direction_loss_weight = torch.autograd.Variable(torch.zeros(1), requires_grad=True)

    def init_resnet_fpn(self):
        def _validate_trainable_layers(max_value, default_value):
            # dont freeze any layers if pretrained model or backbone is not used
            trainable_backbone_layers_ = default_value
            assert 0 <= trainable_backbone_layers_ <= max_value
            return trainable_backbone_layers_

        trainable_backbone_layers = _validate_trainable_layers(5, 3)

        self.backbone = resnet_fpn_backbone('resnet50', True, trainable_layers=trainable_backbone_layers)

        min_size = 224
        max_size = 224
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        out_channels = self.backbone.out_channels
        resolution = self.box_roi_pool.output_size[0]
        representation_size = self.config.hidden_size
        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    def resnet_forward(self, image, page_box):
        hidden_states = self.resnet50.conv1(image)
        hidden_states = self.resnet50.bn1(hidden_states)
        hidden_states = self.resnet50.relu(hidden_states)
        hidden_states = self.resnet50.maxpool(hidden_states)
        hidden_states = self.resnet50.layer1(hidden_states)
        hidden_states = self.resnet50.layer2(hidden_states)
        hidden_states = self.resnet50.layer3(hidden_states)
        hidden_states = self.resnet50.layer4(hidden_states)

        device = page_box.device
        page_box = page_box * hidden_states.shape[-1] / 1000
        box_size = page_box.shape
        box_padding = torch.zeros((box_size[0], box_size[1], 1), device=device)
        page_box = torch.cat([box_padding, page_box], dim=-1).squeeze(0)
        res = roi_pool(hidden_states, page_box, output_size=(1, 1)).squeeze().unsqueeze(0)
        res = self.projection(res)

        return res

    def resnet_fpn_forward(self, image, page_box):
        images = [image.squeeze(0)]
        page_boxs = page_box * 224 / 1000
        page_boxs = [page_boxs.squeeze(0)]
        outputs, _ = self.transform(images)
        features = self.backbone(outputs.tensors)
        box_features = self.box_roi_pool(features, page_boxs, outputs.image_sizes)
        box_features = self.box_head(box_features).unsqueeze(0)

        return box_features

    def pooler1(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense1(first_token_tensor)
        pooled_output = self.activation1(pooled_output)

        return pooled_output

    def pooler2(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense2(first_token_tensor)
        pooled_output = self.activation2(pooled_output)

        return pooled_output

    def attention_compose(self, h1, h2):
        q = self.q_linear(h2)
        k = self.k_linear(h1)
        v = self.v_linear(h1)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.config.hidden_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        outputs = torch.matmul(attention_probs, v).contiguous()

        return outputs

    def forward(self, input_ids, image, page_bbox, attention_mask, token_type_ids, item_matrix, actions=None,
                starts=None, ends=None, target_items=None, directions=None):
        device = input_ids.device
        if "layoutlmv2" in self.config.encoder_model_type:
            history_length = image.shape[0]
            if history_length != 1:
                input_ids = input_ids.repeat(history_length, 1)
                page_bbox = page_bbox.repeat(history_length, 1, 1)
                token_type_ids = token_type_ids.repeat(history_length, 1)
                attention_mask = attention_mask.repeat(history_length, 1)
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox, image=image,
                                                       token_type_ids=token_type_ids, attention_mask=attention_mask)
            encoder_outputs = encoder_hidden_states.last_hidden_state
            if history_length != 1:
                image_outputs = encoder_outputs[:, :49, :]
                text_encoder_outputs = encoder_outputs[-1, 49:, :]
                text_encoder_outputs = text_encoder_outputs.unsqueeze(0)
                if history_length == 2:
                    image_embeds = self.attention_compose(image_outputs[0], image_outputs[1]).unsqueeze(0)
                elif history_length == 3:
                    image_embeds = self.attention_compose(image_outputs[0], image_outputs[1])
                    image_embeds = self.attention_compose(image_embeds, image_outputs[2]).unsqueeze(0)
                encoder_outputs = torch.cat((image_embeds, text_encoder_outputs), dim=1)
                encoder_outputs = self.compose(encoder_outputs)
                encoder_outputs = self.activation2(encoder_outputs)
                pooled_outputs = self.pooler2(encoder_outputs)
                text_encoder_outputs = encoder_outputs[:, 49:, :]
            else:
                text_encoder_outputs = encoder_outputs[:, 49:, :]
                pooled_outputs = encoder_hidden_states.pooler_output

            action_loss = 0
            action_pred = self.action_type_classifier(pooled_outputs)
            if actions is not None:
                action_loss = F.cross_entropy(action_pred, actions)

            action_index = torch.argmax(action_pred, dim=-1).squeeze()

            starts_loss, ends_loss = 0, 0

            dialog_encoder_hidden_states = text_encoder_outputs[:, :self.config.dialog_seq_length, :]
            dialog_decoder_inputs = dialog_encoder_hidden_states
            starts_pred, ends_pred = self.typing_classifier(dialog_decoder_inputs)
            if starts is not None and ends is not None:
                if len(starts.size()) > 1:
                    starts = starts.squeeze(-1)
                if len(ends.size()) > 1:
                    ends = ends.squeeze(-1)
                starts_loss = F.cross_entropy(starts_pred, starts, ignore_index=-100)
                ends_loss = F.cross_entropy(ends_pred, ends, ignore_index=-100)

            start_index = torch.argmax(starts_pred, dim=-1).squeeze()
            end_index = torch.argmax(ends_pred, dim=-1).squeeze()

            items_loss = 0
            item_encoder_hidden_states = torch.matmul(item_matrix, text_encoder_outputs)
            item_decoder_inputs = item_encoder_hidden_states
            items_pred = self.item_score_classifier(item_decoder_inputs)

            if target_items is not None:
                items_loss = F.cross_entropy(items_pred, target_items, ignore_index=-100)

            item_index = torch.argmax(items_pred, dim=-1).squeeze()

            direction_loss = 0
            direction_encoder_hidden_states = pooled_outputs
            direction_pred = self.direction_classifier(direction_encoder_hidden_states)

            if directions is not None:
                direction_loss = F.cross_entropy(direction_pred, directions, ignore_index=-100)

            direction_index = torch.argmax(direction_pred, dim=-1).squeeze()

            if self.config.weight_loss:
                self.action_loss_weight = self.action_loss_weight.to(device)
                self.type_loss_weight = self.type_loss_weight.to(device)
                self.item_loss_weight = self.item_loss_weight.to(device)
                self.direction_loss_weight = self.direction_loss_weight.to(device)

                loss = torch.exp(-self.action_loss_weight) * action_loss + self.action_loss_weight + \
                       torch.exp(-self.type_loss_weight) * (starts_loss + ends_loss) / 2 + self.type_loss_weight + \
                       torch.exp(-self.item_loss_weight) * items_loss + self.item_loss_weight + \
                       torch.exp(-self.direction_loss_weight) * direction_loss + self.direction_loss_weight
            else:
                loss = action_loss + (starts_loss + ends_loss) / 2 + items_loss + direction_loss

            return loss, action_loss, (starts_loss + ends_loss) / 2, items_loss, direction_loss, action_index, \
                   start_index, end_index, item_index, direction_index

        else:
            if "layoutlm-" in self.config.encoder_model_type:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                           attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids,
                                                           attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_outputs = encoder_hidden_states.last_hidden_state

            history_length = image.shape[0]

            if history_length != 1:
                if history_length == 2:
                    image_outputs_his = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his, image_outputs_cur)
                elif history_length == 3:
                    image_outputs_his1 = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_his2 = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[2].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his1, image_outputs_his2)
                    image_outputs = self.attention_compose(image_outputs, image_outputs_cur)
            else:
                image_outputs = self.resnet_fpn_forward(image, page_bbox)

            for i, layer in enumerate(self.struc):
                encoder_outputs = layer(encoder_outputs, image_outputs, attention_mask=attention_mask)

            pooled_outputs = self.pooler1(encoder_outputs)

            action_loss = 0
            action_pred = self.action_type_classifier(pooled_outputs)
            if actions is not None:
                action_loss = F.cross_entropy(action_pred, actions)

            action_index = torch.argmax(action_pred, dim=-1).squeeze()

            starts_loss, ends_loss = 0, 0

            dialog_encoder_hidden_states = encoder_outputs[:, :self.config.dialog_seq_length, :]
            dialog_decoder_inputs = dialog_encoder_hidden_states
            starts_pred, ends_pred = self.typing_classifier(dialog_decoder_inputs)
            if starts is not None and ends is not None:
                if len(starts.size()) > 1:
                    starts = starts.squeeze(-1)
                if len(ends.size()) > 1:
                    ends = ends.squeeze(-1)
                starts_loss = F.cross_entropy(starts_pred, starts, ignore_index=-100)
                ends_loss = F.cross_entropy(ends_pred, ends, ignore_index=-100)

            start_index = torch.argmax(starts_pred, dim=-1).squeeze()
            end_index = torch.argmax(ends_pred, dim=-1).squeeze()

            items_loss = 0
            item_encoder_hidden_states = torch.matmul(item_matrix, encoder_outputs)
            item_decoder_inputs = item_encoder_hidden_states
            items_pred = self.item_score_classifier(item_decoder_inputs)

            if target_items is not None:
                items_loss = F.cross_entropy(items_pred, target_items, ignore_index=-100)

            item_index = torch.argmax(items_pred, dim=-1).squeeze()

            direction_loss = 0
            direction_encoder_hidden_states = pooled_outputs
            direction_pred = self.direction_classifier(direction_encoder_hidden_states)

            if directions is not None:
                direction_loss = F.cross_entropy(direction_pred, directions, ignore_index=-100)

            direction_index = torch.argmax(direction_pred, dim=-1).squeeze()

            if self.config.weight_loss:
                self.action_loss_weight = self.action_loss_weight.to(device)
                self.type_loss_weight = self.type_loss_weight.to(device)
                self.item_loss_weight = self.item_loss_weight.to(device)
                self.direction_loss_weight = self.direction_loss_weight.to(device)

                loss = torch.exp(-self.action_loss_weight) * action_loss + self.action_loss_weight + \
                       torch.exp(-self.type_loss_weight) * (starts_loss + ends_loss) / 2 + self.type_loss_weight + \
                       torch.exp(-self.item_loss_weight) * items_loss + self.item_loss_weight + \
                       torch.exp(-self.direction_loss_weight) * direction_loss + self.direction_loss_weight
            else:
                loss = action_loss + (starts_loss + ends_loss) / 2 + items_loss + direction_loss

            return loss, action_loss, (
                        starts_loss + ends_loss) / 2, items_loss, direction_loss, action_index, start_index, \
                   end_index, item_index, direction_index


class MultiModalResponseModelWithHistory(nn.Module):
    def __init__(self, config: Config):
        super(MultiModalResponseModelWithHistory, self).__init__()
        if "layoutlmv2" in config.encoder_model_type:
            self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type, revision="no_ocr")
        else:
            self.encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type)

        self.encoder_model = AutoModel.from_config(self.encoder_model_config)
        if config.history == "all" or config.history == "action":
            self.encoder_model.resize_token_embeddings(self.encoder_model_config.vocab_size + 7)
        self.reply_text_decoder = ReplyTextDecoder(config)
        self.config = config

        if "layoutlm-" in config.encoder_model_type or "bert" in config.encoder_model_type:
            self.init_resnet_fpn()
            self.projection = nn.Linear(2048, config.hidden_size)
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation1 = nn.Tanh()
            self.struc = nn.ModuleList([AttentionBlock(config) for _ in range(3)])

        if config.history == "all" or config.history == "screen":
            self.q_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.k_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.compose = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation2 = nn.Tanh()
            self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

    def init_resnet_fpn(self):
        def _validate_trainable_layers(max_value, default_value):
            # dont freeze any layers if pretrained model or backbone is not used
            trainable_backbone_layers_ = default_value
            assert 0 <= trainable_backbone_layers_ <= max_value
            return trainable_backbone_layers_

        trainable_backbone_layers = _validate_trainable_layers(5, 3)

        self.backbone = resnet_fpn_backbone('resnet50', True, trainable_layers=trainable_backbone_layers)

        min_size = 224
        max_size = 224
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        out_channels = self.backbone.out_channels
        resolution = self.box_roi_pool.output_size[0]
        representation_size = self.config.hidden_size
        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    def resnet_fpn_forward(self, image, page_box):
        images = [image.squeeze(0)]
        page_boxs = page_box * 224 / 1000
        page_boxs = [page_boxs.squeeze(0)]
        outputs, _ = self.transform(images)
        features = self.backbone(outputs.tensors)
        box_features = self.box_roi_pool(features, page_boxs, outputs.image_sizes)
        box_features = self.box_head(box_features).unsqueeze(0)

        return box_features

    def pooler1(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense1(first_token_tensor)
        pooled_output = self.activation1(pooled_output)

        return pooled_output

    def pooler2(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense2(first_token_tensor)
        pooled_output = self.activation2(pooled_output)

        return pooled_output

    def attention_compose(self, h1, h2):
        q = self.q_linear(h2)
        k = self.k_linear(h1)
        v = self.v_linear(h1)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.config.hidden_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        outputs = torch.matmul(attention_probs, v).contiguous()

        return outputs

    def forward(self, input_ids, image, page_bbox, attention_mask, token_type_ids, reply_text):
        device = input_ids.device
        reply_seq_length = reply_text.shape[1]

        if "layoutlmv2" in self.config.encoder_model_type:
            history_length = image.shape[0]
            if history_length != 1:
                input_ids = input_ids.repeat(history_length, 1)
                page_bbox = page_bbox.repeat(history_length, 1, 1)
                token_type_ids = token_type_ids.repeat(history_length, 1)
                attention_mask = attention_mask.repeat(history_length, 1)
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox, image=image,
                                                       token_type_ids=token_type_ids, attention_mask=attention_mask)
            encoder_outputs = encoder_hidden_states.last_hidden_state
            encoder_outputs = encoder_outputs[:, 49:, :]
        else:
            if "layoutlm" in self.config.encoder_model_type:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                           attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                           token_type_ids=token_type_ids)
            encoder_outputs = encoder_hidden_states.last_hidden_state

            history_length = image.shape[0]

            if history_length != 1:
                if history_length == 2:
                    image_outputs_his = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his, image_outputs_cur)
                elif history_length == 3:
                    image_outputs_his1 = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_his2 = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[2].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his1, image_outputs_his2)
                    image_outputs = self.attention_compose(image_outputs, image_outputs_cur)
            else:
                image_outputs = self.resnet_fpn_forward(image, page_bbox)

            for i, layer in enumerate(self.struc):
                encoder_outputs = layer(encoder_outputs, image_outputs, attention_mask=attention_mask)

        decoder_input_ids = shift_tokens_right(reply_text, 0)
        tgt_mask = torch.triu(torch.ones((reply_seq_length, reply_seq_length), device=device)).transpose(0, 1)
        tgt_mask = (1.0 - tgt_mask) * -10000.0
        reply_pred = self.reply_text_decoder(decoder_input_ids.transpose(0, 1),
                                             encoder_outputs.transpose(0, 1),
                                             tgt_mask, attention_mask).transpose(0, 1).contiguous()

        reply_loss = F.cross_entropy(reply_pred.view(-1, self.config.vocab_size), reply_text.view(-1), ignore_index=0)

        return reply_loss

    def generate(self, input_ids, image, page_bbox, attention_mask, token_type_ids):
        device = input_ids.device
        if "layoutlmv2" in self.config.encoder_model_type:
            history_length = image.shape[0]
            if history_length != 1:
                input_ids = input_ids.repeat(history_length, 1)
                page_bbox = page_bbox.repeat(history_length, 1, 1)
                token_type_ids = token_type_ids.repeat(history_length, 1)
                attention_mask = attention_mask.repeat(history_length, 1)
            encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox, image=image,
                                                       token_type_ids=token_type_ids, attention_mask=attention_mask)
            encoder_outputs = encoder_hidden_states.last_hidden_state
            encoder_outputs = encoder_outputs[:, 49:, :]
        else:
            if "layoutlm" in self.config.encoder_model_type:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids, bbox=page_bbox,
                                                           attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                encoder_hidden_states = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                           token_type_ids=token_type_ids)
            encoder_outputs = encoder_hidden_states.last_hidden_state

            history_length = image.shape[0]

            if history_length != 1:
                if history_length == 2:
                    image_outputs_his = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his, image_outputs_cur)
                elif history_length == 3:
                    image_outputs_his1 = self.resnet_fpn_forward(image[0].unsqueeze(0), page_bbox)
                    image_outputs_his2 = self.resnet_fpn_forward(image[1].unsqueeze(0), page_bbox)
                    image_outputs_cur = self.resnet_fpn_forward(image[2].unsqueeze(0), page_bbox)
                    image_outputs = self.attention_compose(image_outputs_his1, image_outputs_his2)
                    image_outputs = self.attention_compose(image_outputs, image_outputs_cur)
            else:
                image_outputs = self.resnet_fpn_forward(image, page_bbox)

            for i, layer in enumerate(self.struc):
                encoder_outputs = layer(encoder_outputs, image_outputs, attention_mask=attention_mask)

        if not self.config.beam_search:

            decoder_input = torch.tensor([[self.config.cls_token_id]], dtype=torch.long)
            decoder_input = decoder_input.to(device)
            current_len = 1
            decoded_words = []
            for i in range(self.config.reply_seq_length):
                tgt_mask = torch.triu(torch.ones((current_len, current_len), device=device)).transpose(0, 1)
                tgt_mask = (1.0 - tgt_mask) * -10000.0
                outs = self.reply_text_decoder(decoder_input.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                               tgt_mask, attention_mask).transpose(0, 1).contiguous()

                pred_text_id = torch.argmax(outs.view(-1, self.config.vocab_size)[-1, :], dim=-1)
                decoded_words.append(pred_text_id)
                if pred_text_id.item() == self.config.sep_token_id:
                    break
                decoder_input = torch.cat((decoder_input, pred_text_id.view(1, 1)), dim=1)
                current_len += 1

            return decoded_words

        else:
            decoded_candidates = []
            candidates = Queue()
            candidates.put(Node([self.config.cls_token_id], 0, 0))
            while not candidates.empty():
                current_candidates = []
                for _ in range(candidates.qsize()):
                    node = candidates.get()
                    input_words = node.words
                    prob = node.prob
                    if input_words[-1] == self.config.sep_token_id or len(input_words) >= self.config.reply_seq_length:
                        decoded_candidates.append(node)
                        continue
                    decoder_input = torch.tensor([input_words], dtype=torch.long).view(1, -1)
                    decoder_input = decoder_input.to(device)
                    current_len = len(input_words)
                    tgt_mask = torch.triu(torch.ones((current_len, current_len), device=device)).transpose(0, 1)
                    tgt_mask = (1.0 - tgt_mask) * -10000.0
                    outs = self.reply_text_decoder(decoder_input.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                                   tgt_mask, attention_mask).transpose(0, 1).contiguous()
                    outs_prob = F.softmax(outs.view(-1, self.config.vocab_size)[-1, :], dim=-1)
                    probs, indices = outs_prob.topk(k=self.config.beam_width)
                    probs = probs.squeeze()
                    indices = indices.squeeze()
                    for i in range(self.config.beam_width):
                        new_prob = prob + math.log2(probs[i].item())
                        new_words = input_words + [indices[i].item()]
                        current_candidates.append(Node(new_words, new_prob, new_prob / len(new_words)))

                current_candidates = sorted(current_candidates, key=lambda x: x.avg_prob, reverse=True)
                length = min(len(current_candidates), self.config.beam_width)
                for i in range(length):
                    candidates.put(current_candidates[i])
            decoded_candidates = sorted(decoded_candidates, key=lambda x: x.avg_prob, reverse=True)
            decoded_words = decoded_candidates[0].words

            return decoded_words
