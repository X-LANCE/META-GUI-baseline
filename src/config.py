from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    image_size: tuple = (1280, 720)
    patch_size: int = 80
    dialog_seq_length: int = 128
    hidden_size: int = 768
    item_image_size: tuple = (224, 224)
    item_seq_length: int = 32
    item_type_size: int = 19
    item_type_pad_idx: int = 18
    item_patch_size: int = 56
    item_embedding_length: int = 1
    action_size: int = 7
    reply_seq_length: int = 64
    page_seq_length: int = 512
    scroll_direction: int = 2

    history: str = None
    multi_modal: bool = False
    encoder_model_type: str = "microsoft/layoutlm-base-uncased"
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    max_position_embeddings: int = 2048
    weight_loss: bool = False
    beam_search: bool = False
    beam_width: int = 3

    def __init__(self):
        bert_config = AutoConfig.from_pretrained(self.encoder_model_type)
        self.vocab_size = bert_config.vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.screenshot_embedding_length = int(self.image_size[0] * self.image_size[1] / self.patch_size / self.patch_size + 1)
        self.item_image_embedding_length = int(self.item_image_size[0] * self.item_image_size[1] / self.item_patch_size / self.item_patch_size)
