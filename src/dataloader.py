import json
import random
from PIL import Image
from DataProcessing import ActionToIdx
import torch
from config import Config
from transformers import AutoTokenizer, AutoProcessor
from transformers import ViTFeatureExtractor


fake_box = [[0, 0, 0, 0]]


def extract_image_feature(config, image, processor):
    fake_text = ["hello"]
    if "layoutlmv2" in config.encoder_model_type:
        image_feature = processor(image, fake_text, boxes=fake_box, return_tensors="pt")["image"]
    elif "clip" in config.encoder_model_type:
        image_feature = processor(text=fake_text, images=image, return_tensors="pt")["pixel_values"]
    else:
        image_feature = torch.tensor(processor(image)["pixel_values"][0]).unsqueeze(0)

    return image_feature


def get_start_end(tokenizer, utterance_split, text_split):
    """
    extract the start position and end position, which is the parameter of "input" action,  from the dialogue histories
    This function requires that the dialog histories are in reverse order, and there exists a "sep_token" between two
    different utterances.
    """

    start = -1
    end = -1
    for i in range(0, len(utterance_split) - len(text_split) + 3):
        turn_flag = True
        for j in range(0, len(text_split)-2):
            if utterance_split[i + j] == text_split[j+1]:
                continue
            else:
                turn_flag = False
                break
        if turn_flag:
            start = i
            end = i + len(text_split) - 3
            break
    if start == -1:
        print(utterance_split)
        print(text_split)
        raise ValueError("Can not extract the start postion and end postion!")
    return start, end


def reply_data_loader(batch_size, data_path, config: Config, train=True):
    with open(data_path, 'r') as reader:
        data = json.load(reader)
    if batch_size != 1:
        raise ValueError("only support batch_size=1!")

    if train:
        random.shuffle(data)

    if config.multi_modal:
        if "layoutlmv2" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type, revision="no_ocr")
        elif "clip" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type)
        else:
            processor = ViTFeatureExtractor()

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_type)
    if config.history == "all" or config.history == "action":
        tokenizer.add_tokens(list(ActionToIdx.keys()))

    input_ids = []
    attention_masks = []
    token_type_ids = []
    reply_texts = []
    bbox_s = []

    for d in data:
        if d["action"] != "response":
            continue
        input_id = []
        token_type_id = []
        attention_mask = []
        bbox = []
        item_input_id = []
        item_attention_mask = []

        dialogue_text_list = d["dialog"].copy()
        dialogue_text_list.reverse()

        history_flag = False
        if config.history == "all" or config.history == "action":
            action_history = d["action_history"]

            if len(action_history) != 0:
                action_lists = []
                for action_history_ in action_history:
                    action_lists.append(f"[{action_history_['action_info'].split('|')[1]}]")
                action_lists = action_lists[-3:]
                action_infos = " ".join(action_lists)

                dialogue_text_list = [action_infos] + dialogue_text_list

                dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
                if "layoutlmv2" in config.encoder_model_type:
                    dialog_token = tokenizer([dialogue_text], boxes=fake_box,
                                             max_length=config.dialog_seq_length, truncation=True)
                else:
                    dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length, truncation=True)

                input_id += dialog_token["input_ids"]
                token_type_id += dialog_token["token_type_ids"]
                attention_mask += dialog_token["attention_mask"]
                max_page_length = 512 - len(dialog_token["input_ids"])
                bbox += [[0, 0, 1000, 1000]]
                bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)
                history_flag = True

        if not history_flag:
            dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
            if "layoutlmv2" in config.encoder_model_type:
                dialog_token = tokenizer([dialogue_text], boxes=fake_box,
                                         max_length=config.dialog_seq_length, truncation=True)
            else:
                dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length, truncation=True)

            input_id += dialog_token["input_ids"]
            token_type_id += dialog_token["token_type_ids"]
            attention_mask += dialog_token["attention_mask"]
            max_page_length = 512 - len(dialog_token["input_ids"])
            bbox += [[0, 0, 1000, 1000]]
            bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)

        items = d["items"]
        text = d["response"]

        for item in items:
            item_text = item["text"]
            if "layoutlmv2" in config.encoder_model_type:
                item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
            else:
                item_text_token = tokenizer(item_text, add_special_tokens=False)
            if len(item_text_token["input_ids"]) == 0:
                item_text = item["type"].split(".")[-1]
                if "layoutlmv2" in config.encoder_model_type:
                    item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
                else:
                    item_text_token = tokenizer(item_text, add_special_tokens=False)
            item_input_id += item_text_token["input_ids"]
            item_attention_mask += item_text_token["attention_mask"]
            border = item["border"]
            resize_border = [int(border[0]*1000/1440), int(border[1]*1000/2560),
                             int(border[2]*1000/1440), int(border[3]*1000/2560)]
            bbox += [resize_border] * len(item_text_token["input_ids"])
            if len(item_input_id) > max_page_length:
                break
        if len(item_input_id) > max_page_length:
            item_input_id = item_input_id[:max_page_length]
            item_attention_mask = item_attention_mask[:max_page_length]
            bbox = bbox[:512]
        input_id += item_input_id
        attention_mask += item_attention_mask
        token_type_id += [1] * len(item_input_id)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        bbox_s.append(bbox)

        if config.multi_modal:
            image_path = d["screenshot_history"][-1]
            image = Image.open(image_path).convert("RGB")
            image_feature = extract_image_feature(config, image, processor)

            if config.history == "all" or config.history == "screen":
                image_histories = d["screenshot_history"]
                if len(image_histories) == 2:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                elif len(image_histories) >= 3:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                    image_history = image_histories[-3]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)

        if "layoutlmv2" in config.encoder_model_type:
            reply_text_tokenized = tokenizer([text], boxes=fake_box, padding=True)["input_ids"]
        else:
            reply_text_tokenized = tokenizer(text, padding=True)["input_ids"]

        reply_texts.append(reply_text_tokenized)

        if config.multi_modal:
            yield torch.tensor(input_ids), \
                  image_feature, \
                  torch.tensor(attention_masks), \
                  torch.tensor(token_type_ids), \
                  torch.tensor(bbox_s), \
                  torch.tensor(reply_texts),
        else:
            yield torch.tensor(input_ids), \
                  torch.tensor(attention_masks), \
                  torch.tensor(token_type_ids), \
                  torch.tensor(bbox_s), \
                  torch.tensor(reply_texts),

        input_ids = []
        attention_masks = []
        token_type_ids = []
        bbox_s = []
        reply_texts = []


def action_data_loader(batch_size, data_path, config: Config, train=True):
    with open(data_path, 'r') as reader:
        data = json.load(reader)
    if batch_size != 1:
        raise ValueError("only support batch_size=1!")

    if train:
        random.shuffle(data)

    if config.multi_modal:
        if "layoutlmv2" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type, revision="no_ocr")
        elif "clip" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type)
        else:
            processor = ViTFeatureExtractor()

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_type)
    if config.history == "all" or config.history == "action":
        tokenizer.add_tokens(list(ActionToIdx.keys()))

    input_ids = []
    attention_masks = []
    token_type_ids = []
    item_matrixes = []

    target_items = []
    actions = []
    starts = []
    ends = []
    directions = []

    bbox_s = []
    mat_length = 0

    for d in data:
        input_id = []
        token_type_id = []
        attention_mask = []
        item_matrix = []
        bbox = []
        item_input_id = []
        item_attention_mask = []

        dialogue_text_list = d["dialog"].copy()
        dialogue_text_list.reverse()

        history_flag = False
        if config.history == "all" or config.history == "action":
            action_history = d["action_history"]
            if len(action_history) != 0:
                action_lists = []
                for action_history_ in action_history:
                    action_lists.append(f"[{action_history_['action_info'].split('|')[1]}]")
                action_lists = action_lists[-3:]
                action_infos = " ".join(action_lists)

                dialogue_text_list = [action_infos] + dialogue_text_list

                if "clip" in config.encoder_model_type:
                    dialogue_text = " ".join(dialogue_text_list)
                else:
                    dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
                if "layoutlmv2" in config.encoder_model_type:
                    dialog_token = tokenizer([dialogue_text], boxes=fake_box, max_length=config.dialog_seq_length,
                                             padding="max_length", truncation=True)
                else:
                    dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length,
                                             padding="max_length", truncation=True)

                mat_length += len(dialog_token["input_ids"])
                input_id += dialog_token["input_ids"]
                if "clip" not in config.encoder_model_type:
                    token_type_id += dialog_token["token_type_ids"]
                attention_mask += dialog_token["attention_mask"]
                max_page_length = 512 - len(dialog_token["input_ids"])
                bbox += [[0, 0, 1000, 1000]]
                bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)
                history_flag = True

        if not history_flag:
            if "clip" in config.encoder_model_type:
                dialogue_text = " ".join(dialogue_text_list)
            else:
                dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
            if "layoutlmv2" in config.encoder_model_type:
                dialog_token = tokenizer([dialogue_text], boxes=fake_box, max_length=config.dialog_seq_length,
                                         padding="max_length", truncation=True)
            else:
                dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length,
                                         padding="max_length", truncation=True)

            mat_length += len(dialog_token["input_ids"])
            input_id += dialog_token["input_ids"]
            if "clip" not in config.encoder_model_type:
                token_type_id += dialog_token["token_type_ids"]
            attention_mask += dialog_token["attention_mask"]
            # max_page_length -= len(dialog_token["input_ids"]) - 1
            max_page_length = 512 - len(dialog_token["input_ids"])
            bbox += [[0, 0, 1000, 1000]]
            bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)

        actions.append(ActionToIdx[d["action"]])

        if d["input"] is not None:
            inputs_text = d["input"]
            if "layoutlmv2" in config.encoder_model_type:
                inputs_text_token = tokenizer([inputs_text], boxes=fake_box)
            else:
                inputs_text_token = tokenizer(inputs_text)
            start, end = get_start_end(tokenizer, dialog_token["input_ids"],
                                       inputs_text_token["input_ids"])
            starts.append(start)
            ends.append(end)
        else:
            starts.append(-100)
            ends.append(-100)

        if d["scroll"] is not None:
            directions.append(d["scroll"])
        else:
            directions.append(-100)

        target_item = d["target"]
        if target_item is None:
            target_items.append(-100)
        else:
            target_items.append(d["target"])

        items = d["items"]
        for item in items:

            item_text = item["text"]
            if "layoutlmv2" in config.encoder_model_type:
                item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
            else:
                item_text_token = tokenizer(item_text, add_special_tokens=False)
            if len(item_text_token["input_ids"]) == 0:
                item_text = item["type"].split(".")[-1]
                if "layoutlmv2" in config.encoder_model_type:
                    item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
                else:
                    item_text_token = tokenizer(item_text, add_special_tokens=False)
            item_input_id += item_text_token["input_ids"]
            item_attention_mask += item_text_token["attention_mask"]
            border = item["border"]
            resize_border = [int(border[0] * 1000 / 1440), int(border[1] * 1000 / 2560),
                             int(border[2] * 1000 / 1440), int(border[3] * 1000 / 2560)]
            item_token_length = len(item_text_token["input_ids"])
            bbox += [resize_border] * item_token_length
            item_matrix.append([0.0]*mat_length + [1/item_token_length]*item_token_length)
            mat_length += item_token_length
            if len(item_input_id) > max_page_length:
                break
        if len(item_input_id) > max_page_length:
            item_input_id = item_input_id[:max_page_length]
            item_attention_mask = item_attention_mask[:max_page_length]
            bbox = bbox[:512]
            mat_length = 512
        input_id += item_input_id
        attention_mask += item_attention_mask
        token_type_id += [1] * len(item_input_id)

        for i, item_mat in enumerate(item_matrix):
            if len(item_mat) < mat_length:
                item_mat += [0] * (mat_length - len(item_mat))
            else:
                item_matrix[i] = item_mat[:mat_length]
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        if "clip" not in config.encoder_model_type:
            token_type_ids.append(token_type_id)
        bbox_s.append(bbox)
        item_matrixes.append(item_matrix)
        mat_length = 0

        if config.multi_modal:
            image_path = d["screenshot_history"][-1]
            image = Image.open(image_path).convert("RGB")
            image_feature = extract_image_feature(config, image, processor)

            if config.history == "all" or config.history == "screen":
                image_histories = d["screenshot_history"]
                if len(image_histories) == 2:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                elif len(image_histories) >= 3:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                    image_history = image_histories[-3]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)

        if train:
            if config.multi_modal:
                yield torch.tensor(input_ids), \
                      image_feature, \
                      torch.tensor(attention_masks), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions)
            else:
                yield torch.tensor(input_ids), \
                      torch.tensor(attention_masks), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions)
        else:
            if config.multi_modal:
                yield d["screenshot_history"][-1], \
                      torch.tensor(input_ids), \
                      image_feature, \
                      torch.tensor(attention_masks), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions), \
                      d["turn"]
            else:
                yield d["screenshot_history"][-1], \
                      torch.tensor(input_ids), \
                      torch.tensor(attention_masks), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions), \
                      d["turn"]

        input_ids = []
        attention_masks = []
        token_type_ids = []
        bbox_s = []
        item_matrixes = []
        actions = []
        starts = []
        ends = []
        target_items = []
        directions = []
