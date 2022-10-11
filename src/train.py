import os
import random
import torch
import torch.nn as nn
from model import ActionModel, ResponseModel, MultiModalActionModelWithHistory, MultiModalResponseModelWithHistory
import argparse
from config import Config
from tqdm import tqdm
from dataloader import action_data_loader, reply_data_loader
from nltk.translate.bleu_score import corpus_bleu
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import re
import collections
import string
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim.lr_scheduler import StepLR

# writer = SummaryWriter('./log')
torch.autograd.set_detect_anomaly(True)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def set_random_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def cache_train_dataset(args, config):
    if args.model_type == "action":
        dataloader = action_data_loader(args.batch_size, args.train_file, config)
    else:
        dataloader = reply_data_loader(args.batch_size, args.train_file, config)
    dataset = []
    tqdm_iter = tqdm(dataloader, desc="preparing")
    print("preparing dataset")
    if args.model_type == "action":
        if not config.multi_modal:
            for batch in tqdm_iter:
                input_ids, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                target_items, directions = batch
                dataset.append((input_ids, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends,
                                target_items, directions))
        else:
            for batch in tqdm_iter:
                input_ids, image, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                target_items, directions = batch
                dataset.append((input_ids, image, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts,
                                ends, target_items, directions))
    else:
        if not config.multi_modal:
            for batch in tqdm_iter:
                input_ids, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                dataset.append((input_ids, attention_masks, token_type_ids, bbox_s, reply_texts))
        else:
            for batch in tqdm_iter:
                input_ids, image, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                dataset.append((input_ids, image, attention_masks, token_type_ids, bbox_s, reply_texts))

    return dataset


def train(args, model, config):
    # dataset = cache_train_dataset(args, config)
    epoch_iter = tqdm(range(args.epoch), desc="epoch")
    num_iter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    model.to(device)
    for _ in epoch_iter:
        # random.shuffle(dataset)
        if args.model_type == "action":
            dataloader = action_data_loader(args.batch_size, args.train_file, config)
        else:
            dataloader = reply_data_loader(args.batch_size, args.train_file, config)
        batch_iter = tqdm(dataloader, desc="iteration")
        # batch_iter = tqdm(dataset, desc="iteration")
        for batch in batch_iter:
            model.train()
            start_time = time.time()
            if args.model_type == "action":
                if not config.multi_modal:
                    input_ids, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                    target_items, directions = batch
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    token_type_ids = token_type_ids.to(device)
                    bbox_s = bbox_s.to(device)
                    item_matrixes = item_matrixes.to(device)
                    actions = actions.to(device)
                    starts = starts.to(device)
                    ends = ends.to(device)
                    target_items = target_items.to(device)
                    directions = directions.to(device)
                    loss, action_loss, text_loss, item_loss, direction_loss, _, _, _, _, _ = \
                        model(input_ids, bbox_s, attention_masks, token_type_ids, item_matrixes, actions,
                              starts, ends, target_items, directions)
                else:
                    input_ids, image, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                    target_items, directions = batch
                    input_ids = input_ids.to(device)
                    image = image.to(device)
                    attention_masks = attention_masks.to(device)
                    token_type_ids = token_type_ids.to(device)
                    bbox_s = bbox_s.to(device)
                    item_matrixes = item_matrixes.to(device)
                    actions = actions.to(device)
                    starts = starts.to(device)
                    ends = ends.to(device)
                    target_items = target_items.to(device)
                    directions = directions.to(device)
                    loss, action_loss, text_loss, item_loss, direction_loss, _, _, _, _, _ = \
                        model(input_ids, image, bbox_s, attention_masks, token_type_ids, item_matrixes, actions,
                              starts, ends, target_items, directions)
                num_iter += 1
                loss = loss / args.gradient_accumulation_steps
                with torch.autograd.detect_anomaly():
                    loss.backward()
                """
                writer.add_scalar("loss", loss.item(), global_step=num_iter)
                if action_loss.item() != 0:
                    writer.add_scalar("action loss", action_loss.item(), global_step=num_iter)
                if text_loss.item() != 0:
                    writer.add_scalar("text loss", text_loss.item(), global_step=num_iter)
                if item_loss.item() != 0:
                    writer.add_scalar("item loss", item_loss.item(), global_step=num_iter)
                if direction_loss.item() != 0:
                    writer.add_scalar("direction loss", direction_loss.item(), global_step=num_iter)
                """
                end_time = time.time()
                batch_iter.set_description(desc="iter: %d, loss: %.4f, time: %.4f" % (num_iter, loss.item(),
                                                                                      end_time - start_time))
                if num_iter % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            else:
                if not config.multi_modal:
                    input_ids, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    token_type_ids = token_type_ids.to(device)
                    bbox_s = bbox_s.to(device)
                    reply_texts = reply_texts.to(device)

                    loss = model(input_ids, bbox_s, attention_masks, token_type_ids, reply_texts)
                else:
                    input_ids, image, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                    input_ids = input_ids.to(device)
                    image = image.to(device)
                    attention_masks = attention_masks.to(device)
                    token_type_ids = token_type_ids.to(device)
                    bbox_s = bbox_s.to(device)
                    reply_texts = reply_texts.to(device)

                    loss = model(input_ids, image, bbox_s, attention_masks, token_type_ids, reply_texts)
                num_iter += 1
                loss = loss / args.gradient_accumulation_steps
                with torch.autograd.detect_anomaly():
                    loss.backward()
                # writer.add_scalar("loss", loss.item(), global_step=num_iter)
                end_time = time.time()
                batch_iter.set_description(desc="iter: %d, loss: %.4f, time: %.4f" % (num_iter, loss.item(),
                                                                                      end_time - start_time))
                if num_iter % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad()

        scheduler.step()


def evaluate(args, model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.model_type == "action":
        dataloader = action_data_loader(1, args.eval_file, config, False)
    else:
        dataloader = reply_data_loader(1, args.eval_file, config, False)

    batch_iter = tqdm(dataloader, desc="iteration")
    action_target = []
    action_pred = []
    start_target = []
    start_pred = []
    end_target = []
    end_pred = []
    item_target = []
    item_pred = []
    word_target = []
    word_pred = []
    direction_target = []
    direction_pred = []
    action_completion = []
    typing_content = []
    typing_content_pred = []
    turn_info = []
    ids_info = []
    right_action_right_p = []
    right_action_wrong_p = []
    wrong_action_right_p = []
    wrong_action_wrong_p = []

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_type)

    for batch in batch_iter:
        model.eval()
        if args.model_type == "action":
            if not config.multi_modal:
                id_info, input_ids, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                target_items, directions, turn_path = batch

                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                bbox_s = bbox_s.to(device)
                item_matrixes = item_matrixes.to(device)
                actions = actions.to(device)
                starts = starts.to(device)
                ends = ends.to(device)
                target_items = target_items.to(device)
                directions = directions.to(device)

                _, _, _, _, _, action_out, starts_out, ends_out, items_out, directions_out = \
                    model(input_ids, bbox_s, attention_masks, token_type_ids, item_matrixes, actions, starts, ends,
                          target_items, directions)
            else:
                id_info, input_ids, image, attention_masks, token_type_ids, bbox_s, item_matrixes, actions, starts, ends, \
                target_items, directions, turn_path = batch

                input_ids = input_ids.to(device)
                image = image.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                bbox_s = bbox_s.to(device)
                item_matrixes = item_matrixes.to(device)
                actions = actions.to(device)
                starts = starts.to(device)
                ends = ends.to(device)
                target_items = target_items.to(device)
                directions = directions.to(device)

                _, _, _, _, _, action_out, starts_out, ends_out, items_out, directions_out = \
                    model(input_ids, image, bbox_s, attention_masks, token_type_ids, item_matrixes, actions, starts,
                          ends, target_items, directions)

            action_target += actions.tolist()
            start_target += starts.tolist()
            end_target += ends.tolist()
            typing_content.append(input_ids[0, starts.item(): ends.item() + 1, ].detach())
            item_target += target_items.tolist()
            direction_target += directions.tolist()
            turn_info.append(turn_path)
            ids_info.append(id_info)

            action_pred.append(action_out.detach().tolist())
            start_pred.append(starts_out.detach().tolist())
            end_pred.append(ends_out.detach().tolist())
            typing_content_pred.append(input_ids[0, starts_out.item(): ends_out.item() + 1, ].detach())
            item_pred.append(items_out.detach().tolist())
            direction_pred.append(directions_out.detach().tolist())
        else:
            if not config.multi_modal:
                input_ids, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                bbox_s = bbox_s.to(device)

                decoded_words = model.generate(input_ids, bbox_s, attention_masks, token_type_ids)
            else:
                input_ids, image, attention_masks, token_type_ids, bbox_s, reply_texts = batch
                input_ids = input_ids.to(device)
                image = image.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                bbox_s = bbox_s.to(device)

                decoded_words = model.generate(input_ids, image, bbox_s, attention_masks, token_type_ids)

            word_target += reply_texts
            word_pred.append(decoded_words)

    if args.model_type == "action":
        action_accuracy = np.sum(np.array(action_target) == np.array(action_pred)) / len(action_target)
        turn_completions = {}
        turn_action_completions = {}
        pred_results = {}

        def divide_into_completions(turn_path_, greed, action_type):
            if greed == 0:
                if turn_path_ in turn_action_completions.keys():
                    turn_action_completions[turn_path_].append(f"rarp_{action_type}")
                else:
                    turn_action_completions[turn_path_] = [f"rarp_{action_type}"]
            elif greed == 1:
                if turn_path_ in turn_action_completions.keys():
                    turn_action_completions[turn_path_].append(f"rawp_{action_type}")
                else:
                    turn_action_completions[turn_path_] = [f"rawp_{action_type}"]
            elif greed == 2:
                if turn_path_ in turn_action_completions.keys():
                    turn_action_completions[turn_path_].append(f"warp_{action_type}")
                else:
                    turn_action_completions[turn_path_] = [f"warp_{action_type}"]
            else:
                if turn_path_ in turn_action_completions.keys():
                    turn_action_completions[turn_path_].append(f"wawp_{action_type}")
                else:
                    turn_action_completions[turn_path_] = [f"wawp_{action_type}"]

        for i in range(len(action_target)):
            flag = False
            turn_path = turn_info[i]
            id_info = ids_info[i]
            if action_target[i] == action_pred[i]:
                if action_target[i] == 0:
                    pred_results[id_info] = f"{action_pred[i]}_{item_pred[i]}"
                    if item_target[i] == item_pred[i]:
                        action_completion.append(1)
                        flag = True
                        right_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 0, action_target[i])
                    else:
                        action_completion.append(0)
                        right_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 1, action_target[i])
                elif action_target[i] == 1:
                    pred_results[id_info] = f"{action_pred[i]}_{tokenizer.decode(typing_content_pred[i], skip_special_tokens=True)}"
                    if start_pred[i] == start_target[i] and end_pred[i] == end_target[i]:
                        action_completion.append(1)
                        flag = True
                        right_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 0, action_target[i])
                    else:
                        action_completion.append(0)
                        right_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 1, action_target[i])
                elif action_target[i] == 4:
                    pred_results[id_info] = f"{action_pred[i]}_{direction_pred[i]}"
                    if direction_target[i] == direction_pred[i]:
                        action_completion.append(1)
                        flag = True
                        right_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 0, action_target[i])
                    else:
                        action_completion.append(0)
                        right_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 1, action_target[i])
                else:
                    pred_results[id_info] = f"{action_pred[i]}"
                    action_completion.append(1)
                    flag = True
                    right_action_right_p.append(action_target[i])
                    divide_into_completions(turn_path, 0, action_target[i])
            else:
                action_completion.append(0)
                if action_target[i] == 0:
                    pred_results[id_info] = f"{action_pred[i]}_{item_pred[i]}"
                    if item_target[i] == item_pred[i]:
                        wrong_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 2, action_target[i])
                    else:
                        wrong_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 3, action_target[i])
                elif action_target[i] == 1:
                    pred_results[id_info] = f"{action_pred[i]}_{tokenizer.decode(typing_content_pred[i], skip_special_tokens=True)}"
                    if start_pred[i] == start_target[i] and end_pred[i] == end_target[i]:
                        wrong_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 2, action_target[i])
                    else:
                        wrong_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 3, action_target[i])
                elif action_target[i] == 4:
                    pred_results[id_info] = f"{action_pred[i]}_{direction_pred[i]}"
                    if direction_target[i] == direction_pred[i]:
                        wrong_action_right_p.append(action_target[i])
                        divide_into_completions(turn_path, 2, action_target[i])
                    else:
                        wrong_action_wrong_p.append(action_target[i])
                        divide_into_completions(turn_path, 3, action_target[i])
                else:
                    pred_results[id_info] = f"{action_pred[i]}"
                    wrong_action_right_p.append(action_target[i])
                    divide_into_completions(turn_path, 2, action_target[i])

            if flag:
                if turn_path in turn_completions.keys():
                    turn_completions[turn_path].append(1)
                else:
                    turn_completions[turn_path] = [1]
            else:
                if turn_path in turn_completions.keys():
                    turn_completions[turn_path].append(0)
                else:
                    turn_completions[turn_path] = [0]

        cnt = 0
        start_cnt = 0
        end_cnt = 0
        for i in range(len(start_target)):
            if start_target[i] == -100:
                continue
            cnt += 1
            if start_pred[i] == start_target[i]:
                start_cnt += 1
            if end_pred[i] == end_target[i]:
                end_cnt += 1
        start_accuracy = start_cnt / cnt
        end_accuracy = end_cnt / cnt

        typing_em = []
        typing_f1 = []

        for i in range(len(action_target)):
            if action_target[i] == 1:
                typing_content_string = tokenizer.decode(typing_content[i], skip_special_tokens=True)
                typing_content_pred_string = tokenizer.decode(typing_content_pred[i], skip_special_tokens=True)

                typing_em.append(compute_exact(typing_content_string, typing_content_pred_string))
                typing_f1.append(compute_f1(typing_content_string, typing_content_pred_string))

        cnt = 0
        item_cnt = 0
        for i in range(len(item_target)):
            if item_target[i] == -100:
                continue
            cnt += 1
            if item_target[i] == item_pred[i]:
                item_cnt += 1
        item_accuracy = item_cnt / cnt

        cnt = 0
        direction_cnt = 0
        for i in range(len(direction_target)):
            if direction_target[i] == -100:
                continue
            cnt += 1
            if direction_target[i] == direction_pred[i]:
                direction_cnt += 1
        direction_accuracy = direction_cnt / cnt

        turn_complete = []
        turn_keys = list(turn_completions.keys())
        for key in turn_keys:
            completions = turn_completions[key]
            if sum(completions) == len(completions):
                turn_complete.append(1)
            else:
                turn_complete.append(0)

        result = {
            "action_accuracy": action_accuracy,
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "typing_em": sum(typing_em) / len(typing_em),
            "typing_f1": sum(typing_f1) / len(typing_f1),
            "item_accuracy": item_accuracy,
            "direction_accuracy": direction_accuracy,
            "action_completion": sum(action_completion) / len(action_completion),
            "turn_completion": sum(turn_complete) / len(turn_complete)
        }
        analysis_res = {
            "action": {
                "right_action_right_par": right_action_right_p,
                "right_action_wrong_par": right_action_wrong_p,
                "wrong_action_right_par": wrong_action_right_p,
                "wrong_action_wrong_par": wrong_action_wrong_p,
            },
            "turn": turn_completions,
            "turn_action": turn_action_completions,
            "pred_results": pred_results,
        }

        print(result)
        with open(os.path.join(args.save_path, "action_result.json"), 'w') as file:
            json.dump(result, file, indent=1)
        with open(os.path.join(args.save_path, "analysis_result.json"), 'w') as file:
            json.dump(analysis_res, file, indent=1)

    else:
        pred_text = []
        word_belu = []
        for i in range(len(word_target)):
            if word_target[i][0] == config.cls_token_id and word_target[i][1] == config.sep_token_id:
                continue
            reply_text = tokenizer.decode(word_target[i], skip_special_tokens=True)
            reply_text_pred = tokenizer.decode(word_pred[i], skip_special_tokens=True)
            word_belu.append(corpus_bleu([[reply_text.split(" ")]], [reply_text_pred.split(" ")]))
            pred_text.append({
                "target": reply_text,
                "pred": reply_text_pred,
            })
        reply_belu_score = np.mean(word_belu)

        result = {
            "pred": pred_text,
            "reply_belu_score": reply_belu_score,
        }
        print(result)
        with open(os.path.join(args.save_path, "reply_result.json"), 'w') as file:
            json.dump(result, file, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=1, help="num train epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="saved model path, if none it will be initialized")
    parser.add_argument("-tr", "--train", action='store_true', help="train model")
    parser.add_argument("-el", "--eval", action='store_true', help="eval model")
    parser.add_argument("-te", "--test", action='store_true', help="test model")
    parser.add_argument("-s", "--save", action='store_true', help="save model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("-epi", "--eval_per_iterations", type=int, default=1000,
                        help="perform evaluation per iterations")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("-t", "--model_type", type=str, default="action", choices=["action", "reply"],
                        help="--model type to train or eval, action or reply")
    parser.add_argument("-tf", "--train_file", type=str, default="dataset/train/data.json", help="train file path")
    parser.add_argument("-ef", "--eval_file", type=str, default="dataset/dev/data.json", help="eval file path")
    parser.add_argument("-sd", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("-em", "--encoder_model", type=str, default="microsoft/layoutlm-base-uncased",
                        help="encoder model type")
    parser.add_argument("-sp", "--save_path", type=str, default="save", help="save path")
    parser.add_argument("-wl", "--weight_loss", action="store_true", help="multi task weight loss")
    parser.add_argument("-bs", "--beam_search", action="store_true", help="using beam search when decoding")
    parser.add_argument("-bm", "--beam_width", type=int, default=3, help="beam width")
    parser.add_argument("-mm", "--multi_modal", action="store_true", default=False, help="using multi-modal model")
    parser.add_argument("-hy", "--history", type=str, default=None, choices=["action", "screen", "all"],
                        help="using history as input")
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_random_seed(args)
    config = Config()
    config.encoder_model_type = args.encoder_model
    config.weight_loss = args.weight_loss
    config.history = args.history

    model_config = AutoConfig.from_pretrained(args.encoder_model)
    config.hidden_size = model_config.hidden_size
    config.beam_search = args.beam_search
    config.beam_width = args.beam_width
    config.multi_modal = args.multi_modal

    if args.model_type == "action":
        if config.multi_modal:
            model = MultiModalActionModelWithHistory(config)
        else:
            model = ActionModel(config)
    else:
        if config.multi_modal:
            model = MultiModalResponseModelWithHistory(config)
        else:
            model = ResponseModel(config)

    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    else:
        if "layoutlmv2" in config.encoder_model_type:
            layout_model = AutoModel.from_pretrained(config.encoder_model_type, revision="no_ocr")
        else:
            layout_model = AutoModel.from_pretrained(config.encoder_model_type)
        if config.history == "action" or config.history == "all":
            encoder_model_config = AutoConfig.from_pretrained(config.encoder_model_type)
            layout_model.resize_token_embeddings(encoder_model_config.vocab_size + 7)
        model.encoder_model.load_state_dict(layout_model.state_dict())

    print(args)
    print(config)

    if args.train:
        train(args, model, config)
    if args.save:
        if args.model_type == "action":
            torch.save(model.state_dict(), os.path.join(args.save_path, "action_model.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(args.save_path, "reply_model.pt"))

    if args.eval:
        evaluate(args, model, config)


if __name__ == "__main__":
    main()
