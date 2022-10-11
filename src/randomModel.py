import json
import random
from transformers import BertTokenizer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import re
import string
import collections

with open("../dataset/train/data.json", 'r') as reader:
    train_data = json.load(reader)


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


def get_start_end(token, dialog_token_input_ids, inputs_token_input_ids):
    """
    extract the start position and end position, which is the parameter of "input" action,  from the dialogue histories
    This function requires that the dialog histories are in reverse order, and there exists a "sep_token" between two
    different utterances.
    """
    max_index = dialog_token_input_ids.index(token.sep_token_id) - 1
    search_index = []
    for i in range(1, len(inputs_token_input_ids)-1):
        try:
            index = dialog_token_input_ids.index(inputs_token_input_ids[i])
            if index > max_index:
                index = max_index
            search_index.append(index)
        except ValueError:
            continue
    if len(search_index) == 0:
        start_idx = 1
        end_idx = max_index
    else:
        start_idx = min(search_index)
        end_idx = max(search_index)

    return start_idx, end_idx


action_dict = {}
start_dict = {}
end_dict = {}
item_dict = {}
direction_dict = {}
reply_dict = {}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


for d in train_data:
    action = d["action"]
    items = d["items"]
    direction = d["scroll"]
    if action in action_dict.keys():
        action_dict[action] += 1
    else:
        action_dict[action] = 1

    dialogue_text_list = d["dialog"].copy()
    dialogue_text_list.reverse()
    dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
    dialog_token = tokenizer(dialogue_text, max_length=128, padding="max_length", truncation=True)

    if d["input"] is not None:
        inputs_text = d["input"]
        start, end = get_start_end(tokenizer, dialog_token["input_ids"],
                                   tokenizer(inputs_text)["input_ids"])
        if start in start_dict.keys():
            start_dict[start] += 1
        else:
            start_dict[start] = 1
        if end in end_dict.keys():
            end_dict[end] += 1
        else:
            end_dict[end] = 1
    for item in items:
        item_rep = f"{item['text']} {item['type']} {item['border']}"
        if item_rep in item_dict.keys():
            item_dict[item_rep] += 1
        else:
            item_dict[item_rep] = 1
    if direction is not None:
        if direction in direction_dict.keys():
            direction_dict[direction] += 1
        else:
            direction_dict[direction] = 1
    reply = d["response"]
    if reply is not None:
        if reply in reply_dict.keys():
            reply_dict[reply] += 1
        else:
            reply_dict[reply] = 1

with open("../dataset/dev/data.json", 'r') as reader:
    dev_data = json.load(reader)

all_actions = list(action_dict.keys())
all_action_weights = list(action_dict.values())
all_starts = list(start_dict.keys())
all_start_weights = list(start_dict.values())
all_ends = list(end_dict.keys())
all_end_weights = list(end_dict.values())
all_items = list(item_dict.keys())
all_item_weights = list(item_dict.values())
all_directions = list(direction_dict.keys())
all_direction_weights = list(direction_dict.values())
all_replies = list(reply_dict.keys())
all_reply_weights = list(reply_dict.values())

most_action = all_actions[np.argmax(all_action_weights)]
most_start = all_starts[np.argmax(all_start_weights)]
most_end = all_ends[np.argmax(all_end_weights)]
most_item = all_items[np.argmax(all_item_weights)]
most_direction = all_directions[np.argmax(all_direction_weights)]
most_reply = all_replies[np.argmax(all_reply_weights)]

random_result = []
frequency_result = []
most_result = []

for num in range(0, 10):
    random_actions = []
    random_starts = []
    random_ends = []
    random_typing_em = []
    random_typing_f1 = []
    random_items = []
    random_directions = []
    random_replies = []
    random_action_completions = []
    random_task_completions = {}

    frequency_actions = []
    frequency_starts = []
    frequency_ends = []
    frequency_typing_em = []
    frequency_typing_f1 = []
    frequency_items = []
    frequency_directions = []
    frequency_replies = []
    frequency_action_completions = []
    frequency_task_completions = {}

    most_actions = []
    most_starts = []
    most_ends = []
    most_typing_em = []
    most_typing_f1 = []
    most_items = []
    most_directions = []
    most_replies = []
    most_action_completions = []
    most_task_completions = {}

    for d in dev_data:
        turn = d["turn"]
        random_action = random.sample(all_actions, k=1)[0]
        frequency_action = random.choices(all_actions, weights=all_action_weights, k=1)[0]
        if random_action == d["action"]:
            random_actions.append(1)
        else:
            random_actions.append(0)
        if frequency_action == d["action"]:
            frequency_actions.append(1)
        else:
            frequency_actions.append(0)
        if most_action == d["action"]:
            most_actions.append(1)
        else:
            most_actions.append(0)

        if d["action"] == "input":
            dialogue_text_list = d["dialog"].copy()
            dialogue_text_list.reverse()
            dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
            dialog_token = tokenizer(dialogue_text, max_length=128, padding="max_length", truncation=True)
            inputs_text = d["input"]
            true_start, true_end = get_start_end(tokenizer, dialog_token["input_ids"],
                                                 tokenizer(inputs_text)["input_ids"])

            start, end = random.sample(range(len(dialog_token)), k=2)
            if start > end:
                start, end = end, start
            if start == true_start:
                random_starts.append(1)
            else:
                random_starts.append(0)
            if end == true_end:
                random_ends.append(1)
            else:
                random_ends.append(0)
            if random_action == d["action"] and start == true_start and end == true_end:
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(1)
                else:
                    random_task_completions[turn] = [1]
                random_action_completions.append(1)
            else:
                random_action_completions.append(0)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(0)
                else:
                    random_task_completions[turn] = [0]

            typing_content = dialog_token["input_ids"][true_start:true_end+1]
            typing_content_pred = dialog_token["input_ids"][start:end+1]
            typing_content_string = tokenizer.decode(typing_content, skip_special_tokens=True)
            typing_content_pred_string = tokenizer.decode(typing_content_pred, skip_special_tokens=True)

            random_typing_em.append(compute_exact(typing_content_string, typing_content_pred_string))
            random_typing_f1.append(compute_f1(typing_content_string, typing_content_pred_string))

            position_candid = [i for i in range(len(dialog_token))]
            starts_weight = []
            ends_weight = []
            for i in position_candid:
                if i in start_dict.keys():
                    starts_weight.append(start_dict[i])
                else:
                    starts_weight.append(0)
                if i in end_dict.keys():
                    ends_weight.append(end_dict[i])
                else:
                    ends_weight.append(0)
            start_pred = random.choices(position_candid, weights=starts_weight, k=1)[0]
            end_pred = random.choices(position_candid, weights=ends_weight, k=1)[0]
            if start_pred == true_start:
                frequency_starts.append(1)
            else:
                frequency_starts.append(0)
            if end_pred == true_end:
                frequency_ends.append(1)
            else:
                frequency_ends.append(0)
            typing_content_pred = dialog_token["input_ids"][start_pred:end_pred + 1]
            typing_content_pred_string = tokenizer.decode(typing_content_pred, skip_special_tokens=True)

            frequency_typing_em.append(compute_exact(typing_content_string, typing_content_pred_string))
            frequency_typing_f1.append(compute_f1(typing_content_string, typing_content_pred_string))

            if frequency_action == d["action"] and start_pred == true_start and end_pred == true_end:
                frequency_action_completions.append(1)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(1)
                else:
                    frequency_task_completions[turn] = [1]
            else:
                frequency_action_completions.append(0)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(0)
                else:
                    frequency_task_completions[turn] = [0]

            if most_start == true_start:
                most_starts.append(1)
            else:
                most_starts.append(0)
            if most_end == true_end:
                most_ends.append(1)
            else:
                most_ends.append(0)

            if most_action == d["action"] and most_start == true_start and most_end == true_end:
                most_action_completions.append(1)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(1)
                else:
                    most_task_completions[turn] = [1]
            else:
                most_action_completions.append(0)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(0)
                else:
                    most_task_completions[turn] = [0]

            typing_content_pred = dialog_token["input_ids"][most_start:most_end + 1]
            typing_content_pred_string = tokenizer.decode(typing_content_pred, skip_special_tokens=True)

            most_typing_em.append(compute_exact(typing_content_string, typing_content_pred_string))
            most_typing_f1.append(compute_f1(typing_content_string, typing_content_pred_string))

        elif d["action"] == "click":
            items = d["items"]
            items_candid = range(len(items))
            items_pred = random.sample(items_candid, k=1)[0]
            if items_pred == d["target"]:
                random_items.append(1)
            else:
                random_items.append(0)

            if random_action == d["action"] and items_pred == d["target"]:
                random_action_completions.append(1)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(1)
                else:
                    random_task_completions[turn] = [1]

            else:
                random_action_completions.append(0)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(0)
                else:
                    random_task_completions[turn] = [0]

            items_weight = []
            for item in items:
                item_rep = f"{item['text']} {item['type']} {item['border']}"
                if item_rep in item_dict.keys():
                    items_weight.append(item_dict[item_rep])
                else:
                    items_weight.append(0)
            item_pred = random.choices(items_candid, weights=items_weight, k=1)[0]
            if item_pred == d["target"]:
                frequency_items.append(1)
            else:
                frequency_items.append(0)

            if frequency_action == d["action"] and item_pred == d["target"]:
                frequency_action_completions.append(1)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(1)
                else:
                    frequency_task_completions[turn] = [1]
            else:
                frequency_action_completions.append(0)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(0)
                else:
                    frequency_task_completions[turn] = [0]

            flag = False
            for item in items:
                item_rep = f"{item['text']} {item['type']} {item['border']}"
                if item_rep == most_item:
                    flag = True
                    break
            if flag:
                most_items.append(1)
            else:
                most_items.append(0)
            if most_action == d["action"] and flag:
                most_action_completions.append(1)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(1)
                else:
                    most_task_completions[turn] = [1]
            else:
                most_action_completions.append(0)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(0)
                else:
                    most_task_completions[turn] = [0]
        elif d["action"] == "swipe":
            direction_label = d["scroll"]
            direction_pred = random.sample([0, 1], k=1)[0]
            if direction_pred == direction_label:
                random_directions.append(1)
            else:
                random_directions.append(0)
            if random_action == d["action"] and direction_pred == direction_label:
                random_action_completions.append(1)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(1)
                else:
                    random_task_completions[turn] = [1]

            else:
                random_action_completions.append(0)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(0)
                else:
                    random_task_completions[turn] = [0]

            direction_weights = []
            for i in [0, 1]:
                if i in direction_dict.keys():
                    direction_weights.append(direction_dict[i])
                else:
                    direction_weights.append(0)
            direction_pred = random.choices([0, 1], weights=direction_weights, k=1)[0]
            if direction_pred == direction_label:
                frequency_directions.append(1)
            else:
                frequency_directions.append(0)
            if frequency_action == d["action"] and direction_pred == direction_label:
                frequency_action_completions.append(1)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(1)
                else:
                    frequency_task_completions[turn] = [1]
            else:
                frequency_action_completions.append(0)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(0)
                else:
                    frequency_task_completions[turn] = [0]

            if most_direction == direction_label:
                most_directions.append(1)
            else:
                most_directions.append(0)
            if most_action == d["action"] and most_direction == direction_label:
                most_action_completions.append(1)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(1)
                else:
                    most_task_completions[turn] = [1]
            else:
                most_action_completions.append(0)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(0)
                else:
                    most_task_completions[turn] = [0]

        elif d["action"] == "response":
            random_reply = random.sample(all_replies, k=1)[0]
            random_replies.append(corpus_bleu([[d["response"].split()]], [random_reply.split()]))

            frequency_reply = random.choices(all_replies, weights=all_reply_weights, k=1)[0]
            frequency_replies.append(corpus_bleu([[d["response"].split()]], [frequency_reply.split()],
                                                 weights=(1, 0, 0, 0)))

            most_replies.append(corpus_bleu([[d["response"].split()]], [most_reply.split()]))

            if random_action == d["action"]:
                random_action_completions.append(1)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(1)
                else:
                    random_task_completions[turn] = [1]
            else:
                random_action_completions.append(0)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(0)
                else:
                    random_task_completions[turn] = [0]

            if frequency_action == d["action"]:
                frequency_action_completions.append(1)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(1)
                else:
                    frequency_task_completions[turn] = [1]
            else:
                frequency_action_completions.append(0)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(0)
                else:
                    frequency_task_completions[turn] = [0]
            if most_action == d["action"]:
                most_action_completions.append(1)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(1)
                else:
                    most_task_completions[turn] = [1]
            else:
                most_action_completions.append(0)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(0)
                else:
                    most_task_completions[turn] = [0]
        else:
            if random_action == d["action"]:
                random_action_completions.append(1)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(1)
                else:
                    random_task_completions[turn] = [1]
            else:
                random_action_completions.append(0)
                if turn in random_task_completions.keys():
                    random_task_completions[turn].append(0)
                else:
                    random_task_completions[turn] = [0]

            if frequency_action == d["action"]:
                frequency_action_completions.append(1)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(1)
                else:
                    frequency_task_completions[turn] = [1]
            else:
                frequency_action_completions.append(0)
                if turn in frequency_task_completions.keys():
                    frequency_task_completions[turn].append(0)
                else:
                    frequency_task_completions[turn] = [0]
            if most_action == d["action"]:
                most_action_completions.append(1)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(1)
                else:
                    most_task_completions[turn] = [1]
            else:
                most_action_completions.append(0)
                if turn in most_task_completions.keys():
                    most_task_completions[turn].append(0)
                else:
                    most_task_completions[turn] = [0]

    random_turn_completions = []
    frequency_turn_completions = []
    most_turn_completions = []
    keys = list(random_task_completions.keys())
    for key in keys:
        random_turn_completion = random_task_completions[key]
        if sum(random_turn_completion) == len(random_turn_completion):
            random_turn_completions.append(1)
        else:
            random_turn_completions.append(0)

        frequency_turn_completion = frequency_task_completions[key]
        if sum(frequency_turn_completion) == len(frequency_turn_completion):
            frequency_turn_completions.append(1)
        else:
            frequency_turn_completions.append(0)

        most_turn_completion = most_task_completions[key]
        if sum(most_turn_completion) == len(most_turn_completion):
            most_turn_completions.append(1)
        else:
            most_turn_completions.append(0)

    random_result.append([sum(random_actions) / len(random_actions),
                          sum(random_starts) / len(random_starts),
                          sum(random_ends) / len(random_ends),
                          sum(random_typing_em) / len(random_typing_em),
                          sum(random_typing_f1) / len(random_typing_f1),
                          sum(random_items) / len(random_items),
                          sum(random_directions) / len(random_directions),
                          sum(random_replies) / len(random_replies),
                          sum(random_action_completions) / len(random_action_completions),
                          sum(random_turn_completions) / len(random_turn_completions)])

    frequency_result.append([sum(frequency_actions) / len(frequency_actions),
                             sum(frequency_starts) / len(frequency_starts),
                             sum(frequency_ends) / len(frequency_ends),
                             sum(frequency_typing_em) / len(frequency_typing_em),
                             sum(frequency_typing_f1) / len(frequency_typing_f1),
                             sum(frequency_items) / len(frequency_items),
                             sum(frequency_directions) / len(frequency_directions),
                             sum(frequency_replies) / len(frequency_replies),
                             sum(frequency_action_completions) / len(frequency_action_completions),
                             sum(frequency_turn_completions) / len(frequency_turn_completions)])

    most_result.append([sum(most_actions) / len(most_actions),
                        sum(most_starts) / len(most_starts),
                        sum(most_ends) / len(most_ends),
                        sum(most_typing_em) / len(most_typing_em),
                        sum(most_typing_f1) / len(most_typing_f1),
                        sum(most_items) / len(most_items),
                        sum(most_directions) / len(most_directions),
                        sum(most_replies) / len(most_replies),
                        sum(most_action_completions) / len(most_action_completions),
                        sum(most_turn_completions) / len(most_turn_completions)])


random_res = np.mean(np.array(random_result), axis=0)
frequency_res = np.mean(np.array(frequency_result), axis=0)
most_res = np.mean(np.array(most_result), axis=0)

print(random_res)
print(frequency_res)
print(most_res)
