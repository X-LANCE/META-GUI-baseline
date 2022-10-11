import xml.dom.minidom as xdm
from PIL import ImageDraw
import os
import json
from tqdm import tqdm
import re


class Processor:
    def __init__(self):
        self.border = []
        self.border_item = []

    def checkClickable(self, tree):
        if tree.attributes:
            clickable = tree.attributes.getNamedItem('clickable')
            if clickable and clickable.nodeValue == 'true':
                return True
            if tree.parentNode is not None:
                return self.checkClickable(tree.parentNode)
        else:
            return False

    def dfs(self, tree):
        if len(tree.childNodes) != 0:
            for child in tree.childNodes:
                self.dfs(child)
        else:

            if tree.attributes:
                attributes = tree.attributes.getNamedItem('bounds')
                enable = tree.attributes.getNamedItem('enabled')
                if attributes and enable and enable.nodeValue == 'true' and self.checkClickable(tree):
                    bounds = attributes.nodeValue
                    left_top, right_bottom, _ = bounds.split(']')
                    left_top = left_top[1:]
                    right_bottom = right_bottom[1:]
                    left, top = map(int, left_top.split(','))
                    right, bottom = map(
                        int, right_bottom.split(','))
                    if right - left <= 5 or bottom - top <= 5:
                        pass
                    else:
                        self.border.append((left, top, right, bottom))
                        self.border_item.append(tree)

    def getAllNode(self, tree):
        if tree.attributes:
            attributes = tree.attributes.getNamedItem('bounds')
            if attributes:
                bounds = attributes.nodeValue
                left_top, right_bottom, _ = bounds.split(']')
                left_top = left_top[1:]
                right_bottom = right_bottom[1:]
                left, top = map(int, left_top.split(','))
                right, bottom = map(
                    int, right_bottom.split(','))
                self.border.append((left, top, right, bottom))
                self.border_item.append(tree)

        if len(tree.childNodes) != 0:
            for child in tree.childNodes:
                self.getAllNode(child)

    @staticmethod
    def draw(img, borders):
        for bd in borders:
            left, top, right, bottom = bd
            shape = [(left, top), (right, bottom)]
            img1 = ImageDraw.Draw(img)
            img1.rectangle(shape, outline="red", width=3)
        return img

    @staticmethod
    def drawWithClick(img, borders, x, y):
        for bd in borders:
            left, top, right, bottom = bd
            shape = [(left, top), (right, bottom)]
            img1 = ImageDraw.Draw(img)
            img1.rectangle(shape, outline="red", width=3)
        img2 = ImageDraw.Draw(img)
        img2.ellipse(((x - 20, y - 20), (x + 20, y + 20)), outline=(0, 0, 0), width=5)
        return img

    @staticmethod
    def checkFatherContain(border1, border2):
        if border1[0] <= border2[0] and border1[1] <= border2[1] and border1[2] >= border2[2] \
                and border1[3] >= border2[3]:
            return True
        return False

    @staticmethod
    def checkIfContain(border1, border2):
        x_min = max(border1[0], border2[0])
        x_max = min(border1[2], border2[2])
        y_min = max(border1[1], border2[1])
        y_max = min(border1[3], border2[3])
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            return False
        area = w * h
        if area != 0:
            return True
        return False

    def postProcessing(self, pre_border):
        border_flag = [0] * len(pre_border)
        for i in range(len(pre_border)):
            if border_flag[i] == 1:
                continue
            for j in range(i + 1, len(pre_border)):
                if border_flag[j] == 1:
                    continue
                if self.checkFatherContain(pre_border[i], pre_border[j]):
                    border_flag[i] = 1
                elif self.checkFatherContain(pre_border[j], pre_border[i]):
                    border_flag[j] = 1

        return border_flag

    @staticmethod
    def extractImage(raw_image, borders):
        images = []
        for b in borders:
            images.append(raw_image.crop(b))

        return images

    @staticmethod
    def extractText(tree):
        attributes = tree.attributes
        if attributes.getNamedItem('text').value:
            text = attributes.getNamedItem('text').value
        elif attributes.getNamedItem('content-desc').value:
            text = attributes.getNamedItem('content-desc').value
        else:
            text = attributes.getNamedItem('resource-id').value
        if text.startswith("com.") or text.startswith("me."):
            text = text.split("/")[-1]
            text = text.replace("_", " ")
        if len(re.findall("%", text)) >= 3:
            text = "link"

        return text

    @staticmethod
    def extractType(tree):
        attributes = tree.attributes
        return attributes.getNamedItem('class').value

    def updateTreeFlag(self, tree, tree_flag):
        tree_flag[tree] = tree_flag.get(tree, 0) + 1
        if tree.parentNode is not None:
            self.updateTreeFlag(tree.parentNode, tree_flag)
        else:
            return

    def process(self, path, reply):
        files = os.listdir(path)
        actions_path = os.path.join(path, "actions.json")
        if not os.path.exists(actions_path):
            return None
        with open(actions_path, 'r') as reader:
            actions = json.load(reader)

        turn_infos = {}
        actions_info = []
        files = [file for file in files if file.endswith("png")]
        files.sort(key=lambda file: int(file.split(".")[0]))

        for file_num, file in enumerate(files):
            i = int(file.split('.')[0])
            self.border = []
            self.border_item = []
            # image = Image.open(os.path.join(path, f"{i}.png"), 'r')
            image = os.path.join(path, f"{i}.png")
            dom = xdm.parse(os.path.join(path, f"{i}.xml"))

            self.dfs(dom)
            flag = self.postProcessing(self.border)
            border = [self.border[i] for i in range(len(self.border)) if flag[i] == 0]
            border_item = [self.border_item[i] for i in range(len(self.border_item)) if flag[i] == 0]

            # raw_image = image.copy()
            # fragments = self.extractImage(raw_image, border)

            texts = []
            for item in border_item:
                texts.append(self.extractText(item))

            types = []
            for item in border_item:
                types.append(self.extractType(item))

            if f"{i}" not in actions.keys():
                if file_num == len(files) - 1:
                    all_items = []
                    for k in range(len(border_item)):
                        all_items.append({
                            # "fragment": fragments[k],
                            "text": texts[k],
                            "type": types[k],
                            "border": border[k],
                        })

                    actions_info.append({
                        "image": image,
                        "action_info": f"test|response|NULL|{reply}",
                        "items": all_items,
                        "targets": None
                    })
                    continue
                else:
                    continue
            action = actions[f'{i}']
            if action is None:
                continue
            else:
                action = action.split("|")

            if action[1] == 'click':
                x, y = map(int, action[2].split(','))
                target_items = []
                all_items = []
                for k in range(len(border_item)):
                    all_items.append({
                        # "fragment": fragments[k],
                        "text": texts[k],
                        "type": types[k],
                        "border": border[k],
                    })
                    if border[k][0] <= x <= border[k][2] and border[k][1] <= y <= border[k][3]:
                        target_items.append(k)

                if y >= 2392:
                    actions_info.append({
                        "image": image,
                        "action_info": "text|back|NULL|NULL",
                        "items": all_items,
                        "targets": None
                    })
                else:
                    actions_info.append({
                        "image": image,
                        "action_info": actions[f"{i}"],
                        "items": all_items,
                        "targets": target_items
                    })
            elif action[1] in ["swipe", "input", "back", "clear", "enter"]:
                all_items = []
                for k in range(len(border_item)):
                    all_items.append({
                        # "fragment": fragments[k],
                        "text": texts[k],
                        "type": types[k],
                        "border": border[k],
                    })
                actions_info.append({
                    "image": image,
                    "action_info": actions[f"{i}"],
                    "items": all_items,
                    "targets": None
                })
            elif action[1] == 'read':
                continue

        turn_infos["actions"] = actions_info
        return turn_infos


def reformat_scroll(start_y, end_y):
    move_y = end_y - start_y
    if move_y > 0:
        return 1
    else:
        return 0


def process(path):
    processor = Processor()
    data = []
    dialog_folders = os.listdir(path)
    dialog_folders = [dialog_folder for dialog_folder in dialog_folders
                      if os.path.isdir(os.path.join(path, dialog_folder))]
    dialog_folders.sort(key=lambda x: int(x.split("_")[1]))
    dialog_tqdm = tqdm(dialog_folders)
    for dialog_folder in dialog_tqdm:
        dialog_path = os.path.join(path, dialog_folder)
        turn_folders = os.listdir(dialog_path)
        turn_folders = [turn_folder for turn_folder in turn_folders
                        if os.path.isdir(os.path.join(dialog_path, turn_folder))]
        turn_folders.sort(key=lambda x: int(x.split("_")[1]))

        dialogue_path = os.path.join(dialog_path, "dialog.json")
        with open(dialogue_path, 'r') as f:
            dialog = json.load(f)

        category_path = os.path.join(dialog_path, "category.txt")
        with open(category_path, 'r') as f:
            category = f.read().strip()

        dialog_history = []
        for turn_num, turn_folder in enumerate(turn_folders):
            turn_path = os.path.join(dialog_path, turn_folder)
            system_turn_dialog = dialog[2 * turn_num + 1]["text"]
            user_turn_dialog = dialog[2 * turn_num]["text"]
            items = processor.process(turn_path, system_turn_dialog)
            if items is None:
                continue

            dialog_history.append(user_turn_dialog)

            actions = items["actions"]
            screenshot_history = []
            action_history = []
            for action in actions:
                screenshot = action["image"]
                screenshot_history.append(screenshot)
                _, action_type, idx, text = action["action_info"].split("|")
                items = action["items"]
                if len(items) == 0:
                    continue
                target = action["targets"]
                if action_type == 'click' and len(target) == 0:
                    continue
                if action_type == 'click':
                    target = target[0]
                if action_type == 'home':
                    action_type = 'back'
                scroll_direction = None
                if action_type == 'swipe':
                    start_x, start_y = map(int, idx.split(","))
                    end_x, end_y = map(int, text.split(","))
                    scroll_direction = reformat_scroll(start_y, end_y)
                if action_type == "input":
                    inputs = text
                else:
                    inputs = None

                reply = None
                if action_type == "response":
                    reply = text

                data.append({
                    "screenshot_history": screenshot_history.copy(),
                    "action_history": action_history.copy(),
                    "dialog": dialog_history.copy(),
                    "items": items,
                    "action": action_type,
                    "response": reply if action_type == "response" else None,
                    "target": target,
                    "category": category,
                    "input": inputs,
                    "scroll": scroll_direction,
                    "turn": turn_path,
                })
                action_history.append(action)
            dialog_history.append(system_turn_dialog)

    with open(os.path.join(path, "data.json"), 'w') as writer:
        json.dump(data, writer, indent=1)


process("../dataset/train")
process("../dataset/dev")
