# META-GUI-baseline
This repository contains the full pipeline to train and evaluate the baseline models in the paper [META-GUI: Towards Multi-modal Conversational Agents on Mobile GUI](https://arxiv.org/abs/2205.11029) on the META-GUI dataset. The dataset and leaderboard can be found [here](https://x-lance.github.io/META-GUI-Leaderboard/).

## Latest Experiment Result

method | Action CR | Turn CR | Reply BLEU score 
-------| --- | --- |---
Random | 5.71 | 3.99 | 0.71
MFM | 8.91 | 0.00 | 9.29
FM | 10.00 | 6.76 | 7.88
LayoutLMv2 | 64.48 | 36.88 | 58.20
LayoytLM | 67.76 | 38.12 | 50.43
BERT |78.42 | 52.08 | 62.19
m-BASH | **82.74** | **56.88** | **63.11**

## Requirements

The required python packages is listed in "requirements.txt". You can install them by
```commandline
pip install -r requirements.txt
```
or
```commandline
conda install --file requirements.txt
```

## Dataset Description

Please first download the dataset from [Amazon](https://meta-gui.s3.us-west-1.amazonaws.com/meta_gui.tar), and unzip the file in the main folder.

The train and development dataset are stored in `/dataset/train` and `/dataset/dev` respectively. And the `data.json` file under these two folders are the processed data, generated with `/src/processors.py`. You can modify `/src/processors.py` to generate data with the format you need.

The format of `data.json` is `List[Dict]`. The keys contains `screenshot_history`, `action_history`, `dialog`, `items`, `action`, `response`, `target`, `category`, `input`, `scroll` and `turn`. 

* `screenshot_history`: `List[str]`, the screenshot history of the current dialogue turn.
* `action history`: `List[Dict]`, the action history of the current dialogue turn. Each dict contains the corresponding screen `image`, the action performed on the screen `action_info`, the items extracted from the corresponding view hierarchy `items` and the target item to be clicked if the action type is click `target`.
* `dialog`: `List[str]`, the dialogue history.
* `items`: `List[Dict]`, the items extracted from corresponding view hierarchy. Each dict contains the text information `text`, the item type `type` and the bounding box `border`.
* `action`: `str`, the action type.
* `response`: `Union[str, None]`, the response text.
* `target`: `Union[int, None]`, the id of the target item from `items` if the action type is `click`.
* `category`: `str`, the domain of current data point.
* `input`: `Union[str, None]`, the parameter for `input` action.
* `scroll`: `Union[int, None]`, the parameter for `swipe` action.
* `turn`: `str`, the turn id. 

The folders with prefix `dialog` are the raw data, whose format are as follows:
```
dialog_{id}
  - dialog_id.txt
  - dialog.json
  - category.txt
  - meta.json
  - turn_0
    - actions.json
    - 0.png
    - 0.xml
    - 1.png
    - 1.xml
    - ...
  - turn_1
  - ...
```

* `dialog_id.txt` contains the `id` for this dialogue data. 

* `dialog.json` contains the dialogue data, and the format is `List[Dict]`. The keys contain `isUser`, `program` and `text`. `isUser` means whether the speaker is user or not, `program` is the Chinese translation of `text` which is used for annotation for the convenient of annotators and may be missing, and `text` is what the speaker says. 

* `category.txt` identifies the domain for this dialogue data.

* `meta.json` contains the related apps of each dialogue turn.

* `actions.json` contains the step-by-step actions performed on the screen.

* `*.png` is the screenshot and `*.xml` is the corresponding view hierarchy.

## Training

After downloading the data, the baseline models can be trained. To do so, stay in the `src` directory and run the `run_action_layout.sh` or `run_reply_layout.sh` files in the directory `./script`, which are used for training Action model and Reply model respectively. For example, to train the Action model, run the following command under the `src` folder:
```commandline
bash ./script/run_action_layout.sh
```

## Evaluation

The `eval.sh` and `eval_reply.sh` files which can evaluate the performance of Action model and Reply model on the development set are placed in the same folder as the `run_action_layout.sh` files for the same method. For example, to evaluate the performance of Action model, run the following command under the `src` folder:
```commandline
bash ./script/eval.sh
```

## Reference

If you use any source codes or datasets included in this repository in your work, please cite the corresponding papers. The bibtex are listed below:
```text
@article{sun2022meta,
  title={META-GUI: Towards Multi-modal Conversational Agents on Mobile GUI},
  author={Sun, Liangtai and Chen, Xingyu and Chen, Lu and Dai, Tianle and Zhu, Zichen and Yu, Kai},
  journal={arXiv preprint arXiv:2205.11029},
  year={2022}
}
```
