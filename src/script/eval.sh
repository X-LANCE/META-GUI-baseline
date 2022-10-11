export CUDA_VISIBLE_DEVICES=1;
CUDA_LAUNCH_BLOCKING=1 python train.py -b 1 -el -s -t action -em bert-base-uncased -sp experiments/action_bert_res -m experiments/action_bert_res/action_model.pt -tf ../dataset/train/data.json -ef ../dataset/dev/data.json
