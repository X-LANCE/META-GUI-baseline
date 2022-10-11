export CUDA_VISIBLE_DEVICES=1;
CUDA_LAUNCH_BLOCKING=1 python train.py -b 1 -el -s -t reply -em microsoft/layoutlm-base-uncased -m experiments/reply_layout_res_beam_3/reply_model.pt -sp experiments/reply_layout_res_beam_3 -tf ../dataset/train/data.json -ef ../dataset/dev/data.json
