export CUDA_VISIBLE_DEVICES=0;
CUDA_LAUNCH_BLOCKING=1 python train.py -e 8 -b 1 -tr -el -wl -bs -s -t reply -ga 4 -em microsoft/layoutlmv2-base-uncased -sp experiments/reply_layoutlmv2_res_mm -tf ../dataset/train/data.json -ef ../dataset/dev/data.json -mm
