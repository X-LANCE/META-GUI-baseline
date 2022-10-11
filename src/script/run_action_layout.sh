export CUDA_VISIBLE_DEVICES=0;
CUDA_LAUNCH_BLOCKING=1 python train.py -e 8 -b 1 -tr -el -wl -s -t action -ga 4 -em microsoft/layoutlmv2-base-uncased -sp experiments/action_layoutlmv2_res_loss_weight_mm_test -tf ../dataset/train/data.json -ef ../dataset/dev/data.json -mm
