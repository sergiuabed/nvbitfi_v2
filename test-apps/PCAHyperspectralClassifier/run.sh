#!/bin/bash
#eval ${PRELOAD_FLAG} ${BIN_DIR}/simple_add > stdout.txt 2> stderr.txt

#eval ${PRELOAD_FLAG} python3 ${BIN_DIR}/inference.py --cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10 > stdout.txt 2> stderr.txt
eval ${PRELOAD_FLAG} python3 ${BIN_DIR}/inference.py --cuda 0 --image IndianPines --checkpoint ${LI_MODEL_CHECKPOINTS_PATH}/ip/10_IP.pth --model li --pca 10 > stdout.txt 2> stderr.txt

#python3 inference.py --cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10