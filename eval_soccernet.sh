#!/bin/bash
gpu=$1

#CUDA_VISIBLE_DEVICES=${gpu} python eval_soccer_net.py --server_run --data_folder england_epl/2014-2015 --model gpt
#CUDA_VISIBLE_DEVICES=${gpu} python eval_soccer_net.py --server_run --data_folder england_epl/2014-2015 --model gemini

CUDA_VISIBLE_DEVICES=${gpu} python eval_soccer_net.py --server_run --data_folder england_epl/2015-2016 --model gemini
CUDA_VISIBLE_DEVICES=${gpu} python eval_soccer_net.py --server_run --data_folder england_epl/2016-2017 --model gemini