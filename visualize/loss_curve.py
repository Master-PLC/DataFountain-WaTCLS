#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :loss_curve.py
@Description  :
@Date         :2022/01/11 12:48:22
@Author       :Arctic Little Pig
@version      :1.0
'''

import matplotlib.pyplot as plt
import pandas as pd

train_log_file = "../results/wxf.2layer.8head.bsz_32.adam0.0002.clip10.resnet50.ep30/train.log"
valid_log_file = "../results/wxf.2layer.8head.bsz_32.adam0.0002.clip10.resnet50.ep30/train.log"

if __name__ == "__main__":
    loss_table = pd.read_table(train_log_file, sep=',', header=None)
    loss = loss_table.iloc[:, 1].to_numpy()
    mAP_table = pd.read_table(valid_log_file, sep=',', header=None)
    mAP = mAP_table.iloc[:, -1].to_numpy() * 100
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.tick_params(labelsize=17)
    plt.plot(loss, 'r', label="train loss")
    plt.xlabel("epochs", fontsize=17)
    plt.ylabel("train loss", fontsize=17)
    plt.legend(fontsize=17, loc='upper left')

    ax2 = ax1.twinx()
    plt.plot(mAP, 'g', label="valid meanAP")
    plt.legend(loc='upper right', fontsize=17)
    ax2.tick_params(labelsize=17)
    ax2.set_ylabel("valid meanAP", fontsize=17)

    plt.xlim(0, 30)
    plt.show()
