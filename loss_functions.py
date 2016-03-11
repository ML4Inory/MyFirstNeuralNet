# -*- coding:utf-8 -*-

def svm_loss(scores, y, margin):
	loss = max(0, scores[i]-scores[y]+margin)