#!/usr/bin/env python
import toml
import numpy as np

config_dict = toml.load("config.toml")

class Metrics:
    def macro_f_measure(self, labels, outputs):
        tp, fp, = (labels & outputs).sum(axis=0), ((~labels)&outputs).sum(axis=0)
        fn, tn = (labels&(~outputs)).sum(axis=0),((~labels)&(~outputs)).sum(axis=0)
        n, d = 2*tp, (2*tp+fp+fn)
        n = n[d!=0]
        d = d[d!=0]
        return np.mean(n/d)
    def micro_f_measure(self, labels, outputs):
        tp, fp, fn, tn = (labels & outputs).sum(), ((~labels)&outputs).sum(), (labels&(~outputs)).sum(),((~labels)&(~outputs)).sum()
        return 2*tp/(2*tp+fp+fn)
    def subset_accuracy(self, labels, outputs):
        return np.mean(np.all(np.where(labels == outputs, True, False), axis=1))
    def accuracy(self, labels, outputs):
        n,d = np.sum(np.logical_and(labels, outputs), axis = 1) , np.sum(np.logical_or(labels, outputs), axis=1)
        n = n[d!=0]
        d = d[d!=0]
        return np.mean(n/d)
    def hamming_loss(self, labels, outputs):
        _, num_classes = np.shape(labels)
        return np.nanmean((np.sum(np.logical_or(labels, outputs), axis=1)-np.sum(np.logical_and(labels, outputs), axis=1))/num_classes)
    def f_measure(self, labels, outputs):
        n,d = 2*np.sum(np.logical_and(labels, outputs),axis=1), np.sum(labels,axis=1)+np.sum(outputs, axis=1)
        n = n[d!=0]
        d = d[d!=0]
        return np.mean(n/d)
    @property
    def metrics(self):
        return {'Micro F1': lambda x,y : self.micro_f_measure(x,y),
                'Macro F1': lambda x,y : self.macro_f_measure(x,y),
                'F1': lambda x,y : self.f_measure(x,y),
                'Accuracy': lambda x,y : self.accuracy(x,y),
                'Subset Accuracy': lambda x,y : self.subset_accuracy(x,y),
                'Hamming Loss': lambda x,y : self.hamming_loss(x,y),}
    def compute(self, labels, outputs):
        res = dict()
        for key,value in self.metrics.items():
            res[key] = value(labels, outputs)
        return res
    def eval_and_show(self, labels, outputs):
        res = self.compute(labels, outputs)
        answer = [f"{key} = {value:.4f} | " for key,value in res.items()]
        return ''.join(answer)
