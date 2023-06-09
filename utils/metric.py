#!/usr/bin/env python
import toml
import numpy as np

class Metrics:
    def _compute_confusion_matrices(self, labels, outputs, ):
        num_recordings, num_classes = np.shape(labels)
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
        return A

    def macro_f_measure(self, labels, outputs):
        _, num_classes = np.shape(labels)
        A = self._compute_confusion_matrices(labels, outputs)
        f_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if 2 * tp + fp + fn:
                f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                f_measure[k] = float('nan')
        macro_f_measure = np.nanmean(f_measure)
        return macro_f_measure

    def micro_f_measure(self, labels, outputs):
        num_recordings, num_classes = np.shape(labels)
        A = self._compute_confusion_matrices(labels, outputs)
        tp, fp, fn, tn = np.sum(A[:, 1, 1]), np.sum(A[:, 1, 0]), np.sum(A[:, 0, 1]), np.sum(A[:, 0, 0])
        if 2 * tp + fp + fn:
            return float(2 * tp) / float(2 * tp + fp + fn)
        else:
            return float('nan')

    def subset_accuracy(self, labels, outputs):
        num_recordings, num_classes = np.shape(labels)
        cnt = 0
        for label, output in zip(labels, outputs):
            if all(label == output):
                cnt+=1
        return float(cnt)/num_recordings

    def accuracy(self, labels, outputs):
        num_recordings, num_classes = np.shape(labels)
        su = 0
        for label, output in zip(labels, outputs):
            temp = np.sum(np.logical_and(label, output))/np.sum(np.logical_or(label, output))
            su += temp
        return float(su)/num_recordings

    def hamming_loss(self, labels, outputs):
        num_recordings, num_classes = np.shape(labels)
        su = 0
        for label, output in zip(labels, outputs):
            score = (np.sum(np.logical_or(label, output))-np.sum(np.logical_and(label, output)))/num_classes
            su+=score
        return su/num_recordings

    def f_measure(self, labels, outputs):
        num_recordings, num_classes = np.shape(labels)
        su = 0
        for label, output in zip(labels, outputs):
            temp = 2*np.sum(np.logical_and(label, output))/(np.sum(label)+np.sum(output))
            su += temp
        return float(su)/num_recordings

    @property
    def metrics(self):
        return {'MicroF1': lambda x,y : self.micro_f_measure(x,y),
                'MacroF1': lambda x,y : self.macro_f_measure(x,y),
                'F1': lambda x,y : self.f_measure(x,y),
                'Accuracy': lambda x,y : self.accuracy(x,y),
                'SubsetAccuracy': lambda x,y : self.subset_accuracy(x,y),
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

metrics = Metrics()
