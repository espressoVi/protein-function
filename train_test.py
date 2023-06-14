#!/usr/bin/env python
import torch,toml
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from tqdm import trange,tqdm

config_dict = toml.load("config.toml")
train_param = config_dict['train']

def train(model, device, dataset, metrics):
    train_dataset, val_dataset = dataset.get_train_dataset()
    infer = dataset.fill
    """ train_dataset, val_dataset are torch TensorDatasets
    which contains (input_ids, attention_masks, labels). """
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['MINI_BATCH_SIZE'], shuffle = True, )
    optimizer_parameters = [param for name, param in model.named_parameters() if 'protbert' not in name]
    #optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-4)
    epochs = train_param['EPOCHS']
    counter, train_loss, loss = 1, 0.0,0.0
    model.zero_grad()
    for _ in trange(int(epochs), desc="Epoch"):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],
                      "attention_masks": batch[1],
                      "labels":batch[2],}
            loss, predicts = model(**inputs)
            train_loss += loss.item()
            loss.backward()
            if (i+1)%train_param['ACCUMULATE'] == 0:
                optimizer.step()
                model.zero_grad()
            counter += 1
            epoch_iterator.set_description("Loss :%f" % (train_loss/counter))
            epoch_iterator.refresh()
        evaluate(model, device, val_dataset, infer, metrics)

def evaluate(model, device, val_dataset, infer, metrics):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    F1 = metrics.metrics['CAFAMetric']
    preds, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        inputs = { "input_ids": batch[0],
                  "attention_masks": batch[1]}
        with torch.no_grad():
            pred = model(**inputs)
        pred_batch = pred.detach().cpu().numpy()
        label_batch = batch[2].detach().cpu().numpy()
        preds.extend(pred_batch)
        labels.extend(label_batch)
    preds,labels = np.array(preds),np.array(labels)
    threshold = find_threshold(F1, labels, preds, infer)
    outputs = infer((preds>threshold).astype(int))
    print(metrics.eval_and_show(labels, outputs))

def find_threshold(metric, labels, preds, infer):
    best = 0
    for i in tqdm(np.linspace(0.1,np.amax(preds),100), desc="Tuning threshold"):
        score = metric(labels, infer((preds>i).astype(int)))
        if score > best:
            best = score 
            threshold = i
    return threshold

def write_predictions(model, device, dataset):
    pass
