import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error, roc_auc_score, precision_score, recall_score,
                             f1_score, precision_recall_curve, auc)
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = (output>0.5).int()
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def auc(output, target):
    with torch.no_grad():
        prediction = output.cpu().numpy().ravel()
        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        auc_score = roc_auc_score(target_label, prediction)
    return auc_score

'''
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score

'''

def cg_auc(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        auc_score = roc_auc_score(target_label, prediction[:, 1])
    return auc_score

def cg_precision(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        pred_label = np.argmax(prediction, axis=1)

        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        precision = precision_score(target_label, pred_label)
    return precision

def cg_recall(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        pred_label = np.argmax(prediction, axis=1)

        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        recall = recall_score(target_label, pred_label)
    return recall

def cg_f1(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        pred_label = np.argmax(prediction, axis=1)

        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        f1 = f1_score(target_label, pred_label)
    return f1

def cg_aupr(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        pred_label = np.argmax(prediction, axis=1)

        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        precision, recall, _ = precision_recall_curve(target_label,prediction[:, 1])
        aupr = auc(recall, precision)
    return aupr



def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mae(output, target):
    with torch.no_grad():
        error = torch.absolute(output - target)
        return torch.mean(error).item()

def rmse(output, target):
    with torch.no_grad():
        error = torch.nn.functional.mse_loss(output, target)
        return torch.sqrt(error).item()

def mse(output, target):
    with torch.no_grad():
        error = torch.nn.functional.mse_loss(output, target)
        return error.item()

def g_auc(output, target):
    with torch.no_grad():
        prediction = np.exp(output.cpu().numpy())
        target = target.cpu().numpy()
        target_label = np.squeeze(target)
        auc_score = roc_auc_score(target_label, output.cpu().numpy())
    return auc_score