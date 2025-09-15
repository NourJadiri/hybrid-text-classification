
from sklearn.metrics import roc_auc_score
import torch


def compute_metric(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    try:
        roc_auc_micro = roc_auc_score(y_true=labels, y_score=probs, average='micro')
    except ValueError:
        roc_auc_micro = 0.0

    return {'roc_auc_micro': roc_auc_micro}