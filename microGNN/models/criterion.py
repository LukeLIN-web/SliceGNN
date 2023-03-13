import torch.nn.functional as F


def criterion(logits, labels, dataset_name):
    if dataset_name == "ogbn-products" or dataset_name == "papers100M":
        loss = F.cross_entropy(logits, labels.squeeze())
    elif dataset_name == "yelp":
        loss = F.cross_entropy(logits, labels)
    elif dataset_name == "reddit":
        loss = F.nll_loss(logits, labels)
    else:
        raise NotImplementedError
    return loss
