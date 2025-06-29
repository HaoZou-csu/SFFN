import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)


def clips_loss(graph_embed, text_embed, temperature):
    logits = (graph_embed @ text_embed.T) / temperature
    graph_similarity = graph_embed @ graph_embed.T
    texts_similarity = text_embed @ text_embed.T

    targets = F.softmax(
        (graph_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    graph_loss = cross_entropy(logits, targets, reduction='none')
    text_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (graph_loss + text_loss) / 2.0  # shape: (batch_size)
    return loss.mean()

def gt_loss(graph_embed, text_embed, output, target, temperature, alpha):
    con_loss = clips_loss(graph_embed, text_embed, temperature)
    loss = mse_loss(output, target)
    return loss + alpha*con_loss


def gt3_loss(graph_embed, text_embed, mag_embed, output, target, temperature, alpha):
    con_loss = clips_loss(graph_embed, text_embed, temperature) + clips_loss(graph_embed, mag_embed, temperature) +\
    clips_loss(text_embed, mag_embed, temperature)
    loss = mse_loss(output, target)
    return loss + alpha*con_loss

def bce_loss(output, target):
    # bce = torch.nn.BCELoss()
    return F.binary_cross_entropy(output, target)

def bce_loss_with_logit(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target.view(-1).long())

def cross_entropy_loss(output, target):
    return cross_entropy(output, target)

def L1_loss(output, target):
    return F.l1_loss(output, target)