import torch

def l2_reg_loss(net1, net2):
    return sum(torch.norm(p1 - p2, p=2) for p1, p2 in zip(net1.parameters(), net2.parameters()))

def mmd_loss(source_features, target_features, source_labels, target_pseudo_labels):
    classes = torch.unique(source_labels)
    loss = 0.0
    for cls in classes:
        s_mask = source_labels == cls
        t_mask = target_pseudo_labels == cls
        if torch.sum(s_mask) > 1 and torch.sum(t_mask) > 1:
            s_feat = source_features[s_mask]
            t_feat = target_features[t_mask]
            loss += _mmd_rbf(s_feat, t_feat)
    return loss / len(classes)

def _mmd_rbf(x, y, gamma=1.0):
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())
    rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
    ry = torch.diag(yy).unsqueeze(0).expand_as(yy)
    K_xx = torch.exp(-gamma * (rx.t() + rx - 2 * xx))
    K_yy = torch.exp(-gamma * (ry.t() + ry - 2 * yy))
    K_xy = torch.exp(-gamma * (torch.cdist(x, y, p=2) ** 2))
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
