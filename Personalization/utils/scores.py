import torch
import torch.nn.functional as F
# from tabulate import tabulate
# from sklearn.metrics import confusion_matrix

def ece(preds, target, device, minibatch=True):
    confidences, predictions = torch.max(preds, 1)
    target = target.float() # to avoid a warning error
    _, target_cls = torch.max(target, 1)
    accuracies = predictions.eq(target_cls)
    n_bins = 100 
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin 

    return ece.item()

def nll(preds, target, minibatch=True):
    logpred = torch.log(preds + 1e-8)
    if minibatch:
        return -(logpred * target).sum(1).item()
    else:
        return -(logpred * target).sum(1).mean().item()

def acc(preds, target, minibatch=True):
    preds = preds.float() #to avoid a warning error
    preds = preds.argmax(1)
    target = target.float() # to avoid a warning error
    target = target.argmax(1)
    if minibatch:
        return (((preds == target) * 1.0).sum() * 100).item()
    else:
        return (((preds == target) * 1.0).mean() * 100).item()

def compute_scores(model, dataloader, args, optimizer, device="cpu", n_sample=1):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
        
    preds = []
    targets = []
    model.to(device)
    if args.optimizer == 'ivon': 
        optimizer._device = device

    if args.optimizer == 'ivon': 
        with torch.no_grad():
            for tmp in dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device,dtype=torch.int64)
                    outs = []
                    if n_sample == 0: #predict with the mean
                        out = model(x)
                        out = F.softmax(out, 1)
                        outs.append(out)

                    else : 
                        for _ in range(n_sample):
                            with optimizer.sampled_params(): #the only difference with the following block
                                out = model(x)
                                out = F.softmax(out, 1)
                                outs.append(out)

                    preds.append(torch.stack(outs).mean(0))
                    targets.append(F.one_hot(target, model.output_dim))
    else:
        with torch.no_grad():
            for tmp in dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device,dtype=torch.int64)
                    outs = []
                    for _ in range(n_sample):
                        out = model(x)
                        out = F.softmax(out, 1)
                        outs.append(out)

                    preds.append(torch.stack(outs).mean(0))
                    targets.append(F.one_hot(target, model.output_dim))

    targets = torch.cat(targets)
    preds = torch.cat(preds)

    _acc = acc(preds, targets, minibatch=False)
    _ece = ece(preds, targets, device, minibatch=False)
    _nll = nll(preds, targets, minibatch=False)

    if was_training:
        model.train()
    return _acc, _ece, _nll