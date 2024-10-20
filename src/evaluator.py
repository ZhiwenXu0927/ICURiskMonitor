# References:
# https://github.com/sindhura97/STraTS

from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
        true, pred = torch.cat(true), torch.cat(pred)
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        result = {'auroc':roc_auc, 'auprc':pr_auc, 'minrp':minrp}
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                              +str(train_step)+': '+str(result))
        return result

    def evaluate_gru_explained(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples, start+self.args.eval_batch_size)]
            data_batch = dataset.get_batch(batch_ind) #returns a dictionary
            tensor_a = data_batch['ts']
            tensor_b = data_batch['demo'] 
            tensor_b_expanded = tensor_b.unsqueeze(1).expand(-1, 24, -1)
            data = torch.cat((tensor_a, tensor_b_expanded), dim=2)
            true.append(data_batch['labels'])
            with torch.no_grad():
                p, _ = model(data)
                pred.append(p.cpu())
        true, pred = torch.cat(true), torch.cat(pred)
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        result = {'model_name':self.args.model_type,
                  'train_step':train_step,
                  'split':split,
                  'auroc':roc_auc, 
                  'auprc':pr_auc, 
                  'minrp':minrp}
        if train_step is not None:
            self.args.logger.write('Result on '
                    +result['split']+' split at train step '
                    +str(result['train_step'])+': AUROC: '
                    +str(result['auroc'])+': AUPRC: '
                    +str(result['auprc'])+': MINRP: '
                    +str(result['minrp']))
        return result
