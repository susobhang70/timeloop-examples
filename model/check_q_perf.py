import kaldiio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
import random
#Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
from sklearn.metrics import classification_report
import torch.quantization.quantize_fx as quantize_fx
import copy
torch.backends.quantized.engine = 'qnnpack'

from cnn_raw import CNN
from cnn_small_v1 import CNNSmallV1
from cnn_small_v2 import CNNSmallV2
from cnn_small_v3 import CNNSmallV3
from cnn_effnet import EffNetOri


class FHSData(torch.utils.data.Dataset):
    def __init__(self, feats_scp, utt2label, cmvn_file=None):
        super().__init__()
        self.uttids = []
        self.utt2feats = {}
        self.utt2labs = {}
        with open(feats_scp) as f:
            for line in f:
                splits = line.rstrip().split()
                self.uttids.append(splits[0])
                self.utt2feats[splits[0]] = splits[1]
        
        with open(utt2label) as f:
            for line in f:
                splits = line.rstrip().split()
                self.utt2labs[splits[0]] = int(splits[1])
        
        if cmvn_file is not None:
            self.cmvn = self._load_cmvn(cmvn_file)
        else:
            self.cmvn = None
    
    def __len__(self):
        return len(self.uttids)
    
    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        feats_path = self.utt2feats[uttid]
        features = kaldiio.load_mat(feats_path).copy()
        if self.cmvn is not None:
            features = self._apply_cmvn(features, self.cmvn)
        label = self.utt2labs[uttid]
        return {'uttid': uttid, 'feats': torch.tensor(features).float(),
               'target': torch.tensor(label).long()}
    
    def _load_cmvn(self, cmvn_file):
        cmvn = kaldiio.load_mat(cmvn_file)
        assert cmvn.shape[0] == 2
        cnt = cmvn[0, -1]
        sums = cmvn[0, :-1]
        sums2 = cmvn[1, :-1]
        means = sums / cnt
        stds = np.sqrt(np.maximum(1e-10, sums2 / cnt - means ** 2))
        return means, stds
    
    def _apply_cmvn(self, features, cmvn):
        # https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc
        means, stds = cmvn
        features -= means
        features /= stds
        return features

def _collate_fn(batch):
    # max_len = max(len(ex['feats']) for ex in batch)
    max_len = 1024
    batch_feats = torch.zeros(len(batch), max_len, batch[0]['feats'].shape[-1])
    batch_targets = []
    batch_lens = []
    for i, ex in enumerate(batch):
        feats = ex['feats']
        tgt = ex['target']
        batch_feats[i, :len(feats)] = feats
        batch_lens.append(len(feats))
        batch_targets.append(tgt)
    
    batch_targets = torch.stack(batch_targets, dim=0).long()
    batch_lens = torch.tensor(batch_lens).long()
    return {'feats': batch_feats,'feats_len': batch_lens, 'targets': batch_targets}


def count_parameters(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return mem

dev_wav_scp = '/data/sls/u/sameerk/code/kaldi/egs/librispeech/s5/data/dev_fra/feats.scp'
dev_utt2label = '/data/sls/u/sameerk/code/kaldi/egs/librispeech/s5/data/dev_fra/utt2label'
cmvn_file = '/data/sls/scratch/sameerk/fhs_prepared/kaldi_data/fbanks/cmvn.ark'
dev_ds = FHSData(dev_wav_scp, dev_utt2label, cmvn_file)
dev_ds = torch.utils.data.DataLoader(dev_ds, batch_size=128, collate_fn=_collate_fn, drop_last=False, 
                                    num_workers=2, shuffle=False)

ckpt_root = '/data/sls/scratch/alexhliu/course/hw4dl/final_project_ckpts/'



modules = [ CNN, CNNSmallV1, CNNSmallV2, CNNSmallV3, EffNetOri]
ckpts = [ 'auc_0.852169_seed_2222_wd_1e-07_lr_0.010000.pt',
          'auc_0.856023_seed_10001_wd_1e-07_lr_0.010000.pt',
          'auc_0.903145_seed_1234_wd_1e-07_lr_0.010000.pt',
          'auc_0.892956_seed_6825_wd_0.001_lr_0.010000.pt',
          'auc_0.911797_seed_10001_wd_1e-07_lr_0.030000.pt',
        ]


for module, ckpt in zip(modules, ckpts):
    print('###############################')
    print(f'###### {module.__name__} #####')
    print('###############################')
    # fp32 model
    model_fp = module()
    ckpt = model_fp.load_state_dict(torch.load(ckpt_root+ckpt))
    y_pred = []
    q_pred = []
    y_true = []
    model_fp.eval()
    # model_fp.cuda()

    # quantized model
    model_to_quantize = copy.deepcopy(model_fp)
    qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
    model_to_quantize.eval()
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
    model_quantized = quantize_fx.convert_fx(model_prepared)
    print("Raw size:",float(count_parameters(model_fp))/1000000)
    print("Quantized size:",float(count_parameters(model_quantized))/1000000)

    # GPU
    model_fp = model_fp.cuda()
    # model_quantized = model_quantized.cuda()

    with torch.no_grad():
        for batch in tqdm.tqdm(dev_ds, total=len(dev_ds)):
            x = batch['feats']
            # use lens to do average pooling properly
            x_lens = batch['feats_len']
            tgts = batch['targets']
            x = x.cuda()
            tgts = tgts.cuda()
            p = model_fp(x).squeeze(1)
            y_pred.append(p.detach().cpu().numpy())
            y_true.append(tgts.cpu().numpy())

            q = model_quantized(x.cpu()).squeeze(1)
            q_pred.append(q.detach().cpu().numpy())

        auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))
        y_pred_class = (np.concatenate(y_pred)>0.5).astype(float)

        q_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(q_pred))
        q_pred_class = (np.concatenate(q_pred)>0.5).astype(float)
        print('--------------- fp32 ---------------')
        print('     AUC:',auc)
        print(classification_report(np.concatenate(y_true), y_pred_class))
        print('--------------- int8 ---------------')
        print('     AUC:',q_auc)
        print(classification_report(np.concatenate(y_true), q_pred_class))

