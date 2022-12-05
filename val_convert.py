import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from tqdm import tqdm
import random

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:1'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load('./autovc.ckpt')
G.load_state_dict(g_checkpoint['model'])

data_root_dir = './spmel_test_old_emb'

metadata = pickle.load(open(data_root_dir + '/train.pkl', "rb"))

spect_vc = []

for sbmt_i in metadata:
             
    x_org_files = sbmt_i[2:]
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in tqdm(metadata):
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)

        # f = random.choice(x_org_files)

        # x_org = np.load(os.path.join(data_root_dir, f))
        # x_org, len_pad = pad_seq(x_org)
        # uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

        # with torch.no_grad():
        #     _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        # if len_pad == 0:
        #     uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        # else:
        #     uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        # spect_vc.append( ('{}/{}_{}_{}'.format(sbmt_j[0], sbmt_i[0], sbmt_j[0], f.split('/')[-1].split('_')[1]), uttr_trg) )
        
        # print(f'converting from {sbmt_i[0]} to {sbmt_j[0]}')

        for f in tqdm(x_org_files):

            x_org = np.load(os.path.join(data_root_dir, f))
            x_org, len_pad = pad_seq(x_org)
            uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
            spect_vc.append( ('{}/{}_{}_{}'.format(sbmt_j[0], sbmt_i[0], sbmt_j[0], f[:-4].split('/')[-1].split('_')[1]), uttr_trg) )

with open(os.path.join(data_root_dir, 'mel_results.pkl'), 'wb') as handle:
    pickle.dump(spect_vc, handle)    