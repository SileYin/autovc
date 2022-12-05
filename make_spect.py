import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
from resemblyzer import VoiceEncoder, preprocess_wav
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = '/home/sile/VCTK/sentence_split/val'
# spectrogram directory
targetDir = './spmel_pitch_shift'

device = 'cuda:0'
# embedding_enc = VoiceEncoder()
embedding_enc = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder').to(device)
embedding_enc.eval()

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

speakers = []
subdirList = ['p225', 'p228', 'p252', 'p256', 'p261', 'p270']
# subdirList.remove('p225')
# subdirList.remove('p228')
# subdirList.remove('p256')
# subdirList.remove('p270')

for subdir in sorted(subdirList):
    emb = []
    utterances = []
    utterances.append(subdir)
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    prng = RandomState(int(subdir[-1:])) 
    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        emb.append(embedding_enc(torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(0)).squeeze().detach().cpu().numpy())
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    
        # save spect    
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)  
        utterances.append(os.path.join(subdir, fileName[:-4]+'.npy'))  
    utterances.insert(1, np.mean(np.array(emb), axis=0))
    speakers.append(utterances)
        
with open(os.path.join(targetDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)