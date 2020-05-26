# %%
import os
import numpy as np
import mne
#from properties import filename
import matplotlib.pyplot as plt                     # Este codigo es un plot basico para ver la señal, los datos concretos.
from matplotlib.transforms import Bbox
from scipy import signal
import matplotlib.pyplot as plt

mne.sys_info()     #chequear sistema

# Acá leemos un archivo particular
mne.set_log_level("WARNING")
raw= mne.io.read_raw_brainvision('C:/Users/Nicola/Documents/eeg/VISBRAIN/ExpS35.vhdr',
    preload=True, 
    eog=('EOG1_1','EOG2_1'),
    misc=('EMG1_1','EMG2_1'),
    verbose=True)
raw.rename_channels(lambda s: s.strip("."))

# -----------------------------------------------------------------
data =raw.get_data()         # data(chan,samp), times(1xsamples)
info = raw.info              #info
sfreq = info.get('sfreq')    #frecuencia de muestreo

#Con este código extraigo los datos que necesito y me rearmo la estructura que necesito para poder analizarlo mejor
data =raw.get_data()                                 # Saco los datos concretos, una matriz de numpy
new_data=data.copy()
canal_eogs = data[6,:] - data[7,:]                   # Cree la variable de la resta de las dos señales
canal_emgs =  data[8,:] - data[9,:]
t = np.linspace(1, round(data.shape[1]/sfreq), data.shape[1], endpoint=False)   
new_data[0]= signal.square(2 * np.pi * 1 * t)
new_data[1]=data[[0], :]
new_data[2]=data[[1], :]
new_data[3]=canal_eogs
new_data[4]=canal_emgs

new_data=new_data[[0,1,2,3,4], :]        #Elimino los otros canales

new_chnames =[ "Pulso", raw.ch_names[0], raw.ch_names[1], "EOG_resta", "EMG_resta"] 
new_chtypes = ['misc'] +['eeg' for _ in new_chnames[0:2]] + ['misc','misc'] # Recompongo los canales.

new_info = mne.create_info(new_chnames, sfreq, ch_types=new_chtypes)
new_info['meas_date'] = raw.info['meas_date']       # Registro el timestamp para las anotaciones.

new_raw=mne.io.RawArray(new_data, new_info)
new_raw.set_annotations(raw.annotations)           # Construyo un nuevo objeto raw que tiene lo que necesito.

scal = dict(mag=1e-12, grad=4e-11, eeg=20e-5, eog=150e-6, ecg=5e-4,emg=1e-4, ref_meg=1e-12, misc=1e-3, stim=1,
    resp=1, chpi=1e-4, whitened=1e2)

pplot=new_raw.plot(scalings=scal, duration=30, n_channels=10, block=True, )