# %%
#  "import" de las librerías que vamos a estar usando.
import os
import numpy as np
import mne
from properties import filename

from matplotlib.transforms import Bbox


# versión de MNE, chequear que estamos usando mne3
mne.sys_info()
# Acá leemos un archivo particular
mne.set_log_level("WARNING")
raw= mne.io.read_raw_brainvision(filename,
    preload=True, 
    eog=('EOG1_1','EOG2_1'),
    misc=('EMG1_1','EMG2_1'),
    verbose=True)
raw.rename_channels(lambda s: s.strip("."))

# -----------------------------------------------------------------
# data(chan,samp), times(1xsamples)

# (1) Aca voy a ver los datos en crudo, ploteandolos por afuera de MNE.  Fijense que los datos estan en Volts lo paso a microvolts.

channel = 0
eeg = raw[channel][0][0][0:250*4]  * pow(10,6)      # Tomo 4 segundos.


import matplotlib.pyplot as plt                     # Este codigo es un plot basico para ver la señal, los datos concretos.
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(eeg,'r', label='EEG')
plt.legend(loc='upper left')
plt.show(block=False)

#a = raw.plot(show_options=True,title='KComplex2',start=504,duration=30,n_channels=10, scalings=dict(eeg=20e-6))
# chequean que la frecuencia de sampleo sea la esperada
print('Sampling Frequency: %d' %  raw.info['sfreq'] )

pplot = raw.plot(scalings='auto',n_channels=10,block=True, )


# (2) Con este código extraigo los datos que necesito y me rearmo la estructura que necesito para poder analizarlo mejor

ch_names = ['peak'] + raw.ch_names              # Saco el nombre de los canales pero agrego uno 'peak'
sfreq = raw.info['sfreq']
data =raw.get_data()                            # Saco los datos concretos, una matriz de numpy

dat = np.concatenate( (np.zeros((1,data.shape[1])), data), axis=0)    # Le agrego a los datos un array con zeros.
                                                                      # aca si ustedes quieren pueden agregarle 20 o algo asi
                                                                      # cada vez que la señal que ustedes analizan supera los 75

ch_types = ['misc'] + ['eeg' for _ in ch_names[0:6]] + ['eog','eog','emg', 'emg']   # Recompongo los canales.
info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

info['meas_date'] = raw.info['meas_date']       # Registro el timestamp para las anotaciones.

reraw = mne.io.RawArray(dat, info)
reraw.set_annotations(raw.annotations)          # Construyo un nuevo objeto raw que tiene lo que necesito.

reraw.plot(scalings='auto',n_channels=10,block=True, )



# POR Ejemplo con esto pueden restar EMG1 y EMG2 y dejar solo uno, lo mismo con EOG.