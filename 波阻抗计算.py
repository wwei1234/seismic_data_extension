import segyio
import numpy as np
import matplotlib.pyplot as plt

def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) #number of all trace
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) #shot number 
        len_shot = trace_num//shot_num   #The length of the data in each shot data
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data
    
density = read_segy(r'elastic-marmousi-model\model\MODEL_DENSITY_1.25m.segy', shotnum=1)[0]
vel = read_segy(r'elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy', shotnum=1)[0]
impe = density * vel

np.save(r'data/impe', impe)
np.save(r'data/density', density)
np.save(r'data/velocity', vel)
plt.figure()
plt.imshow(impe, "seismic", aspect='auto')
plt.colorbar()
plt.show()