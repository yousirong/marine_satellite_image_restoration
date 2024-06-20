import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import geometric 
import seaborn as sns

#plt.rcParams['figure.figsize'] = (10,10)
mask = np.loadtxt('/home/pmilab/Documents/validation_data/recon/GOCI_RRS_degree/2021/2/50/mask/COMS_GOCI_L2A_GA_20210326041643._r4096_c4608', delimiter=',',dtype='float32')
gt = np.loadtxt('/home/pmilab/Documents/GOCI/Chl-a/2021/0/COMS_GOCI_L2A_GA_20210326041643._r4096_c4608', delimiter=',',dtype='float32')
restored = np.loadtxt('/home/pmilab/Documents/validation_data/recon/chl/2021/50/COMS_GOCI_L2A_GA_20210326041643._r4096_c4608', delimiter=' ',dtype='float32')

loss_data = gt * mask 
restored_data = (1-mask) * restored + loss_data

#ax1 = sns.heatmap(loss_data, annot=False, vmin=0,vmax=1)
#ax2 = sns.heatmap(restored_data, annot=False, vmin=0,vmax=1)
ax3 = sns.heatmap(gt, annot=False, vmin=0,vmax=1)

#fig1 = ax1.figure
#fig2 = ax2.figure
fig3 = ax3.figure
#fig.tight_layout()
#fig1.savefig('./recon/test/input.png')
#fig2.savefig('./recon/test/restored_data.png')
fig3.savefig('./recon/test/gt.png')