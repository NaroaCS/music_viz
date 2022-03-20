from spectrogram import*
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks

filepath='permanent.wav' #music file path.

#Define size for visualization (x,y)
a=30
b=30
result = plotstft(filepath) #Compute averaged spectogram with windows a and b.

x = np.arange(a)
plt.style.use("dark_background")

fig, ax = plt.subplots()
for i in range(b):
    y=np.ones(a)*i
    plt.scatter(x,y,s=0.00000002*result[:,i]**4.5, c='white')
ax.axis('off')
plt.show()
