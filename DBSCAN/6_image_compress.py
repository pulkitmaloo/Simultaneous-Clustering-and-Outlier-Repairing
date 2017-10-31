# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 02:46:09 2017

@author: PulkitMaloo
"""

from skimage import io
import pandas as pd
from QDORC import QDORC
import numpy as np
image = io.imread('image_text.png')
io.imshow(image)
io.show()
rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0]*image.shape[1],4 )
data = pd.DataFrame(image)


print('start')
import time
start_time = time.time()
P = QDORC(data,.0005,15)
print("--- %s seconds ---" % (time.time() - start_time))
io.imshow(np.array(P).reshape(rows, cols, 4))
io.show()
print('end')