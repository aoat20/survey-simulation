import survey_simulation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

'''
img = np.asarray(Image.open('maps/Map2.png'))
print(img.shape)
img_tmp = img[:,:,0]
img_sz = img_tmp.shape
print(img_sz)
img_nan = np.empty(img_sz)
img_nan[:] = np.nan
img_mask = np.where(img_tmp==0, img_tmp, img_nan)
print(img_mask)

fig,ax = plt.subplots()
imgplot = ax.imshow(img)
ax.plot((0, 100, 200), (20, 300, 60))
ax.imshow(img_mask)
plt.show()
fig.canvas.draw()
'''

ss = survey_simulation.SurveySimulation('manual',
                                       save_loc='data')

#ss = survey_simulation.SurveySimulation('playback', 
#                                       save_loc='data/Episode0')

#ss = survey_simulation.SurveySimulation('test',
#                                       save_loc='data')
#for n in range(100):
#    rnd_mv = np.random.randint(0,100,size=(2)).tolist()
#    t, cov_map, contacts = ss.new_action('move', rnd_mv)
    
#    print(contacts)
    
    # At two arbitrary steps, demo group and ungroup actions
#    if n==45: 
#        ss.new_action('group', [0,1,2])
#    if n==65:
#        ss.new_action('ungroup', [0])
#        ss.new_action('group', [1,3,4])