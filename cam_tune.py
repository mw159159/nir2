# Copyright (C) 2019 Max Masalsky
#
# GNU General Public License
# <http://www.gnu.org/licenses/>.
#

import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import json

#---------------------------------
imageL = './imageL.jpg'
imageR = './imageR.jpg'

pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
loading_settings=0
evx,evy=0.0,0.0

#cap = cv2.VideoCapture(2)
#cap2 = cv2.VideoCapture(3)
print('Read image...')
imgL = cv2.imread(imageL)
imgR = cv2.imread(imageR)
#print(imgR.shape)
imgR = imgR[:,:,::-1]

# Set up and draw interface
axcolor = 'lightgoldenrodyellow'
#fig = plt.subplots(1,2)
#plt.subplots_adjust(left=0.15, bottom=0.5)
#plt.subplot(1,2,1)
#img2Object = plt.imshow(imgR)
#cv2.imshow('it', imgL)
#cv2.waitKey()
#cv2.destroyAllWindows()
fig1 = plt.figure()
img2=imgR
img2Object = plt.imshow(img2, aspect='equal', cmap='jet')
plt.axis('off')
#fig1.gca().axes.get_xaxis().set_visible(False)
#fig1.gca().axes.get_yaxis().set_visible(False)
fig1.subplots_adjust(bottom=0,top=1,left=0,right=1)
#plt.show()
#exit(0)
#Button Save----------------------------------------
def onclick(event):
    global evx,evy
    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #      ('double' if event.dblclick else 'single', event.button,
    #       event.x, event.y, event.xdata, event.ydata))
    if event.dblclick:
        evx=event.xdata
        evy=event.ydata
    update(2)

cid = fig1.canvas.mpl_connect('button_press_event', onclick)
plt.figure()
saveax = plt.axes([0.3, 0.38, 0.15, 0.04]) #stepX stepY width height
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
def save_map_settings( event ):
    buttons.label.set_text ("Saving...")
    print('Saving to file...')
    result = json.dumps({'pt1x':pt1x, 'pt1y':pt1y, 'pt2x':pt2x, \
             'pt2y':pt2y, 'pt3x':pt3x, 'pt3y':pt3y, \
             'ptAllx':ptAllx, 'ptAlly':ptAlly},\
             sort_keys=False, indent=4, separators=(',',':'))
    fName = 'cam_set.txt'
    f = open (str(fName), 'w')
    f.write(result)
    f.close()
    buttons.label.set_text ("Save to file")
    print ('Settings saved to file '+fName)
buttons.on_clicked(save_map_settings)
#Button Load----------------------------------------
loadax = plt.axes([0.5, 0.38, 0.15, 0.04]) #stepX stepY width height
buttonl = Button(loadax, 'Load', color=axcolor, hovercolor='0.975')
def load_map_settings( event ):
    global pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly, loading_settings
    loading_settings = 1
    fName = 'cam_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text ("Loading...")
    f=open(fName, 'r')
    data = json.load(f)
    spt1x.set_val(data['pt1x'])
    spt1y.set_val(data['pt1y'])
    spt2x.set_val(data['pt2x'])
    spt2y.set_val(data['pt2y'])
    spt3x.set_val(data['pt3x'])
    spt3y.set_val(data['pt3y'])
    sptAllx.set_val(data['ptAllx'])
    sptAlly.set_val(data['ptAlly'])
    f.close()
    buttonl.label.set_text ("Load settings")
    print ('Parameters loaded from file '+fName)
    print ('Redrawing image Affine with loaded parameters...')
    loading_settings = 0
    update(0)
buttonl.on_clicked(load_map_settings)
#------------------------------
#plt.subplot(1,2,2)
#img2Object = plt.imshow(imgR, aspect='equal', cmap='jet')

# Draw interface
print('Start interface (few seconds)...')

pt1x_axe = plt.axes([0.15, 0.01, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
pt1y_axe = plt.axes([0.15, 0.05, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
pt2x_axe = plt.axes([0.15, 0.09, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
pt2y_axe = plt.axes([0.15, 0.13, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
pt3x_axe = plt.axes([0.15, 0.17, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
pt3y_axe = plt.axes([0.15, 0.21, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
ptAllx_axe = plt.axes([0.15, 0.25, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height
ptAlly_axe = plt.axes([0.15, 0.29, 0.7, 0.025], axisbg=axcolor) #stepX stepY width height

spt1x = Slider(pt1x_axe, 'pt1x', -200.0, 200.0, valinit=0)
spt1y = Slider(pt1y_axe, 'pt1y', -200.0, 200.0, valinit=0)
spt2x = Slider(pt2x_axe, 'pt2x', -200.0, 200.0, valinit=0)
spt2y = Slider(pt2y_axe, 'pt2y', -200.0, 200.0, valinit=0)
spt3x = Slider(pt3x_axe, 'pt3x', -200.0, 200.0, valinit=0)
spt3y = Slider(pt3y_axe, 'pt3y', -200.0, 200.0, valinit=0)
sptAllx = Slider(ptAllx_axe, 'ptAllx', -200.0, 200.0, valinit=0)
sptAlly = Slider(ptAlly_axe, 'ptAlly', -200.0, 200.0, valinit=0)

# Update depth map parameters and redraw
def update(val):
    global pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly,loading_settings,evx,evy
    pt1x = spt1x.val
    pt1y = spt1y.val
    pt2x = spt2x.val
    pt2y = spt2y.val
    pt3x = spt3x.val
    pt3y = spt3y.val
    ptAllx = sptAllx.val
    ptAlly = sptAlly.val
    if ( loading_settings==0 ):
        #print ('Rebuilding Affine')
        #disparity = stereo_depth_map(rectified_pair)
        #print(imgR.shape)
        rows,cols,_rgb = imgR.shape
        if val==2:
            img2 = cv2.warpAffine(imgR,cv2.getAffineTransform(
            np.float32([[0,0],[rows/2,cols/2],[rows,0]]),
            np.float32([[0,0],[evy,evx],[rows,0]]),
            #np.float32([[ptAlly+pt1y,ptAllx+pt1x],[ptAlly+pt2y,ptAllx+pt2x+cols],[ptAlly+pt3y+rows,ptAllx+pt3x]])
            ),(cols,rows))
        else:
            img2 = cv2.warpAffine(imgR,cv2.getAffineTransform(
            np.float32([[0,0],[0,cols],[rows,0]]),
            np.float32([[ptAlly+pt1y,ptAllx+pt1x],[ptAlly+pt2y,ptAllx+pt2x+cols],[ptAlly+pt3y+rows,ptAllx+pt3x]])
            ),(cols,rows))
        cv2.line(img2,(0,0),(cols,rows),(0,255,0),1)
        cv2.line(img2,(0,rows),(cols,0),(0,255,0),1)
        img2Object.set_data(img2)
        #cv2.imshow("in2", img2)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #print ('Redraw Affine')
        fig1.canvas.draw()


# Connect update actions to control elements
spt1x.on_changed(update)
spt1y.on_changed(update)
spt2x.on_changed(update)
spt2y.on_changed(update)
spt3x.on_changed(update)
spt3y.on_changed(update)
sptAllx.on_changed(update)
sptAlly.on_changed(update)

print('Show interface')
plt.show()
