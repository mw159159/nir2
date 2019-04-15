#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import json
from math import cos,sin,radians
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly,ptAllRp = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
def load_affine_settings():
    global pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly,ptAllRp
    fName = 'cam_set.txt'
    f=open(fName, 'r')
    data = json.load(f)
    pt1x = data['pt1x']
    pt1y = data['pt2y']
    pt2x = data['pt2x']
    pt2y = data['pt2y']
    pt3x = data['pt3x']
    pt3y = data['pt3y']
    ptAllx = data['ptAllx']
    ptAlly = data['ptAlly']
    ptAllRp = data['ptAllRp']
    f.close()
    return
def calc_affine(cols,rows):
    global pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly,ptAllRp
    x1=0.0
    y1=0.0
    x2=float(cols)
    y2=0.0
    x3=0.0
    y3=float(rows)
    x0=float((x2+x3)/2)
    y0=float((y3+y2)/2)
    #print((x1-x0)*cos(radians(ptAllRp)))
    #print((y1-y0)*sin(radians(ptAllRp)))
    nx1=(x1-x0)*cos(radians(ptAllRp))-(y1-y0)*sin(radians(ptAllRp))+x0
    ny1=(x1-x0)*sin(radians(ptAllRp))+(y1-y0)*cos(radians(ptAllRp))+y0
    nx2=(x2-x0)*cos(radians(ptAllRp))-(y2-y0)*sin(radians(ptAllRp))+x0
    ny2=(x2-x0)*sin(radians(ptAllRp))+(y2-y0)*cos(radians(ptAllRp))+y0
    nx3=(x3-x0)*cos(radians(ptAllRp))-(y3-y0)*sin(radians(ptAllRp))+x0
    ny3=(x3-x0)*sin(radians(ptAllRp))+(y3-y0)*cos(radians(ptAllRp))+y0
    #print(cols,rows)
    print(x1,y1,x2,y2,x3,y3, cos(radians(ptAllRp)), sin(radians(ptAllRp)),x0,y0)
    x1=nx1+pt1x+ptAllx
    y1=ny1+pt1y+ptAlly
    x2=nx2+pt2x+ptAllx
    y2=ny2+pt2y+ptAlly
    x3=nx3+pt3x+ptAllx
    y3=ny3+pt3y+ptAlly
    return x1,y1,x2,y2,x3,y3

if __name__ == '__main__':
    print('loading images...')
    #imgL = cv2.pyrDown( cv2.imread('../data/aloeL.jpg') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('../data/aloeR.jpg') )
    imgL = cv2.pyrDown(cv2.imread('imgL2.jpg'))
    imgR = cv2.pyrDown(cv2.imread('imgR2.jpg'))
    #cap = cv2.VideoCapture(2)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap2 = cv2.VideoCapture(3)
    #cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #ret, img = cap.read()
    #ret2, img2 = cap2.read()
    #imgL = cv2.pyrDown(img)#cv2.imread('stul/1r.JPG') )  # downscale images for faster processing
    #imgR = cv2.pyrDown(img2) #cv2.imread('stul/1l.JPG') )
    load_affine_settings()
    rows,cols,_rgb = imgR.shape
    #print(rows)
    x1,y1,x2,y2,x3,y3 = calc_affine(cols,rows)
    imgR = cv2.warpAffine(imgR,cv2.getAffineTransform(
    np.float32([[0,0],[cols,0],[0,rows]]),
    np.float32([[x1,y1],[x2,y2],[x3,y3]])
    ),(cols,rows))
    #cap.release()
    #cap2.release()
    #exit(0)
    #img3 = cv2.warpAffine(imgR,cv2.getAffineTransform(
    #np.float32([[0,0],[0,1920],[1080,1920]]),
    #np.float32([[0,0],[h1,w1],[h1,0]]),
    #),(cols,rows))

    fig,ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.02,right=0.98,top=0.98)
    img2Object = plt.imshow(imgR)
    plt.axis('off')
    axcolor = 'lightgoldenrodyellow'
    #axfreq = plt.axes([0.25, 0.1, 0.65, 0.02], facecolor=axcolor)#stepX stepY width height
    #axamp = plt.axes([0.25, 0.125, 0.65, 0.02], facecolor=axcolor)

    saveax = plt.axes([0.3, 0.25, 0.15, 0.04]) #stepX stepY width height
    buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
    #buttons.on_clicked(save_map_settings)
    loadax = plt.axes([0.5, 0.25, 0.15, 0.04]) #stepX stepY width height
    buttonl = Button(loadax, 'Load', color=axcolor, hovercolor='0.975')
    #buttonl.on_clicked(load_map_settings)

    # Depth map function
    SWS = 5
    PFS = 5
    PFC = 29
    MDS = -25
    NOD = 128
    TTH = 100
    UR = 10
    SR = 15
    SPWS = 100

    def stereo_depth_map(iimgL,iimgR):
        c, r = iimgL.shape
        disparity = np.zeros((c, r), np.uint8)
        sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        #sbm.SADWindowSize = SWS
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(PFS)
        sbm.setPreFilterCap(PFC)
        sbm.setMinDisparity(MDS)
        sbm.setNumDisparities(NOD)
        sbm.setTextureThreshold(TTH)
        sbm.setUniquenessRatio(UR)
        sbm.setSpeckleRange(SR)
        sbm.setSpeckleWindowSize(SPWS)
        disparity = sbm.compute(iimgL, iimgR)
        local_max = disparity.max()
        local_min = disparity.min()
        print ("MAX " + str(local_max))
        print ("MIN " + str(local_min))
        disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
        local_max = disparity_visual.max()
        local_min = disparity_visual.min()
        print ("MAX " + str(local_max))
        print ("MIN " + str(local_min))
        return disparity_visual

    disparity = stereo_depth_map(rectified_pair)

    # Draw interface for adjusting parameters
    print('Start interface creation (it takes up to 30 seconds)...')

    SWSaxe = plt.axes([0.15, 0.01, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    PFSaxe = plt.axes([0.15, 0.05, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    PFCaxe = plt.axes([0.15, 0.09, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    MDSaxe = plt.axes([0.15, 0.13, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    NODaxe = plt.axes([0.15, 0.17, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    TTHaxe = plt.axes([0.15, 0.21, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    URaxe = plt.axes([0.15, 0.25, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    SRaxe = plt.axes([0.15, 0.29, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height
    SPWSaxe = plt.axes([0.15, 0.33, 0.21, 0.02], axisbg=axcolor) #stepX stepY width height

    sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
    sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
    sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
    sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
    sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
    sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
    sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
    sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
    sSPWS = Slider(SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)

    # Update depth map parameters and redraw
    def update(val):
        global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
        SWS = int(sSWS.val/2)*2+1 #convert to ODD
        #print(SWS,sSWS.val)
        #exit(0)
        PFS = int(sPFS.val/2)*2+1
        PFC = int(sPFC.val/2)*2+1
        MDS = int(sMDS.val)
        NOD = int(sNOD.val/16)*16
        TTH = int(sTTH.val)
        UR = int(sUR.val)
        SR = int(sSR.val)
        SPWS= int(sSPWS.val)
        if ( loading_settings==0 ):
            print ('Rebuilding depth map')
            disparity = stereo_depth_map(rectified_pair)
            img2Object.set_data(disparity)
            print ('Redraw depth map')
            plt.draw()

    # Connect update actions to control elements
    sSWS.on_changed(update)
    sPFS.on_changed(update)
    sPFC.on_changed(update)
    sMDS.on_changed(update)
    sNOD.on_changed(update)
    sTTH.on_changed(update)
    sUR.on_changed(update)
    sSR.on_changed(update)
    sSPWS.on_changed(update)

    plt.show()
    exit(0)

        p1_axe = plt.axes([0.15, 0.01+0.4, 0.7, 0.025], axisbg=axcolor)
        p2_axe = plt.axes([0.15, 0.05+0.4, 0.7, 0.025], axisbg=axcolor)
        p3_axe = plt.axes([0.15, 0.09+0.4, 0.7, 0.025], axisbg=axcolor)
        p4_axe = plt.axes([0.15, 0.13+0.4, 0.7, 0.025], axisbg=axcolor)
        p5_axe = plt.axes([0.15, 0.17+0.4, 0.7, 0.025], axisbg=axcolor)
        p6_axe = plt.axes([0.15, 0.21+0.4, 0.7, 0.025], axisbg=axcolor)
        p7_axe = plt.axes([0.15, 0.25+0.4, 0.7, 0.025], axisbg=axcolor)

        sp1 = Slider(p1_axe, 'p1', -200.0, 200.0, valinit=0)
        sp2 = Slider(p2_axe, 'p2', -200.0, 200.0, valinit=0)
        sp3 = Slider(p3_axe, 'p3', -200.0, 200.0, valinit=0)
        sp4 = Slider(p4_axe, 'p4', -200.0, 200.0, valinit=0)
        sp5 = Slider(p5_axe, 'p5', -200.0, 200.0, valinit=0)
        sp6 = Slider(p6_axe, 'p6', -200.0, 200.0, valinit=0)
        sp7 = Slider(p7_axe, 'p7', -200.0, 200.0, valinit=0)

        # Update depth map parameters and redraw
        def update(val):
            global pt1x,pt1y,pt2x,pt2y,pt3x,pt3y,ptAllx,ptAlly,ptAllRp,ptAllRn,imgR,imgL
            p1 = sp.val
            p2 = spt1y.val
            p3 = spt2x.val
            p4 = spt2y.val
            p5 = spt3x.val
            p6 = spt3y.val
            p7 = sptAllx.val

            disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

            img2Object.set_data(img2wl)
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
    sptAllRp.on_changed(update)
    sptAllRn.on_changed(update)

    plt.show()
    exit(0)
    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    #points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    #out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    #write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    fig,ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.02,right=0.98,top=0.98)
    img2Object = plt.imshow(imgR)
    plt.axis('off')
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.02], facecolor=axcolor)#stepX stepY width height
    axamp = plt.axes([0.25, 0.125, 0.65, 0.02], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=0)

    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()
        sfreq.on_changed(update)
        samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        sfreq.reset()
        samp.reset()
    button.on_clicked(reset)

    plt.show()
    exit(0)
