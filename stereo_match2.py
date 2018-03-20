#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys
from math import acos,atan,sqrt,pow,sin,cos,degrees,radians

def toSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,0], xyz[:,1])
    return ptsnew
def toSpherical_np2(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,6] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,7] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,8] = np.arctan2(xyz[:,0], xyz[:,1])
    return ptsnew

NaN = float('nan')
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

def quantImg(im, n=2):
    #n = 2    # Number of levels of quantization
    indices = np.arange(0,256)   # List of all colors
    divider = np.linspace(0,255,n+1)[1] # we get a divider
    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
    palette = quantiz[color_levels] # Creating the palette
    im2 = palette[im]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8
    return im2

if __name__ == '__main__':
    print('loading images...')
    #imgL = cv2.pyrDown( cv2.imread('../data/aloeL.jpg') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('../data/aloeR.jpg') )
    #imgL = cv2.pyrDown( cv2.imread(sys.argv[1]) )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread(sys.argv[2]) )
    imgL = cv2.imread(sys.argv[1])
    imgR = cv2.imread(sys.argv[2])
    #imgL = cv2.pyrDown( cv2.imread('stul/1r.JPG') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('stul/1l.JPG') )

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = int(sys.argv[4]),
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    #print(disp[1000][500],imgR[1000][500])
    #exit(0)
    cv2.imwrite(sys.argv[3], disp*2)
    h, w = disp.shape[:2]
    open('out3','w').write('')
    #mydisp_spher = []
    for x in range(-int(w/2),int(w/2), 5):
        for y in range(-int(h/2),int(h/2), 5):
            try:
                z = disp[y+int(h/2)][x+int(w/2)]
            except IndexError:
                print (w,h,x,y)
            if z>=16 :
                try:
                    #print (w,h,x,y,z)
                    x2 = radians(x*(44/2)/int(w/2))
                    ##y2 = radians(y*(44/2)/int(w/2))
                    ##z2 = 2+(10-2)*((111-z)/(111-16)) #0.0074*z+1.8818
                    ##ro = z2 #sqrt(pow(x2,2)+pow(y2,2)+pow(z2,2))
                    ##phi = x*(44/2)/int(w/2)#acos(z2/ro) #
                    #phi = degrees(acos(z2/ro))
                    ##psi = y*(44/2)/int(w/2)#atan(y2/x2) #
                    #if ro<5:
                    #    ro=ro + pow(ro,(5-ro))*0.13
                    #else:
                    #    ro=sqrt(pow(ro,2)+pow(x/280,2))
                    #    ro=sqrt(pow(ro,2)+pow(y/(280),2))
                    #if psi<0:
                    #    ro=0;phi=0;psi=0 #ro = ro*1.7
                    #psi = degrees(atan(y2/x2))
                    #ro = z2 - sin(radians(phi)) + sin(radians(psi))
                except (ZeroDivisionError,ValueError) as e:
                    import sys
                    print(x,y,z,ro,phi,psi,file=sys.stderr)
                    ro, phi, psi = NaN, NaN, NaN
                #mydisp_spher += [[ro,phi,psi]]
                #print(x,y,z,ro,phi,psi)
                #if -1<phi<1:
                #open('out3','a').write("%s %s %s %s\n" % (phi,psi,ro,ro/cos(radians(phi))))
            else:
                z=0
    print('generating 3d point cloud ...',)
    h, w = imgL.shape[:2]
    f = 0.6*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    #print(points.shape)
    #print(points[0,0])
#    imgR = quantImg(imgR, n=4)
#    imgR_Gray = cv2.cvtColor( imgR, cv2.COLOR_RGB2GRAY )
#    imgR_Gray[:,:] = np.ceil(imgR_Gray[:,:]/10)#rint
#    imgR_Gray[:,:] = imgR_Gray[:,:]*10
#    points2 = np.dstack((points,imgR_Gray[:,:]))
    #points2[:,:,3] = int(points2[:,:,3]/5)*5
#    points2 = np.dstack((points2,imgR[:,:,1:3]))
    imgR = cv2.cvtColor( imgR, cv2.COLOR_BGR2RGB )
    points2 = np.dstack((points,imgR[:,:,0:3]))
    #print(points2.shape)
    #print(points2[0,0])
    mask = disp > disp.min()
    #print(disp.shape)
    #print(disp[0,0])
    out_points = points2[mask]
    #print(out_points.shape)
    #print(out_points[0])
    #exit(0)
    #np.savetxt("out4", out_points, delimiter=" ")
    #points2 = toSpherical_np(out_points)
    #np.savetxt("out3", points2[::10], delimiter=" ")

    print('convert 3d to spherical...')
    #for i in out_points:
    #    if i del
    out_points[:,0]+=100#-1*out_points[:,0].min()
    out_points[:,1]+=100#-1*out_points[:,1].min()
    out_points[:,2]+=100#-1*out_points[:,2].min()
    print(out_points[:,0].min(),out_points[:,1].min(),out_points[:,2].min())
    print(out_points[:,0].max(),out_points[:,1].max(),out_points[:,2].max())
    points2 = toSpherical_np2(out_points)
    np.savetxt("out3", points2[::5,0:9], delimiter=" ")
    exit(0)
    #open('out3','a').write("%s\n%s" % (points[0],points2[0]))
    exit(0)
    #cv2.bilateralFilter(disp, disp_filt,)

    print('generating 3d point cloud 2...',)
    for i3 in range(imgL.shape[:1]):
        x3 = mydisp_spher[i3]
        print(x3,y3,z3,110,80,110,file='out2.ply')

    write_ply('out2.ply', out_points, out_colors)
    print('%s saved' % 'out2.ply')

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.6*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    #cv2.imshow('disparity', (disp-min_disp)/num_disp)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
