
ply_header = '''ply
format ascii 1.0
comment author: MakcM
comment object: cloudpoint
element vertex %(vert_tri)d
property float x
property float y
property float z
property uchar red                   { start of vertex color }
property uchar green
property uchar blue
element face %(vert_num)d
property list uchar int vertex_index  { number of vertices for each face }
element edge 1                        { five edges in object }
property int vertex                  { index to first vertex of edge }
property uchar red                    { start of edge color }
property uchar green
property uchar blue
end_header
'''
ply_bottom = '''0 1 255 255 255                   { start of edge list, begin with white edge }'''

def write_plyTri(fn, verts):
    #print("Write ply start...")
    #verts = verts.reshape(-1, 3)
    #colors = colors.reshape(-1, 3)
    #verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=3*len(verts),vert_tri=9*len(verts))).encode('utf-8'))
        delt=0.05
        for i in range(len(verts)):
            c1 = verts[i,3]#((verts[i,3]) % 255)
            c2 = verts[i,4]
            c3 = verts[i,5]
            #print(verts[i])
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1],verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0]+delt,verts[i,1],verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1],verts[i,2]+delt,c1,c2,c3)).encode('utf-8'))

            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1],verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1]+delt,verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1],verts[i,2]+delt,c1,c2,c3)).encode('utf-8'))

            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1],verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0]+delt,verts[i,1],verts[i,2],c1,c2,c3)).encode('utf-8'))
            f.write(('%f %f %f %d %d %d\n' % (verts[i,0],verts[i,1]+delt,verts[i,2],c1,c2,c3)).encode('utf-8'))
        #np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        for i in range(len(verts)):
            f.write(('%d %d %d %d\n' % (3,i*9+0,i*9+1,i*9+2)).encode('utf-8'))
            f.write(('%d %d %d %d\n' % (3,i*9+3,i*9+4,i*9+5)).encode('utf-8'))
            f.write(('%d %d %d %d\n' % (3,i*9+6,i*9+7,i*9+8)).encode('utf-8'))
        f.write((ply_bottom).encode('utf-8'))
    #print("Write ply done")

if __name__ == '__main__':
    print("For usage, read source code..")
