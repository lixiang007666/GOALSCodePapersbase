import os
import imageio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.graph import shortest_path, route_through_array

foldername = '/unet_inceptionv4_pseudo3_ep159_tta_gammahflip'
img_folder = 'D:/GOALS2022-Validation/Validation/Image'
# result_folder = 'D:/PyDL/OCTseg-GOALS/output_seg/deeplab_baseline512_best'
# output_folder = 'D:/PyDL/OCTseg-GOALS/output_seg/final/deeplab_baseline512_best_post'
result_folder = 'D:/PyDL/segmentation_models.pytorch-master/output/' + foldername
output_folder = 'D:/PyDL/OCTseg-GOALS/output_seg/final/' + foldername + '_post'
vis_folder = 'D:/PyDL/OCTseg-GOALS/output_seg/final/vis'
vis_folder2 = 'D:/PyDL/OCTseg-GOALS/output_seg/final/vis2'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def find_line(p1,p2,mask,visited):
    line1 = p2 - p1
    line1_prob = line1[1:,:] -line1[0:-1,:]
    line1_prob[line1_prob<0] = 0
    # cost scaling
    line1_invprob = (-np.power(line1_prob,0.5))
    line1_invprob = (line1_invprob - line1_invprob.min()) / (line1_invprob.max()-line1_invprob.min()) * 100
    block = visited[1:,:].copy()
    line1_invprob[block>0] = 1000
    # line1_invprob = np.exp(line1_invprob) * 10
    line1_invprob = np.hstack([np.zeros((line1_invprob.shape[0],1)),line1_invprob])
    line1_invprob = np.hstack([line1_invprob,np.zeros((line1_invprob.shape[0],1))])
    # p, cost = shortest_path(firstline_invprob,reach=1,axis=-1,output_indexlist=True)
    p, cost = route_through_array(line1_invprob,start=[0,0],end=[line1_invprob.shape[0]-1,line1_invprob.shape[1]-1],fully_connected=True)
    l1_path = np.zeros(mask.shape)
    path_coor = []
    for pt in p:
        if pt[1] != 0 and pt[1] != line1_invprob.shape[1]-1:
            x = pt[0] 
            y = pt[1] - 1
            l1_path[x,y] = 1
            path_coor.append(x)
    path_coor = np.asarray(path_coor)

    # fig, axes = plt.subplots(1,3)
    # axes[0].imshow(mask,cmap='jet')
    # axes[1].imshow(line1_invprob)
    # axes[2].imshow(l1_path)
    # plt.show()
    # plt.close()
    return l1_path, path_coor

def mark_visited(visited,path):
    last = 0
    for j in range(visited.shape[1]):
        upper = np.argwhere(path[:,j] == 1)
        if upper.size > 0:
            upper = np.max(upper)
            last = upper
        else:
            upper = last
        visited[0:upper+2,j] = 1
    return visited

for imgfile in os.listdir(result_folder)[0:]:
    # if not '0120.png' in imgfile:  #0113 0114
    #     continue
    ext = imgfile.split('.')[-1]
    if ext == 'png':
        print(imgfile)
        img = imageio.imread(img_folder + '/' + imgfile)
        mask = imageio.imread(result_folder + '/' + imgfile)
        prob = np.load(result_folder + '/' + imgfile.replace('.png','.npy'))

        mask = resize(mask.astype(np.float32),output_shape=(800,1100),preserve_range=True,order=1)
        prob = resize(prob.astype(np.float32),output_shape=(6,800,1100),preserve_range=True,order=1)
        mask = np.round(mask)
        # mask[mask<0] = 0
        # plt.imshow(mask,cmap='jet')
        # plt.show()
        # plt.close()

        old_mask = mask.copy()

        visited = np.zeros(mask.shape)
        l1_path, l1_pt = find_line(prob[0,:,:],prob[1,:,:],mask,visited)
        visited = mark_visited(visited,l1_path)
        
        l2_path, l2_pt = find_line(prob[1,:,:], prob[2,:,:],mask,visited)
        visited = mark_visited(visited,l2_path)

        l3_path, l3_pt = find_line(prob[2,:,:], prob[3,:,:],mask,visited)
        visited = mark_visited(visited,l3_path)

        l4_path, l4_pt = find_line(prob[3,:,:], prob[4,:,:],mask,visited)
        visited = mark_visited(visited,l4_path)

        l5_path, l5_pt = find_line(prob[4,:,:], prob[5,:,:],mask,visited)

        all_path = np.zeros(mask.shape)
        all_path[l1_path>0] = 1
        all_path[l2_path>0] = 2
        all_path[l3_path>0] = 3
        all_path[l4_path>0] = 4
        all_path[l5_path>0] = 5

        # line_pts = np.stack([l1_pt,l2_pt,l3_pt,l4_pt,l5_pt],axis=0)

        mask_post = np.ones(mask.shape) * 5
        for j in range(mask_post.shape[1]):
            for k in range(0,5):
                if k == 0:
                    upper = 0
                else:
                    upper = np.argwhere(all_path[:,j] == k)
                    if not upper.size > 0:
                        print('Error',j,k)
                    else:
                        upper = np.max(upper)

                lower = np.argwhere(all_path[:,j] == k+1)
                if not lower.size > 0:
                        print('Error',j,k+1)
                else:
                    lower = np.max(lower) 
                
                if lower >= upper:
                    mask_post[upper+1:lower+1,j] = k
        
        mask_post[0,:] = 0
        # print(np.max(mask_post-mask))
        # print(np.argwhere((mask_post-mask)!=0))
        # plt.figure(figsize=(20,20))
        # fig, axes = plt.subplots(2,2)
        # axes[0,0].imshow(mask,cmap='jet')
        # axes[0,0].axis('off')
        # axes[0,1].imshow(img,cmap='gray')
        # alphas = np.zeros(all_path.shape)
        # alphas[all_path>0] = 1
        # axes[0,1].imshow(all_path,cmap='jet',alpha=alphas)
        # axes[0,1].axis('off')
        # axes[1,0].imshow(mask_post,cmap='jet')
        # axes[1,0].axis('off')
        # axes[1,1].imshow(mask_post-mask,vmin=-1,vmax=1)
        # axes[1,1].axis('off')
        # plt.tight_layout()
        # # plt.savefig(vis_folder + '/' + imgfile)
        # plt.show()
        # plt.close()

        plt.figure(figsize=(10,8))
        plt.imshow(img,cmap='gray')
        alphas = np.zeros(all_path.shape)
        alphas[all_path>0] = 1
        plt.imshow(all_path,cmap='jet',alpha=alphas)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(vis_folder2 + '/' + imgfile)
        plt.close()
        mask = mask_post.copy()

        mask = np.round(mask).astype(np.uint8)
        new_mask = np.ones(mask.shape) * 255
        new_mask[mask==1] = 0
        new_mask[mask==2] = 80
        new_mask[mask==4] = 160
        new_mask = new_mask.astype(np.uint8)

        imageio.imsave(output_folder + '/' + imgfile.split('.')[0] + '.png',new_mask)
