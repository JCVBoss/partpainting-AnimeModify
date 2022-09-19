import torch
import os
import random
import cv2
import numpy as np
import torch.utils.data as data
import time
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

'''
 file tree:
 ---data---
    |---video---
    |   |---video1---
    |   |---video2---
    |---flow---
    |   |---flow---
    |   |   |---video1---
    |   |   |---video2---
    |   |---flow_r---
    |   |   |---video1---
    |   |   |---video2---
    |---mask---
    |   |---video1---
    |   |---video2---
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/video/')
    parser.add_argument('--flow_root', type=str, default='data/flow/')
    parser.add_argument('--IMAGE_SHAPE', type=list, default=[512, 512, 3])

    args = parser.parse_args()
    return args


class data_generater(data.Dataset):
    def __init__(self,dir_path,
                 target_size=(270,480,3),
                 seq_len=10,
                 isTest=False,
                 # mode='gen_mask'
                 mode='gen_mask'
                 ):
        self.dir_path = dir_path
        self.tar_size = target_size
        self.seq_len=seq_len
        self.isTest = isTest
        self.mode = mode
        # print(self.dir_path)
        self.video_path = os.path.join(self.dir_path,'video')

        self.video_list = [x for x in os.listdir(self.video_path)]
        self.video_list.sort()
        self.video_list = [os.path.join(self.video_path,x) for x in self.video_list]

    def __len__(self):
        # print(len(self.video_list))
        return len(self.video_list)
    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        video_name = os.path.basename(video_path)
        if self.isTest:
            random.seed(7777777)
        else:
            random.seed(time.time())

        video_data = videoloader(video_root=video_path,
                                 seq_len=self.seq_len,
                                 mask=False,
                                 target_size=self.tar_size,
                                 isTest=self.isTest,
                                 mode=self.mode
                                 )
        # print(len(video_data))
        frame_idx = random.randint(0,len(video_data)-1)
        data = video_data[frame_idx]

        return data

class videoloader(data.Dataset):
    def __init__(self,
                 video_root,
                 seq_len=10,
                 mask=False,
                 target_size=(270,480,3),
                 isTest=False,
                 mode='mask'
                 ):
        super(videoloader, self).__init__()
        self.video_name = os.path.basename(video_root)
        self.video_root = video_root
        self.dir_root = os.path.dirname(os.path.dirname(self.video_root))
        self.seq_len = seq_len
        self.isTest = isTest
        self.tar_size = target_size
        self.ori_size = None
        self.mode = mode

        self.video_path = os.path.join(self.dir_root,'video',self.video_name)
        self.flow_path = os.path.join(self.dir_root,'flow','flow',self.video_name)
        self.rflow_path = os.path.join(self.dir_root,'flow','flow_r',self.video_name)

        self.frames = [x for x in os.listdir(self.video_path)]
        self.flows = [x for x in os.listdir(self.flow_path)]
        self.rflows = [x for x in os.listdir(self.rflow_path)]
        self.frames.sort()
        self.flows.sort()
        self.rflows.sort()
        self.frames = [os.path.join(self.video_path,x) for x in self.frames]
        self.flows = [os.path.join(self.flow_path, x) for x in self.flows]
        self.rflows = [os.path.join(self.rflow_path, x) for x in self.rflows]


        self.video_len = len(self.frames)
        self.sample_len = self.video_len


        if mask:
            self.mask_path = os.path.join(self.video_root,mask)
        else:
            self.mask_path = 'mask.png'

        # if isTest:
        #     random.seed([777777])
        #     self.pointer = random.randint(seq_len//2,self.video_len-seq_len//2)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # if self.isTest:
        #     video = self.read_video(self.video_path)
        #     return video

        # frame_range = range(item,item+self.seq_len+1)
        # flow_range = range(item,item+self.seq_len)
        frame_range = range(0,self.video_len)
            # flow_range = range(0,self.video_len-1)
        # print(frame_range,flow_range)
        frames = self.read_frame_sequence(frame_range)
            # flows = self.read_flow_sequence(self.flows,flow_range)
            # rflows = self.read_flow_sequence(self.rflows,flow_range)
        mask = self.read_mask()
        ori_shape = frames.shape

        # mask = np.ones_like(mask)
        # mask[:,:,40:-40] = 0.0
        # create mask 4:3 [270,360], 16:9 [270,480], 21:9 [270,630]
        if self.mode == "gen_mask":
            c,h,w = frames.shape
            w_e = int(w*1.4)
            pad_w = (w_e-w)//2
            mask = np.zeros([1,h,w_e],dtype=np.float32)
            mask[:,:,(pad_w):(w+pad_w)] = 1.0
            frames_ = np.zeros([c,h,w_e],dtype=np.float32)
            frames_[:, :, pad_w:(w + pad_w)] = frames

            # c_, h_, w_ = flows.shape
            # w_e_ = int(w_ * 1.4)
            # pad_w_ = (w_e_ - w_) // 2
            # flows_ = np.zeros([flows.shape[-3], flows.shape[-2], w_e_], dtype=np.float32)
            # rflows_ = np.zeros([rflows.shape[-3], rflows.shape[-2], w_e_], dtype=np.float32)
            # # print(flows.shape, frames.shape, frames_.shape, flows_.shape)
            #
            # flows_[:, :, pad_w_:(w_+pad_w_)] = flows
            # rflows_[:, :, pad_w_:(w_+pad_w_)] = rflows
                # frames_[:,:,pad_w:(w+pad_w)] = torch.tensor(frames)
                # flows_[:,:,pad_w:(w+pad_w)] = torch.tensor(flows)
                # rflows_[:,:,pad_w:(w+pad_w)] = torch.tensor(rflows)
            # mask = torch.nn.functional.avg_pool2d(mask,kernel_size=7,stride=1,padding=3)
            # mask = (mask == 1.0)*1.0
            # frames_ = frames_ * mask
            # flows_ = flows_ * mask
            # rflows_ = rflows_ * mask

            frames = frames_
                # flows = flows_
                # rflows = rflows_

            mask = (mask < 0.5)*1.0
            # frames_ = []
            # for i in range(len(frames)):
            #     frame_t = torch.zeros([b, c, h, w_e], dtype=torch.float32)
            #     frame_t[:,:,:,pad_w:(w+pad_w)] = frames[i]
            #     frames_.append(frame_t)
            #     flow_t = torch.zeros([b, c, h, w_e], dtype=torch.float32)
            #     frame_t[:, :, :, pad_w:(w + pad_w)] = frames[i]
            #     frames_.append(frame_t)
        tar_shape = frames.shape[-2:]

        # print("dataloader",frames.shape)
        self.ori_size = frames.shape[-2:]
        frames = self.pre_frames(frames)
            # flows,f_range = self.pre_flows(flows)
            # rflows,r_range = self.pre_flows(rflows)
        mask = self.pre_mask(mask)
        # print(mask.dtype,torch.max(mask))
        res = {
            'name':self.video_name,
            'index':[x for x in frame_range],
            'frames':frames,
                # 'flows':flows,
                # 'rflows':rflows,
            'mask':mask,
                # 'flow_range': [f_range,r_range],
            'ori_shape': ori_shape,
            'tar_shape':tar_shape,
        }
        # print(res['name'],res['frames'].shape,res['flows'].shape,res['ori_shape'],res['tar_shape'],)

        return res



    def pre_frames(self,frames):
        h,w,c = frames.shape
        # print('2222222222222222', c, h, w, frames.shape)
        # frames = torch.zeros([3,270,480],dtype=torch.float32)
        frames = torch.Tensor(frames).unsqueeze(0)#.float32()

        if h!=self.tar_size[0] or w!=self.tar_size[1]:
            # try:
            #     print('11111111111111111111', frames.size(),
            #         self.video_name)
            #     print(frames.dim(),self.tar_size[:2])
            #     torch.

                frames = torch.nn.functional.interpolate(frames,
                                                         [self.tar_size[0],self.tar_size[1]],
                                                         # mode='nearest')
                                                         mode = 'bilinear',align_corners=True)
                                                         # mode='bilinear',align_corners=True)
            # frames = frames.resize((self.tar_size[0],self.tar_size[1]),)
            # frames = np.resize(frames,(c,self.tar_size[0],self.tar_size[1]))
            # except ValueError:

        # print(frames.shape)
        frames = frames/255.0
        frames = torch.clamp(frames,min=0.0,max=1.0).squeeze()
        return frames

    def pre_mask(self,mask):
        c,h,w = mask.shape
        # print(c,h,w)
        mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)

        if h!=self.tar_size[0] or w!=self.tar_size[1]:
            mask = torch.nn.functional.interpolate(mask,[self.tar_size[0],self.tar_size[1]],
                                                   mode='bilinear',align_corners=True)
            # mask = np.resize(mask,(c,self.tar_size[0],self.tar_size[1]))
        if torch.max(mask)>10.0:
            mask = mask/255.0
        mask = torch.where(mask<0.1,torch.zeros_like(mask),mask)
        return mask.squeeze(0)

    # def flow_normal(self,flows):
    #     c, h, w = flows.shape
    #
    #     flows = torch.tensor(flows)
    #     shift_w = (w - int(w * 0.75)) // 2
    #     flows = torch.reshape(flows, [-1, 2, h, w])
    #     flows_clip = flows[:, :, :, shift_w:(w - shift_w)].reshape([c // 2, 2, -1])
    #     val_min = torch.min(flows_clip, dim=-1).values.unsqueeze(dim=-1)  # [c,1]
    #     val_max = torch.max(flows_clip, dim=-1).values.unsqueeze(dim=-1)  # [c,1]
    #     # print(val_max.shape)
    #     flows = flows.reshape([c // 2, 2, -1])
    #     flows = (flows - val_min) / (val_max - val_min) * 2.0 - 1.0
    #     flows = flows.reshape([c,h,w])
    #     return flows


    def pre_flows(self,flows):
        c,h,w = flows.shape

        flows = torch.Tensor(flows)#.float32()
        shift_w = (w - int(w * 0.5))//2

        if h!=self.tar_size[0] or w!=self.tar_size[1]:
            flows = torch.reshape(flows,[-1,2,h,w])
            flows[:,0,:,:] = flows[:,0,:,:]/(w/1000)
            flows[:,1,:,:] = flows[:,1,:,:]/(h/1000)
            flows_clip = flows[:, :, :, shift_w:(w - shift_w)].reshape([c//2,2,-1])
            val_min = torch.min(flows_clip,dim=-1).values.unsqueeze(dim=-1)# [c,2,1]
            val_max = torch.max(flows_clip,dim=-1).values.unsqueeze(dim=-1)# [c,2,1]
            # val_min = torch.ones_like(val_min) * (-0.073)
            # val_max = torch.ones_like(val_max) * (0.073)
            # print(val_max.shape)
            flows = flows.reshape([c//2,2,-1])
            flows = (flows - val_min)/(val_max - val_min) * 2.0 - 1.0
            flows = flows.reshape([c // 2, 2, h, w])
            flows = torch.nn.functional.interpolate(flows,
                                                    [self.tar_size[0],self.tar_size[1]],
                                                    mode='bilinear',align_corners=True)
            # flows[:, 0, :, :] = flows[:, 0, :, :] * self.tar_size[1]
            # flows[:, 1, :, :] = flows[:, 1, :, :] * self.tar_size[0]
            flows = torch.reshape(flows,[c,self.tar_size[0],self.tar_size[1]])
            val_min = val_min.reshape([c,1])
            val_max = val_max.reshape([c, 1])
        # flows = (flows + 1.0)/2.0
        flow_range = torch.cat([val_min,val_max],dim=-1)
        return flows,flow_range

    def post_flows(self,flows):
        b, c, h, w = flows.shape
        # flows = flows * 2.0 - 1.0
        if not self.ori_size:

            flows = flows.reshape([b, c // 2, 2, h, w])
            flows[:, :, 0, :, :] = flows[:, :, 0, :, :] * w
            flows[:, :, 1, :, :] = flows[:, :, 1, :, :] * h
            flows = flows.reshape([b, c, h, w])
            return flows

        if [h,w] != self.ori_size :
            flows = torch.nn.functional.interpolate(flows,
                                                    [self.ori_size[0], self.ori_size[1]],
                                                    mode='bilinear',align_corners=True)
            flows = flows.reshape([b,c//2,2,h,w])
            flows[:,:,0,:,:] = flows[:,:,0,:,:] * self.ori_size[1]
            flows[:, :, 1, :, :] = flows[:, :, 1, :, :] * self.ori_size[0]
            flows = flows.reshape([b,c,self.ori_size[0],self.ori_size[1]])
            return flows

    def read_mask(self):
        # print(self.mask_path)
        mask = cv2.imread(self.mask_path,cv2.IMREAD_COLOR)
        mask = mask.swapaxes(2,1).swapaxes(1,0)
        return mask
    def read_frame_sequence(self,frame_range):
        frames = []
        for i in frame_range:
            frame_path = self.frames[i]
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # frame = frame.swapaxes(1,2).swapaxes(0,1)
            frames.append(frame)
        frames = np.concatenate(frames,axis=-1).swapaxes(1,2).swapaxes(0,1)
        return frames

    def read_flow_sequence(self,flow_l,flow_range):
        flows = []
        for i in flow_range:
            flow_path = flow_l[i]
            flow = cv2.readOpticalFlow(flow_path)
            flows.append(flow)
        flows = np.concatenate(flows,axis=-1).swapaxes(1,2).swapaxes(0,1)
        return flows

    def read_video(self,video_dir):
        video_ = video_dir
        video_name = os.path.basename(video_)
        frames = []

        if video_.endswith('.mp4'):
            pass
        else:
            video_sequence = [x for x in os.listdir(video_) if x.endswith('.png')]
            video_sequence.sort()
            for frame_name in video_sequence:
                frame = cv2.imread(os.path.join(video_, frame_name), cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        return {'video': frames, 'name': video_name}