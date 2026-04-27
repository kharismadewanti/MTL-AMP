import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import cat
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchsummary import summary
from torchviz import make_dot
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from collections import OrderedDict
from torchvision.models import ResNet50_Weights
from torch.amp import autocast, GradScaler
 

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1, dropout_p=0.1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(
            channelx[0], channelx[1], kernel_size=kernelx,
            stride=stridex, padding=paddingx, padding_mode="zeros",
        )
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channel, final=False):
        super(ConvBlock, self).__init__()
        if final:
            # keep size, then project with 1x1
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Conv2d(channel[0], channel[1], kernel_size=1)
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)

    def forward(self, x):
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y


class resnet50mtlfp16(nn.Module):
    def __init__(self):
        super().__init__()
        n_fmap = [64, 256, 512, 1024, 2048]

        # RGB normalizer expects input in [0,1]
        self.rgb_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.RGB_encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.RGB_encoder.fc = nn.Sequential()
        self.RGB_encoder.avgpool = nn.Sequential()

        # SEG decoder
        self.conv3_ss_f = ConvBlock(channel=[n_fmap[4] + n_fmap[3], n_fmap[3]])
        self.conv2_ss_f = ConvBlock(channel=[n_fmap[3] + n_fmap[2], n_fmap[2]])
        self.conv1_ss_f = ConvBlock(channel=[n_fmap[2] + n_fmap[1], n_fmap[1]])
        self.conv0_ss_f = ConvBlock(channel=[n_fmap[1] + n_fmap[0], n_fmap[0]])
        self.final_ss_f = ConvBlock(channel=[n_fmap[0], 19], final=True)

        # DEPTH decoder
        self.conv3_dep_f = ConvBlock(channel=[n_fmap[4] + n_fmap[3], n_fmap[3]])
        self.conv2_dep_f = ConvBlock(channel=[n_fmap[3] + n_fmap[2], n_fmap[2]])
        self.conv1_dep_f = ConvBlock(channel=[n_fmap[2] + n_fmap[1], n_fmap[1]])
        self.conv0_dep_f = ConvBlock(channel=[n_fmap[1] + n_fmap[0], n_fmap[0]])
        self.final_dep_f = ConvBlock(channel=[n_fmap[0], 1], final=True)

        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb):
        # Expect rgb in [0,1]
        rgb = rgb.float()
        in_rgb = self.rgb_normalizer(rgb)
        x_RGB = self.RGB_encoder.conv1(in_rgb)
        x_RGB = self.RGB_encoder.bn1(x_RGB)
        feat0 = self.RGB_encoder.relu(x_RGB)               
        feat1 = self.RGB_encoder.layer1(self.RGB_encoder.maxpool(feat0))  
        feat2 = self.RGB_encoder.layer2(feat1)         
        feat3 = self.RGB_encoder.layer3(feat2)        
        feat4 = self.RGB_encoder.layer4(feat3)         
        up_feat4 = F.interpolate(feat4, size=feat3.size()[-2:], mode="bilinear", align_corners=True)

        # SEG path
        s3 = self.conv3_ss_f(cat([up_feat4, feat3], dim=1))
        s3u = F.interpolate(s3, size=feat2.shape[-2:], mode="bilinear", align_corners=True)
        s2 = self.conv2_ss_f(cat([s3u, feat2], dim=1))
        s2u = F.interpolate(s2, size=feat1.shape[-2:], mode="bilinear", align_corners=True)
        s1 = self.conv1_ss_f(cat([s2u, feat1], dim=1))
        s1u = F.interpolate(s1, size=feat0.shape[-2:], mode="bilinear", align_corners=True)
        s0 = self.conv0_ss_f(cat([s1u, feat0], dim=1))
        s0u = F.interpolate(s0, size=in_rgb.size()[-2:], mode="bilinear", align_corners=True)
        seg_f = self.final_ss_f(s0u)  # multi-label style; keep BCE

        # DEPTH path
        d3 = self.conv3_dep_f(cat([up_feat4, feat3], dim=1))
        d3u = F.interpolate(d3, size=feat2.size()[-2:], mode="bilinear", align_corners=True)
        d2 = self.conv2_dep_f(cat([d3u, feat2], dim=1))
        d2u = F.interpolate(d2, size=feat1.size()[-2:], mode="bilinear", align_corners=True)
        d1 = self.conv1_dep_f(cat([d2u, feat1], dim=1))
        d1u = F.interpolate(d1, size=feat0.size()[-2:], mode="bilinear", align_corners=True)
        d0 = self.conv0_dep_f(cat([d1u, feat0], dim=1))
        d0u = F.interpolate(d0, size=in_rgb.size()[-2:], mode="bilinear", align_corners=True)
        dep_f = self.relu(self.final_dep_f(d0u))

        return seg_f, dep_f


SEG_MAP = {
    'colors': [
        [128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],
        [152, 251, 152],[70, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],
        [0, 0, 70],[0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32],
    ],
    'classes': [
        'road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation',
        'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle'
    ],
}


def cls2one_hot(ss_gt19, n_class):
    one_hot = (np.arange(n_class) == ss_gt19[..., None]).astype(np.float32)
    return one_hot


def resize_matrix(image, target_size=(256, 512)):
    resized_image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return resized_image


class myData(Dataset):
    def __init__(self, root_dir, part):
        self.rgb = []
        self.seg = []
        self.dep = []

        rgb_folder = os.path.join(root_dir, 'rgb', part)
        seg_folder = os.path.join(root_dir, 'seg', part)
        dep_folder = os.path.join(root_dir, 'dep', part)

        town_list = os.listdir(rgb_folder)
        town_list.sort()
        for town in town_list:
            rgb_dir = os.path.join(rgb_folder, town)
            seg_dir = os.path.join(seg_folder, town)
            dep_dir = os.path.join(dep_folder, town)

            file_list = os.listdir(rgb_dir)
            file_list.sort()
            for filename in file_list:
                self.rgb.append(os.path.join(rgb_dir, filename))
                self.seg.append(os.path.join(seg_dir, filename[:-4] + "_map.png"))
                self.dep.append(os.path.join(dep_dir, filename[:-4] + ".png"))

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        data = dict()
        rgb = cv2.imread(self.rgb[index])
        rgb = resize_matrix(rgb, target_size=(256,512))
        rgb = rgb.astype(np.float32)
        data['rgb'] = np.transpose(rgb, (2, 0, 1))

        segmap = cv2.imread(self.seg[index])[:, :, 0]
        segmap = resize_matrix(segmap, target_size=(256, 512))
        onehotcls = cls2one_hot(segmap, n_class=len(SEG_MAP["classes"]))
        data['seg'] = np.transpose(onehotcls.astype(np.float32), (2, 0, 1))

        dep = cv2.imread(self.dep[index], cv2.IMREAD_GRAYSCALE)
        dep = resize_matrix(dep, target_size=(256,512))
        dep = dep.astype(np.float32) / 255.0
        data['dep'] = np.expand_dims(dep, axis=0)
        return data


def dice(pred, gt, smooth=1e-6):
    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    inter = (pred * gt).sum(1)
    union = pred.sum(1) + gt.sum(1)
    dice = (2 * inter + smooth) / (union + smooth)
    return (1 - dice).mean()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def vis_seg(seg, SEG_CMAP):
    imgx = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint32)
    for cmap in SEG_CMAP['colors']:
        cmap_id = SEG_CMAP['colors'].index(cmap)
        rows, cols = np.where(seg == cmap_id)
        imgx[rows, cols, :] = cmap
    return imgx


def lineardepth(normalized_depth):
    lindep = np.clip(normalized_depth, 0.0, 1.0) * 255  # normalisasi ke 0 - 255
    lindep = np.repeat(lindep[:, :, np.newaxis], 3, axis=2)
    return lindep

rootdir = "/media/xavier/E03D6057/kharisma/"
dir_rgb = rootdir+"rgb/"
dir_seg = rootdir+"seg/"
dir_dep = rootdir+"dep/"
part = 'test/' #train val test
town = 'around_mipa/' #pilih salah satu kota
model = resnet50mtlfp16()
save_dir = os.getcwd() +"/resnet50mtl_fp16_new/"

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

device = torch.device('cuda')
model = model.to(device, dtype=torch.float)

list_file = os.listdir(dir_rgb+part+town)
list_file.sort() #urutkan
idx = 874
filename = list_file[idx]
image = cv2.imread(dir_rgb+part+town+filename)
imagex = image[:, :, [2, 1, 0]]

list_file = os.listdir(dir_seg+part+town)
list_file.sort() #urutkan
idx = 874
idx = 2*idx + 1
filename = list_file[idx]
seg_gt = cv2.imread(dir_seg+part+town+filename, cv2.IMREAD_GRAYSCALE)

list_file = os.listdir(dir_dep+part+town)
list_file.sort() #urutkan
idx = 874
filename = list_file[idx]
dep_gt = (cv2.imread(dir_dep+part+town+filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)

plt.imshow(imagex)
plt.show()

seg_gtx = resize_matrix(seg_gt, target_size=(256, 512))
seg_color = vis_seg(seg_gtx, SEG_MAP)
plt.imshow(seg_color.astype(np.uint8))
plt.show()

dep_lin = lineardepth(dep_gt)
plt.imshow(dep_lin.astype(np.uint8))
plt.show()

model = resnet50mtlfp16() #resunet_mtl
model.load_state_dict(torch.load(save_dir+'best_model.pth'))
model.to(device, dtype=torch.float)
input_img = np.expand_dims(resize_matrix(image, target_size=(256, 512)).transpose(2,0,1), axis=0) #buat jadi NxCxHxW
input_img = torch.from_numpy(input_img).to(device, dtype=torch.float)

pred_seg, pred_dep = model(input_img)

pred_seg = np.argmax(pred_seg.cpu().detach().numpy()[0], axis=0) #ambil batch 0 dan lakukan argmax
pred_dep = pred_dep.cpu().detach().numpy()[0][0] #ambil batch 0 dan channel 0

pseg_color = vis_seg(pred_seg, SEG_MAP)
plt.imshow(pseg_color.astype(np.uint8))
plt.show()

pdep_log = lineardepth(pred_dep)
plt.imshow(pdep_log.astype(np.uint8))
plt.show()

def IOU(Yp, Yt):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    output = Yp.reshape(-1) > 0.5 #maksudnya yang lebih dari 0.5 adalah true
    target = Yt.reshape(-1) > 0.5 #dan yang kurang dari 0.5 adalah false
    intersection = (output & target).sum().float().to(device) #irisan
    union = (output | target).sum().float() #union
    #rumus IoU
    if union == 0:
        return torch.tensor(1.0, device=device)
    else:
        return intersection / union

def iou_multiclass(pred, target, num_classes):
    device = pred.device
    ious = []

    pred = pred.detach()
    target = target.detach()

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (target == cls)

        iou_c = IOU(pred_mask, gt_mask)   # sudah di CUDA
        ious.append(iou_c)

    # semua elemen pasti di CUDA → aman
    return torch.stack(ious).mean()

def test(data_loader, model, device):
    #buat variabel untuk menyimpan kalkulasi loss
    metric_tm = AverageMeter()
    metric_seg = AverageMeter()
    metric_dep = AverageMeter()
    
    #buat dictionary log untuk menyimpan training log di CSV
    logx = OrderedDict([
        ('batch', []),
        ('test_tm', []),
        ('test_seg', []),
        ('test_mae', []),
        ('elapsed_time', [])])

    #masuk ke mode eval torch
    model.eval()
    
    with torch.no_grad():
        #visualisasi progress training dengan tqdm
        prog_bar = tqdm()
        prog_bar.reset(total=len(data_loader))

        #training....
        for batch_ke, data in enumerate(data_loader):
            seg_gt = data['seg'].to(device, dtype=torch.float)
            dep_gt = data['dep'].to(device, dtype=torch.float)
            image = data['rgb'].to(device, dtype=torch.float)

            #forward pass
            start_time = time.time() #waktu mulai
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_seg, pred_dep = model(image)
            elapsed_time = time.time() - start_time #hitung elapsedtime
            
            pred_seg = torch.sigmoid(pred_seg)

            for i in range(pred_seg.shape[0]):

                rgb = image[i].cpu().numpy().transpose(1,2,0)
                rgb = rgb[:,:,::-1] / 255

                seg_pred = pred_seg[i].cpu().numpy()
                seg_pred = np.argmax(seg_pred, axis=0)
                seg_color = vis_seg(seg_pred, SEG_MAP)

                dep_np = pred_dep[i,0].cpu().numpy()
                dep_img = lineardepth(dep_np)

                plt.figure(figsize=(15,4))

                # RGB
                plt.subplot(1,3,1)
                plt.title("RGB")
                plt.imshow(rgb)
                plt.axis("off")

                # Segmentation
                plt.subplot(1,3,2)
                plt.title("Segmentation")
                plt.imshow(seg_color.astype(np.uint8))
                plt.axis("off")

                # Depth
                plt.subplot(1,3,3)
                plt.title("Depth")
                plt.imshow(dep_img.astype(np.uint8))
                plt.axis("off")

                plt.savefig(f"{save_dir}/result_{batch_ke}_{i}.png", bbox_inches='tight')
                plt.close()
                
            #kalkulasi metric
            iou = iou_multiclass(pred_seg, seg_gt, num_classes=19)
            mae = F.l1_loss(pred_dep, dep_gt)
            tm = iou + (1 - mae)
            
            #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
            metric_tm.update(tm.item())
            metric_seg.update(iou.item())
            metric_dep.update(mae.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('tm', metric_tm.avg),
                                  ('iou', metric_seg.avg),
                                  ('mae', metric_dep.avg)])
            
            #simpan ke log
            logx['batch'].append(batch_ke)
            logx['test_tm'].append(tm.item())
            logx['test_seg'].append(iou.item())
            logx['test_mae'].append(mae.item())
            logx['elapsed_time'].append(elapsed_time)
            pd.DataFrame(logx).to_csv(save_dir+'test_log.csv', index=False)
            
            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update()
        prog_bar.refresh()
        
        #ketika semua sudah selesai, hitung rata2 performa pada log
        logx['batch'].append("avg")
        logx['test_tm'].append(np.mean(logx['test_tm']))
        logx['test_seg'].append(np.mean(logx['test_seg']))
        logx['test_mae'].append(np.mean(logx['test_mae']))
        logx['elapsed_time'].append(np.mean(logx['elapsed_time']))
        pd.DataFrame(logx).to_csv(save_dir+'test_log.csv', index=False)

    return postfix


#BUAT DATA BATCH: NxCxHxW
rootdir = "/media/xavier/E03D6057/kharisma/"
test_set = myData(root_dir=rootdir, part='test')
print("Jumlah sample test: "+str(len(test_set)))
dataloader_test = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)

start_time = time.time() #waktu mulai
test_log = test(dataloader_test, model, device)
elapsed_time = time.time() - start_time #hitung elapsedtime
print(test_log)
print(elapsed_time)
torch.cuda.empty_cache()