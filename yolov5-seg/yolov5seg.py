import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
 
from utils.plots import Annotator, colors, save_one_box
 
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from utils.segment.general import masks2segments, process_mask
 
class yolov5_test():
    def __init__(self, 
                    weights=None,    
                    data=None,
                    imgsz=(640, 640),
                    conf_thres=0.25,
                    iou_thres=0.4,
                    max_det=1000,
                    device='cpu',
                    classes=None,
                    agnostic_nms=False,
                    augment=False,
                    visualize=False,
                    line_thickness=3,
                    half=False,
                    dnn=False,
                    vid_stride=1,
                    retina_masks=False):
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        self.retina_masks = retina_masks 
 
    def image_callback(self, image_raw):
 
        save_path = "/home/li/yolov5-seg/mybus.jpg"    # 测试用，使用修改成自己的路径
        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
 
        # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt = [0.0, 0.0, 0.0]
 
        bs = 1  # batch_size
        img = letterbox(image_raw, self.imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        t1 = time_sync()
 
        im = torch.from_numpy(im).to(model.device)
        # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im = im.half() if model.fp16 else im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
 
        t2 = time_sync()
        dt[0] += t2 - t1
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred, proto = model(im, augment=self.augment, visualize=self.visualize)[:2]
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        dt[2] += time_sync() - t3
 
        for i, det in enumerate(pred):
            im0 = image_raw
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size 
                segments = [
                            scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                            for x in reversed(masks2segments(masks))]       
                annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if self.retina_masks else im[i])
                
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg)   # label format
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
        
        
        im0 = annotator.result()
 
        # Save results (image with detections)
        cv2.imwrite(save_path, im0)    

if __name__ == '__main__':
       model = yolov5_test(weights='/home/li/yolov5-seg/yolov5s-seg.pt')
       img_path='/home/li/yolov5-seg/bus.jpg'
       img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
       model.image_callback(img)
