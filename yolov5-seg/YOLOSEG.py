import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
 
from utils.plots import Annotator, colors, save_one_box
 
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from utils.segment.general import masks2segments, process_mask

class YOLOSEG():
    def __init__(self, 
                    weights='/home/li/yolov5-seg/yolov5s-seg.pt',    
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
        print ('Initializing Yolact network...')
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
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        # print(self.names)
        print('Done...')

    def Image_preprocess(self, image):
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        bs = 1  # batch_size
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255  

        if len(im.shape) == 3:
            im = im[None]  

        return im
    
    def Get_object_detection(self, image,im): 
        pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    #line = (cls, *seg)  
                    c = int(cls)  
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        return im0        
     
    def Get_segmentation(self, image,im):
        pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        segmentation_info = []

        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  
                segments = [scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True) for x in reversed(masks2segments(masks))]       
                annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if self.retina_masks else im[i])
                
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1) 
                    c = int(cls)
                    segmentation_info.append({
                    'class': self.names[c],
                    'mask': seg  # Assuming seg is the segmentation mask for the current detection
                        })
                    
                #for idx, seg_info in enumerate(segmentation_info):
                #    class_name = seg_info['class']
                #    mask = seg_info['mask']   
                #    mask_file_path = f'{class_name}_mask_{idx}.png'
                #    cv2.imwrite(mask_file_path, mask)
                #    print(f"Segmentation mask for class '{class_name}' saved to '{mask_file_path}'")
        
        im0 = annotator.result()
        return  im0

    def Get_final_result(self,image,im):
        pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  
                segments = [scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True) for x in reversed(masks2segments(masks))]       
                annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if self.retina_masks else im[i])
                
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1) 
                    line = (cls, *seg)  
                    c = int(cls)  
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
        
        im0 = annotator.result()
        return im0

    def Save_image(self,image,image_path):
         save_path = image_path
         cv2.imwrite(image_path,image)

    def Get_detection_ifo(self, image,im):
        pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        detection_info = []
        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)  
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detection_info.append({'class': self.names[c], 'confidence':float(conf), 'bbox_left_top': int(xyxy[0]),
                                           'bbox_right_top': int(xyxy[1]),'bbox_left_bottom': int(xyxy[2]),'bbox_right_bottom': int(xyxy[3])})  
        im0 = annotator.result()
        return im0,detection_info        

    def Get_segment_ifo(self, image,im):
        pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        segmentation_info = []

        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  
                segments = [scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True) for x in reversed(masks2segments(masks))]       
                annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if self.retina_masks else im[i])
                
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1) 
                    c = int(cls)
                    segmentation_info.append({
                    'class': self.names[c],
                    'mask': seg  # Assuming seg is the segmentation mask for the current detection
                        })
                    
        im0 = annotator.result()
        return  im0

def RemovePerson(self, image):
    # 预处理图像
    im = self.Image_preprocess(image)
    
    # 获取分割结果
    pred, proto = self.model(im, augment=self.augment, visualize=self.visualize)[:2]
    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
    
    image_without_person = None
    mask_without_person = None

    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            segments = [scale_segments(im.shape[2:], x, im.shape, normalize=True) for x in reversed(masks2segments(masks))]

            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if self.names[int(cls)] != 'person':
                    seg = segments[j].reshape(-1)
                    if image_without_person is None:
                        image_without_person = image.copy()
                        mask_without_person = np.zeros_like(seg)
                    mask_without_person |= seg
                    image_without_person[seg.astype(bool)] = 0  

        return image_without_person, mask_without_person

    def GetDet(self, image):
        im=self.Image_preprocess(image)
        detection = self.Get_object_detection(image,im)
        return detection
    
    def GetSeg(self, image):
        im=self.Image_preprocess(image)
        segment= self.Get_segmentation(image,im)
        return segment
    

if __name__ == '__main__':
       model = YOLOSEG()
       img_path='/home/li/yolov5-seg/12.png'
       img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
       im=model.Image_preprocess(img)

       #detection = model.Get_object_detection(img,im)
       #image_path = '/home/li/yolov5-seg/12_det.jpg'
       #model.Save_image(detection,image_path )

       #segment=model.Get_segmentation(img,im)
       #image_path = '/home/li/yolov5-seg/12_seg.png'
       #model.Save_image(segment,image_path)

       #result=model.Get_final_result(img,im)
       #image_path = '/home/li/yolov5-seg/12_final.png'
       #model.Save_image(segment,image_path)

       detection= model.GetDet(img)
       image_path = '/home/li/yolov5-seg/12_1.png'
       model.Save_image(detection,image_path)
       image_path = '/home/li/yolov5-seg/12_3.png'
       model.Save_image(img,image_path)
       print(type(detection))
       segment = model.GetSeg(img)
       image_path = '/home/li/yolov5-seg/12_2.png'
       model.Save_image(segment,image_path)
       image_path = '/home/li/yolov5-seg/12_4.png'
       model.Save_image(img,image_path)
       print(type(segment))

       

       #print(result_list)
       #for entry in segmentation_info:
            #class_name = entry['class']
            #print(class_name)
       # print(dection_ifo)
