import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode
from models.experimental import attempt_download, attempt_load
from utils.augmentations import  letterbox

class YOLOSEG(object):
    def __int__(self, weights='/home/li/yolov5-seg/yolov5s-seg.pt', device=torch.device('cpu'),imgsz=(640.640), dnn=False, data=None, fp16=False):
        print ('Initializing Yolov5-seg network...')
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = select_device(device)
        self.weights = weights
        self.data = data
        self.dnn = dnn
        self.fp16= fp16
        self.bs = 1
        self.classes=None
        w = str(weights[0] if isinstance(weights, list) else weights)

        # Load model
        self.model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=True)
        self.stride = 32  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.half() if fp16 else self.model.float()
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        print(' Done........')

    def inference_image(self, image):
        # 图像 预处理
        im = letterbox(image, (640,640), stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # 图像 推理
        pred, proto = self.model(im, augment=False, visualize=False)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=self.max_det, nm=32)
        
        annotator = Annotator(img, line_width=3, example=str(self.names))
        for i, det in enumerate(pred):  # per image
            if not det is None or len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()  # rescale boxes to im0 size
                # Mask plotting
                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=None if False else im[i])
                    # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):   
                    result_list.append(self.name[cls],float(conf),int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))
                    # result_list = self.inference_image(image) # [name, confidence, xmin, ymin, xmax, ymax]
        img = annotator.result()
        return result_list
    def inference_video(self, opencv_image):
        return result_list, img
    def start_video(self, video_file):
        pass
    def  start_camera(self, camera_index = 0):
        pass
    def draw_image(self, result_list, opencv_image):
        pass
    def show(self, result_list, opencv_image):
        pass

if __name__ == '__main__':
    yolov5 = YOLOSEG()
    # img_path='/home/li/yolov5-seg/bus.jpg'
    # img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    # result_list, src_image = yolov5.inference_image(img)
    #print(result_list)
    # yolov5.show(result_list, src_image)
