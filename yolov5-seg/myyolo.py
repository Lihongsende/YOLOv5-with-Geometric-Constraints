import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import time

from yoloseg import Yoloseg

class YOLOSEG:
    def __init__(self):
        print ('Initializing Yolov5-seg network...')
        trained_model="/home/li/ORBSLAM2-XIUGAI/ORB_SLAM2+yoloseg+ep/src/yolov5-seg/yolov5s-seg.pt"
        dynamic_names=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                            'train', 'truck', 'boat', 'traffic light', 'fire hydrant')
        self.model = Yoloseg(weights=trained_model,dynamic_names=dynamic_names)
        print ('Yolov5-seg network is working')

    def Getmask(self, img):
        im,im_copy,copy_img = self.model.load_image(img)
     
        pred, proto = self.model.forward(im)[:2]
        pred = self.model.non_max_suppression(pred, self.model.conf_thres, self.model.iou_thres, self.model.agnostic_nms, max_det=self.model.max_det, nm=32)

        mask = self.model.GetDetMask(im,pred,im_copy,copy_img,proto)

        return mask

if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    img_path='/home/li/ORBSLAM2-XIUGAI/ORB_SLAM2+yoloseg/src/yolov5-seg/12.png'
    img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    model=YOLOSEG()
    mask = model.Getmask(img)
    cv2.imwrite("./mask_my.jpg",mask)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"Total time taken: {elapsed_time} seconds")

