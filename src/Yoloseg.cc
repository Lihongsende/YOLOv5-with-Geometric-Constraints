//It's my yolov5-seg, create by li on 2024/5/17

#include "Yoloseg.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
namespace ORB_SLAM2
{
    Yoloseg::Yoloseg()
    {
        std::cout << "Importing Yolov5-seg Settings..." << std::endl;
        Py_SetPythonHome(L"/home/li/anaconda3/envs/yolov5");
        Py_Initialize();
	    if(!Py_IsInitialized())
        {
            printf("Python init failed!\n");
        }
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('/home/li/ORBSLAM2-XIUGAI/ORB_SLAM2+yoloseg+ep/src/yolov5-seg')");

        cvt = new NDArrayConverter();

        py_module = PyImport_ImportModule("myyolo");  //YOLACT1.py
        if (py_module == nullptr) {
        // print stack， 看看具体是什么错误
        PyErr_Print(); 
        cout << "PyImport_ImportModule 'hello.py' not found" << endl;
            return ;
        }

        if (!py_module) {
            std::cout << "YOLOSEG.py文件没找到" << std::endl;
            assert(py_module != NULL);
        }

        py_dict = PyModule_GetDict(py_module); 
        if (!py_dict) {
            std::cout << "Can't find py_module!" << std::endl;
            assert(py_dict != NULL);
        }

        py_class = PyDict_GetItemString(py_dict, "YOLOSEG");
        if (!py_class) {
            std::cout << "Can't find YOLOSEG class!" << std::endl;
            assert(py_class != NULL);
        }

        net = PyObject_CallObject(py_class, nullptr); 
        if (!net) {
            std::cout << "Can't find YOLOSEG instance!" << std::endl;
            assert(net != NULL);
        }
        std::cout << "Created YOLOSEGinstance" << std::endl;
    }
    std::pair<cv::Mat,cv::Mat> Yoloseg::GetSegmentation(cv::Mat &image)
{
        std::pair<cv::Mat,cv::Mat> result;
        //cv::Mat result;
        PyObject* mask;
        PyObject* img;
        //py_image = image.clone();
        PyObject* py_image1 = cvt->toNDArray(image.clone());
	    //PyObject* py_image2 = cvt->toNDArray(image.clone());

        //assert(py_image != NULL);
        //py_mask_image = PyObject_CallMethod(net, "GetDynSeg","O",py_image);
        //PyArg_ParseTuple(py_mask_image,"O|O",&img,&mask);
        //py_dect_image = PyObject_CallMethod(net, "GetDet","O",py_image1);

        py_mask_image = PyObject_CallMethod(net, "Getmask","O",py_image1);
	    PyArg_ParseTuple(py_mask_image,"O|O",&mask,&img);
        //cv::Mat mask_ = cvt->toMat(py_dect_image ).clone();
        //cv::Mat img_ = cvt->toMat(py_mask_image).clone();

        //result = std::make_pair(img,mask);
        cv::Mat mask_ = cvt->toMat(mask).clone();
        cv::Mat img_ = cvt->toMat(img).clone();
        //if(mask_.channels() == 3)
        //{
        //    cvtColor(mask_, mask_, CV_RGB2GRAY);
        //}
        //mask_.cv::Mat::convertTo(mask_,CV_8U);//0 background y 1 foreground
        result = std::make_pair(mask_,img_);
        //result = img_;
        return result;
    }
/*
     cv::Mat Yoloseg::GetSegmentation(cv::Mat &image)
    {
        //std::pair<cv::Mat,cv::Mat> result;
        cv::Mat result;
        PyObject* mask;
        PyObject* img;
        //py_image = image.clone();

        PyObject* py_image1 = cvt->toNDArray(image.clone());

	    //PyObject* py_image2 = cvt->toNDArray(image.clone());

        //assert(py_image != NULL);
        //py_mask_image = PyObject_CallMethod(net, "GetDynSeg","O",py_image);
        //PyArg_ParseTuple(py_mask_image,"O|O",&img,&mask);
        //py_dect_image = PyObject_CallMethod(net, "GetDet","O",py_image1);
        py_mask_image = PyObject_CallMethod(net, "Getmask","O",py_image1);

        //cv::Mat mask_ = cvt->toMat(py_dect_image ).clone();
        cv::Mat img_ = cvt->toMat(py_mask_image).clone();

        //result = std::make_pair(img,mask);

        //cv::Mat mask_ = cvt->toMat(mask).clone();
        //cv::Mat img_ = cvt->toMat(img).clone();
        //if(mask_.channels() == 3)
        //{
        //    cvtColor(mask_, mask_, CV_RGB2GRAY);
        //}
        //mask_.cv::Mat::convertTo(mask_,CV_8U);//0 background y 1 foreground
        //result = std::make_pair(img_);
        result = img_;
        return result;
    }
*/
        Yoloseg::~Yoloseg()
    {
        delete this->cvt;
        delete this->py_module;
        delete this->py_dict;
        delete this->py_class;
        delete this->net;
    }
}

