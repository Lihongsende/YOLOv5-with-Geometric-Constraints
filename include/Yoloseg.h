//
// Created by li on 2024/5/19.
//

#ifndef ORB_SLAM2_YOLACT_H
#define ORB_SLAM2_YOLACT_H

#include <Python.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdio>
#include <boost/thread.hpp>
#include "include/Conversion.h"
#include <mutex>
#include "Tracking.h"
#include "System.h"


namespace ORB_SLAM2
{
    class Yoloseg
    {
    private:
        NDArrayConverter *cvt; 	
        PyObject *py_module=NULL;
        PyObject *py_dict = NULL;
        PyObject *py_class=NULL;
        PyObject *net=NULL;
        PyObject *py_image=NULL;
	    PyObject *py_dect_image=NULL;
	    //PyObject *py_dect_image=NULL;
        PyObject *py_mask_image=NULL;
        //PyObject* py_func=NULL;
        //PyObject* args = NULL;
        //void ImportSettings();
    public:
	int fid = 0;
        Yoloseg();
        ~Yoloseg();
        //cv::Mat GetSegmentation(cv::Mat &image, std::string dir="no_save", std::string rgb_name="no_file");
        std::pair<cv::Mat,cv::Mat> GetSegmentation(cv::Mat &image);
        //cv::Mat GetSegmentation(cv::Mat &image);
        //cv::Mat ros_GetSegmentation(cv::Mat &image);
    };
    
}

#endif //ORB_SLAM2_YOLACT_H
