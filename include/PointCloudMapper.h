//
// Created by yuwenlu on 2022/7/2.
//

#ifndef ORB_SLAM3_POINTCLOUDMAPPER_H
#define ORB_SLAM3_POINTCLOUDMAPPER_H
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include "KeyFrame.h"
#include <pcl/common/transforms.h>
using namespace ORB_SLAM2;
using namespace std;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PointCloudMapper
{
public:
    PointCloudMapper();

    void InsertKeyFrame(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth, cv::Mat &imMask);

    PointCloud::Ptr GeneratePointCloud(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth, cv::Mat &imMask);
	void UpdateGlobalMap(PointCloud::Ptr pointCloud_new, KeyFrame* kf, const cv::Mat &imMask);

	void InsertKeyFrame(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth);

    PointCloud::Ptr GeneratePointCloud(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth);
    void run();

    queue<KeyFrame*> mqKeyFrame;
    queue<cv::Mat> mqRGB;
    queue<cv::Mat> mqDepth;
	queue<cv::Mat> mqMask;

    //pcl::VoxelGrid<PointT>::Ptr mpVoxel;
    pcl::VoxelGrid<PointT>  mpVoxel;
    std::mutex mmLoadKFMutex;
    PointCloud::Ptr mpGlobalMap;
   
    int mKeyFrameSize;


};

#endif //ORB_SLAM2_POINTCLOUDMAPPER_H
