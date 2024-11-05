
#include "PointCloudMapper.h"
#include "Converter.h"
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <memory> 
#include <boost/make_shared.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

PointCloudMapper::PointCloudMapper()
{
    mpGlobalMap = boost::make_shared<PointCloud>();
    cout << "voxel set start" << endl;
    mpVoxel.setLeafSize(0.01, 0.01, 0.01);
    cout << "voxel set finish" << endl;
}

void PointCloudMapper::InsertKeyFrame(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth, cv::Mat &mask)
{
    std::lock_guard<std::mutex> lck_loadKF(mmLoadKFMutex);
    mqKeyFrame.push(kf);
    mqRGB.push(imRGB.clone());
    mqDepth.push(imDepth.clone());
    mqMask.push(mask.clone());  
}

PointCloud::Ptr PointCloudMapper::GeneratePointCloud(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth, cv::Mat &mask)
{
    PointCloud::Ptr pointCloud_temp(new PointCloud);

    for (int v=0; v<imRGB.rows; v++)
    {
        for (int u=0; u<imRGB.cols; u++)
        {
            cv::Point2i pt(u, v);
            float d = imDepth.ptr<float>(v)[u];
            if (d < 0.01 || d > 10 || mask.ptr<uint8_t>(v)[u] == 0) continue; 
            PointT p;
            p.z = d;
            p.x = (u - kf->cx) * p.z / kf->fx;
            p.y = (v - kf->cy) * p.z / kf->fy;

            p.b = imRGB.ptr<cv::Vec3b>(v)[u][0];
            p.g = imRGB.ptr<cv::Vec3b>(v)[u][1];
            p.r = imRGB.ptr<cv::Vec3b>(v)[u][2];
            pointCloud_temp->push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr pointCloud(new PointCloud);
    pcl::transformPointCloud(*pointCloud_temp, *pointCloud, T.inverse().matrix());
    pointCloud->is_dense = false;
    return pointCloud;
}

void PointCloudMapper::run()
{
    pcl::visualization::CloudViewer Viewer("Viewer");
    cout << endl << "PointCloudMapping thread start!" << endl;
    int ID = 0;
    while (1)
    {
        {
            std::lock_guard<std::mutex> lck_loadKFSize(mmLoadKFMutex);
            mKeyFrameSize = mqKeyFrame.size();
        }
        if (mKeyFrameSize != 0)
        {
            PointCloud::Ptr pointCloud_new(new PointCloud);
            cv::Mat mask = mqMask.front();
            pointCloud_new = GeneratePointCloud(mqKeyFrame.front(), mqRGB.front(), mqDepth.front(), mask);
            mqKeyFrame.pop();
            mqRGB.pop();
            mqDepth.pop();
            mqMask.pop(); 

            ID++;
            *mpGlobalMap += *pointCloud_new;
            PointCloud::Ptr temp(new PointCloud);
            pcl::copyPointCloud(*mpGlobalMap, *temp);
            mpVoxel.setInputCloud(temp);
            mpVoxel.filter(*mpGlobalMap);
     
            Viewer.showCloud(mpGlobalMap);
	        std::cout << "PointCloud size: " << mpGlobalMap->size() << std::endl;
        }
        
        if (!mpGlobalMap->empty()) { 
            pcl::io::savePCDFileBinary("vslam_final.pcd", *mpGlobalMap); 
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
