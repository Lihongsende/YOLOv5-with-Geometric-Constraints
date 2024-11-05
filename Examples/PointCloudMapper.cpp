//
// Created by yuwenlu on 2022/7/2.
//
#include "PointCloudMapper.h"
#include "Converter.h"
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <memory> // 确保包括这一行
#include <boost/make_shared.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 构造函数
PointCloudMapper::PointCloudMapper()
{
    mpGlobalMap = boost::make_shared<PointCloud>();
    cout << "voxel set start" << endl;
    mpVoxel.setLeafSize(0.01, 0.01, 0.01);
    cout << "voxel set finish" << endl;
}

// 插入关键帧，并添加RGB、深度图像和掩码
void PointCloudMapper::InsertKeyFrame(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth, cv::Mat &imMask)
{
    std::lock_guard<std::mutex> lck_loadKF(mmLoadKFMutex);
    mqKeyFrame.push(kf);
    mqRGB.push(imRGB.clone());
    mqDepth.push(imDepth.clone());
    mqMask.push(imMask.clone()); // 新增掩码存储
}

// 生成点云
PointCloud::Ptr PointCloudMapper::GeneratePointCloud(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth, cv::Mat &imMask)
{
    PointCloud::Ptr pointCloud_temp(new PointCloud);

    for (int v=0; v<imRGB.rows; v++)
    {
        for (int u=0; u<imRGB.cols; u++)
        {
            float d = imDepth.ptr<float>(v)[u];
            if (d < 0.01 || d > 10) continue;

            // 检查掩码，黑色区域表示动态物体
             int val = (int)imMask.at<uchar>(v, u)/255;
            if (val == 0) // 动态物体
            {
                continue; // 跳过动态物体的点云
            }

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

    // 使用关键帧的位姿将点云进行变换
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

    // 确保全局动态物体点云存储
    PointCloud::Ptr global_dynamic_cloud(new PointCloud);

    while (true)
    {
        {
            std::lock_guard<std::mutex> lck_loadKFSize(mmLoadKFMutex);
            mKeyFrameSize = mqKeyFrame.size();
        }

        if (mKeyFrameSize != 0)
        {
            // 从队列获取前一个关键帧及其图像和掩码
            KeyFrame* kf = mqKeyFrame.front();
            cv::Mat imRGB = mqRGB.front();
            cv::Mat imDepth = mqDepth.front();
            cv::Mat imMask = mqMask.front(); // 获取当前掩码

            PointCloud::Ptr pointCloud_new = GeneratePointCloud(kf, imRGB, imDepth, imMask);
            mqKeyFrame.pop();
            mqRGB.pop();
            mqDepth.pop();
            mqMask.pop();

            ID++;
            std::cout << "pointCloud_new PointCloud size before cleaning: " << pointCloud_new->points.size() << std::endl;

            // 将当前帧的动态物体添加到全球动态物体点云中
            PointCloud::Ptr temp_dynamic(new PointCloud);
            for (const auto& point : pointCloud_new->points)
            {
                int u = static_cast<int>((point.x * kf->fx) / point.z + kf->cx);
                int v = static_cast<int>((point.y * kf->fy) / point.z + kf->cy);

                // 确保 (u, v) 在掩码图的范围内
                if (u >= 0 && u < imMask.cols && v >= 0 && v < imMask.rows)
                {
                    int mask_value = (int)imMask.at<uchar>(v, u) / 255; // 获取掩码图对应位置的值
                    if (mask_value == 0) // 动态物体
                    {
                        temp_dynamic->points.push_back(point); // 添加当前动态物体
                    }
                }
            }

            // 清除全局动态物体的点云
            PointCloud::Ptr temp_global(new PointCloud);
            pcl::copyPointCloud(*mpGlobalMap, *temp_global);

            // 移除动态物体点
            for (auto it = temp_global->points.begin(); it != temp_global->points.end();)
            {
                // 将点的图像坐标映射回原图像坐标 (u, v)
                int u = static_cast<int>((it->x * kf->fx) / it->z + kf->cx);
                int v = static_cast<int>((it->y * kf->fy) / it->z + kf->cy);

                // 确保 (u, v) 在掩码图的范围内
                if (u >= 0 && u < imMask.cols && v >= 0 && v < imMask.rows)
                {
                    int mask_value = (int)imMask.at<uchar>(v, u) / 255; // 获取掩码图对应位置的值
                    if (mask_value == 0) // 动态物体
                    {
                        it = temp_global->points.erase(it); // 移除动态物体点云
                        continue;
                    }
                }
                ++it;
            }

            // 合并更新后的全局点云和当前帧的点云
            *mpGlobalMap = *temp_global; // 只保存前一帧合并后的结果
            *mpGlobalMap += *pointCloud_new; // 合并当前帧 (不消除动态点)

	// 在这里加入体素滤波器的设置和应用
            mpVoxel.setInputCloud(mpGlobalMap); // 设置输入点云
            mpVoxel.filter(*mpGlobalMap); // 对合并后的全局点云进行滤波

            // 现在显示合并后的全局点云，而不包括之前帧的动态物体
            Viewer.showCloud(mpGlobalMap); // 显示合并后的全局点云
            std::cout << "PointCloud size: " << mpGlobalMap->size() << std::endl;
        }

        // 保存点云图（在最后一次更新后保存）
        if (!mpGlobalMap->empty()) 
        { 
            pcl::io::savePCDFileBinary("vslam_final.pcd", *mpGlobalMap); // 保存合并后的点云
        }

        // 控制循环速度
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
}
