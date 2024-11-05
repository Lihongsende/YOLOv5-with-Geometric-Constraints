/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
* c++ 调用 python 实现的 mask-rcnn 获取mask分割结果
*/

#include "MaskNet.h"
#include <iostream>
#define SKIP_NUMBER 1
//#include"ConverTool.h"

namespace ORB_SLAM2
{

    SegmentDynObject::SegmentDynObject():mbFinishRequested(false),mbNewImgFlag(false),mSkipIndex(SKIP_NUMBER),mSegmentTime(0),imgIndex(0){}

    SegmentDynObject::~SegmentDynObject() {}

    void SegmentDynObject::SetTracker(Tracking *pTracker)
    {
        mpTracker=pTracker;
    }

    bool SegmentDynObject::isNewImgArrived()
    {
        unique_lock<mutex> lock(mMutexGetNewImg);
        if(mbNewImgFlag)
        {
            mbNewImgFlag=false;
            return true;
        }
        else
            return false;
    }

    void SegmentDynObject::Run()
    {
        yoloseg = new Yoloseg();  //Yolact.cc 
        while (1)
        {
            usleep(1);

		if(CheckFinish()) 
		break;

	    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            if(!isNewImgArrived())
        {  
	
            continue;
        }
   	    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    	double ttrack1= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        cout << "Wait for new RGB img time =" <<ttrack1*1000<< endl;

            if(mSkipIndex==SKIP_NUMBER)
            {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		if(mImgLeft.empty()&&mImgRight.empty())  
                {
		mImgSegment = yoloseg->GetSegmentation(mImg);
		num++;}   
		if(mImg.empty())
		{
		mImgSegmentLeft = yoloseg->GetSegmentation(mImgLeft);
		mImgSegmentRight = yoloseg->GetSegmentation(mImgRight);}
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

                mSegmentTime+=std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	            cout << "Wait for mSegmentTime  =" <<mSegmentTime*1000<< endl;
			cout<<"num..="<<num<<endl;
	            cout << "mean mSegmentTime  =" <<mSegmentTime/num<< endl;
                mSkipIndex=0;
                imgIndex++; 

            }
            mSkipIndex++;
	   
            ProduceImgSegment(); 

       
        }

    }

    bool SegmentDynObject::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void SegmentDynObject::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested=true;
    }

    void SegmentDynObject::ProduceImgSegment()
    {
        std::unique_lock <std::mutex> lock(mMutexNewImgSegment);

	    if(mImgLeft.empty()&&mImgRight.empty())    
            {mImgSegment.first.copyTo(mImgSegmentmask);
            mImgSegment.second.copyTo(mImgSegmentimg);}  

	if(mImg.empty())
		{mImgSegmentLeft.first.copyTo(mImgSegmentmaskLeft);
        mImgSegmentLeft.second.copyTo(mImgSegmentimgLeft);
		mImgSegmentRight.first.copyTo(mImgSegmentmaskRight);
        mImgSegmentRight.second.copyTo(mImgSegmentimgRight);}
        mpTracker->mbNewSegImgFlag=true;
    }

}
