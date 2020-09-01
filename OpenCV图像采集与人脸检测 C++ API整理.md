# OpenCV图像采集与人脸检测 C++ API整理

## 0.Include

```C++
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
using namespace cv;
using namespace std;
```



## 1. Open Camera and VideoCapture

```C++
/*
VideoCapture(const String &filename);
VideoCapture(const String &filename, int apiPreference);
VideoCapture(int index);
VideoCapture(int index, int apiPreference);
*/
/*
    apiPreference can be: 
    reference: docs.opencv.org/3.4  3.2 and 3.4 is the same
    CAP_ANY			auto detect==0
    CAP_VFW			Video For Windows 
    CAP_V4L 		V4L/V4L2 capturing support via libv4l.
    CAP_V4L2		Same as CAP_V4L.
    CAP_DSHOW		DirectShow (via videoInput)
    ......
*/
Mat frame;
VideoCapture cap;
cap.open(CamID,CAP_V4L);
if(!cap.isOpenend())
{
    cout << "Openning camera failed"<<endl;
    cap.release();
    return 0;
}
cap >> frame;
```

## 2.Cascade  Classifier

```C++
#define cascade_file "./lbpcascade_frontalface_improved.xml"
Mat grey;
vector<Rect> face;
CascadeClassifier cascade;
cascade.load(CascadeFile);
cvtColor(frame,grey,CV_BGR2GRAY);
/*
	cvtColor(InputArray src, OutputArray dst, int code, int dstCn=0)
	code can be: opencv 3.4.11+ 
	COLOR_BGR2RGB
	COLOR_BGR2GRAY
	......
	but in opencv 3.2 is
	CV_BGR2GRAY
	......
*/
cascade.detectMultiScale(grey,face,1.15,3,0,Size(32,32),Size(320,320));
/*
	detectMultiScale(InputArray image,
				    std::vector<Rect> &objects,
				    double sacleFactor=1.1,
				    int minNeighbors=3,
				    int flags=0,
				    Size minSize(),
				    Size maxSize())
	这个函数返回的是人脸的矩形区域的位置，信息储存在object的Rect类型的容器中
*/
```

## 3.Save face and Draw face

```C++
Mat faceROI;
if(face.size() > 0)
    for(int i = 0; i< face.size();i++)
	{
	   //存储face区域为*.jpg图片，作为训练集;
        Rect roi_pt(face[i].x-25,face[i].y-25,face[i].width+50,face[i].height+50);
        faceROI = grey(roi_pt);//这里保存为灰度，将grey改为frame则是彩图
        path = PICStoreRoot + to_string(num) + ".jpg";
        cout << "Save image to:" << path <<endl;
        imwrite(path,faceROI);
        
        //实时标出人脸区域;
        rectangle(frame,Point(face[i].x-25, face[i].y-25),
                  Point(face[i].x+face[i].width+50, face[i].y+face[i].height+50),
                  Scalar(0,255,0),2,8,0);
/*
   rectangle(InputOutputArray img, Point pt1, Point pt2, const Scalar &color,
   			int thickness=1, int lineType=LINE_8, int shift=0)
   rectangle(Mat &img, Rect rec, const Scalar &color, int thickness=1,
   			int lineType=LINE_8, int shift=0)
   这里的PT1,是矩形左上角的坐标，pt2是长和宽的值。用第二个函数，也就是x,y,w,h构成的Rect，当然我们也可以用ellipes函数画出圆/椭圆形区域。
   ellipse (InputOutputArray img, Point center, Size axes, double angle,
   		    double startAngle, double endAngle, const Scalar &color,
            int thickness=1, int lineType=LINE_8, int shift=0)
*/
        putText(frame,"num:"+ to_string(num),
                Point(face[i].x + face[i].width+50, face[i].y + face[i].height+50),
                FONT_HERSHEY_PLAIN,2,
                Scalar(0,255,255),2,8,0);
 /*
 	putText(InputOutputArray img, const String &text, Point org, int fontFace,
 			double fontScale, Scalar color, int thickness=1,int lineType=LINE_8, 
 			bool bottomLeftOrigin=false)
 */
    }
```

完整的人脸采集函数代码：

```C++
int face_pic_get(string CascadeFile, int CamID, string PICStoreRoot, bool StoreIsColor, int PICNums)
{
	CascadeClassifier cascade;
	cascade.load(CascadeFile);

	VideoCapture cap;
	cap.release();
	cap.open(CamID,CAP_V4L);
	if (!cap.isOpened())
	{
		cout<<"Openning camera failed!"<<endl;
		cap.release();
		return 0;
	}
	// initial var
	string path; //pic store path
	int num = 0;
	char start = 0;
	Mat frame,grey,faceROI;
	vector<Rect> face;//face region x,y,w,h
	while(cap.isOpened() && (num < PICNums))
	{	
		while(start < 100)
			start ++;
		cap >> frame;
		if(frame.empty())
			break;
		cvtColor(frame,grey,CV_BGR2GRAY);
		cascade.detectMultiScale(grey,face,1.15,3,0,Size(32,32),Size(320,320));
		if(face.size() > 0)
			for(int i = 0; i< face.size();i++)
			{
				//save face pic;
				Rect roi_pt(face[i].x-25,face[i].y-25,face[i].width+50,face[i].height+50); 
				faceROI = grey(roi_pt);
				path = PICStoreRoot + to_string(num) + ".jpg";
				cout << "Save image to:" << path <<endl;
				imwrite(path,faceROI);
				num++;

				//point out the face region;
				rectangle(frame,Point(face[i].x-25, face[i].y-25), 
					  Point(face[i].x+face[i].width+50, face[i].y+face[i].height+50),
					  Scalar(0,255,0),2,8,0);
				putText(frame,"num:"+ to_string(num),
					Point(face[i].x + face[i].width+50, face[i].y + face[i].height+50),
					FONT_HERSHEY_PLAIN,2,
					Scalar(0,255,255),2,8,0);
			}
		//namedWindow("Face-Detection",CV_WINDOW_NORMAL);
		imshow("Face-Detection",frame);
		int c = waitKey(10);
		if((char)c == 'q')
		{
			printf("Stopped by user\n");
			return 0;
		}
	}
	cap.release();
	cout << "Getting face PIC succeed"<<endl;
	return 1;
	
}
```

## 4. Generate train.txt and test.txt

这个部分是读图片保存的根目录，获取目录下所有文件价的名字（用户的名字、工号等）

```c++
string face_input()
{
	string name,path;
	cout<<"Please Input your name:"<<"(Insert Enter to confirm)"<<endl;
	cin >> name;
	path = data_dir + name;
	int isCreate = mkdir(path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
	if (!isCreate)
	{
		cout<<"create path:"<<path<<endl;
		return (path + "/");
	}	
	else
	{
		cout<<"create path failed"<<"The folder may be exsited"<<endl;
		return NULL;
	} 
		
}
int create_pic_lists(string photo_root_path,int train_num,int num_max)
{
	ofstream File1,File2;
	string pic_path;
	string train_lists,test_lists;
	vector<string> folder_name;
	struct dirent *ptr;
	DIR *dir;
	dir = opendir(photo_root_path.c_str());
	if(dir == NULL)
	{
		cout << "cannot open:"<<photo_root_path<<endl;
		return 0;
	}

	while((ptr=readdir(dir))!=NULL)
	{
	        if((ptr->d_name[0] == '.') || (ptr->d_type != DT_DIR))
		{
			continue;
		}
		else 
		{
			folder_name.push_back(ptr->d_name);
		}
	}
	closedir(dir);

	
	train_lists = photo_root_path + "train.txt";
	test_lists = photo_root_path + "test.txt";
	File1.open(train_lists);
	File2.open(test_lists);
	if(!File1 || !File2)
	{
		cout << "Cannot open file:"<<endl;
		return 0;
	}
	File1.close();
	File2.close();


	File1.open(train_lists,ios::app);
	File2.open(test_lists,ios::app);
	for(int i=0; i<folder_name.size();i++)
	{
		pic_path = photo_root_path + folder_name[i]+"/";
		for(int j=0; j < train_num; j++)
		{
			File1 << (pic_path + to_string(j) + ".jpg " + to_string(i) + "\n");
		}
		for(int j=train_num; j < num_max; j++)
		{
			File2 << (pic_path + to_string(j) + ".jpg " + to_string(i) + "\n");
		}
		
	}
	File1.close();
	File2.close();
	cout<<"Successfully create and initalize *.txt :"<<endl<<train_lists<<endl<<test_lists<<endl;
	return 1;
}
```

## 5.main

```C++
int main(void)
{
	string store_root = face_input();
	face_pic_get(cascade_file,0,store_root,0,100);
	create_pic_lists(data_dir,75,NUM_MAX);
}
```



