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
#include <unistd.h>

#define data_dir "/home/jojo/face-detection/data/"
#define cascade_file "/home/jojo/face-detection/opencv_model/lbpcascade_frontalface_improved.xml"
#define NUM_MAX 100
using namespace cv;
using namespace std;
/*
	This fucntion is used to create the caffe train.txt according to the data folder 
	which stored the user face photos.
	@parameter:
		path: user data folder root path
		train_num: photo for training
		num_max: the max num of face photos, normally = 100
	@return
		0: Failed
		1: Success
*/
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
/*
	This function gets your name and create your user dir
	@parameter
		None
	@return
		NULL: Failed
		String path: a string to user folder 
*/
string face_input()
{
	string name,path;
	string s;
	cout<<"Please Input your name:"<<"(Insert Enter to confirm)"<<endl;
	cin >> name;
	path = data_dir + name;

	if(access(path.c_str(),F_OK) == 0)// -1 not exsist ; 0 exsist
	{
		cout <<"You have registered"<<endl;
		cout <<"Do you want to resampled?"<<endl;
		cout <<"Please insert y or n"<<endl;
		cin >> s;
		if (s[0] == 'y')
		{
			cout << "Your first insert is y"<<endl;
			cout << path + "/"<<endl;
			return (path + "/");
		}
		else if(s[0] == 'n')
		{
			cout << "Your first insert is n"<<endl;
			return "STOP";
		}	
		else return "STOP";
	}
	else
	{
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
}
/*
	This function gets your face photo to be store to user dir
	@parameter:
		CascadeFile : OpenCV CascadClassifier XML file
		CamID	    : camero device ID,usually CamID = 0 or 1
		PICStoreRoot: user dir
		StoreIsColor: 0=grey,1=color
		PICNums	    : usually PICNums = 100 < NUM_MAX
	@return:
		0: Failed
		1: Success
		
*/
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
		//drop the first 10 images
		while(start < 10)
		{
			cap >> frame;
			if(frame.empty())
				continue;
			start ++;
		}
		cap >> frame;
		if(frame.empty())
			continue;
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
			cap.release();
			printf("Stopped by user\n");
			return 0;
		}
	}
	cap.release();
	cout << "Getting face PIC succeed"<<endl;
	return 1;
	
}
int main(void)
{
	int status;
	string store_root = face_input();
	if(store_root == "STOP")
		return 0;	
	status = face_pic_get(cascade_file,0,store_root,0,100);
	if (status == 0)
		return 0;
	create_pic_lists(data_dir,75,NUM_MAX);
}
