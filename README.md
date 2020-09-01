# DPU-Based-FaceNet
## 0.Tool Chain and Environment

* ZedBoard：

            CPU: ARM CortexA9
  
            RAM: 512MB
  
            FPGA: Xilinx Kintex 7series
  
            DPU: B1152(90MHz, 180MHz)
  
            SD: 32GB
  
* PC: 

           Vmware Workstation 15 Player
           
           Ubuntu 18.04
           
           Caffe(source code compiled) + OpenCV 3.2(apt installed)
           
           DNNDK3.1(DECENT DNNC DLET)

* Design Flow: OpenCV --->(Face.jpg) ---> Caffe --->(train_val.prototxt, train.txt, .caffemodel) ---> DNNDK --->(deploy.caffemodel, dpu_MyFaceNet_0.elf) ---> ZedBoard 
## 1.Create your Net
  通过修改*src/cnn_model.py*文件中的函数，可以定义和训练出自己的网络结构（详见Caffe API 整理）

## 2.Quantize and compile your Net
  使用Xilinx公司提供的DECENT和DNNC软件可以对**.caffemodel文件进行裁剪（详见文档UG1327）
  
## 3.Face recognition deploy
  需要的开发板是ZedBoard，修改*myfacene/src/main.cpp*中的#define部分，包括网络结构的MACs,网络结构的名字，输入层节点的名字，人脸照片的路径。
  
## 4.Improvement
  目前实现的是对人脸图片的识别，后面会使用摄像头实时采样，然后完成人脸检测和人脸识别，并通过网口发送图片至PC机。
  
  开发板为第三方开发板Zing2（目前缺少正确版本DPU的驱动，导致DPU相关底层移植不成功），因此采用ZedBoard做技术路线的原理验证。
