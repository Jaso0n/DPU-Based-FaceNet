# Caffe Python API 整理

## 1.Import Caffe Python API——PyCaffe

```python
import caffe
from caffe import layers as L
from caffe import params as P
```

## 2.Layers

* Net Spec
```python
net = caffe.NetSpec()
```
* lmdb/leveldb Data Layer

```python
net.data, net.label = L.Data(source = 'mnist/mnist_test_lmdb',
                             backend = P.Data.LMDB,
                             batch_size = 64,
                             nstop=2,
                             transform_param=dict(crop_size=227,
                                                  mean_value=[104,117,123],
                                                  mirror=True))
```

* HDF5 Data Layer

```python
net.image = L.HDF5Data(hdf_data_param={'source':'./train_data_paths.txt',
                                       'batch_size':64},
                       include={'phase':caffe.TRAIN})

net.data,net.label = L.HDF5Data(batch_size = 64,
                                source = source_path,
                                ntop=2,
                                include{'phase':caffe.TRAIN})
```

* ImageData Layer

```python
net.data, net.label = L.ImageData(source = './train.txt',
                                  batch_size=batch_size,
                                  new_width=64,
                                  new_height=64,
                                  ntop=2,
                                  tranform_param=dict(crop_size=40,mirror=True),
                                  root='/data/train/') 
```

这里的root是训练集的位置。如果写了root，则train.txt中只需要写图片名字就行了。但是一般在train.txt中写图片的绝对路径

**写了root，则train.txt的写法**
PIC_CAT.jpg 0
**不写root，则train.txt的写法**
/home/jojo/caffe/data/train/PIC_CAT.jpg 0

* Convolution Layer

```python
net.conv1 = L.Convolution(net.data,
                          kernel_size = 5,
                          num_output=20,
                          weight_filler = dict(type='xavier'),
                          bias_filler = {'type':'xavier'})
```

* ReLU Layer

```python
net.relu1 = L.ReLU(net.conv1, in_place = True)
```

* Pooling Layer

```python
net.pool1 = L.Pooling(net.relu1,kernel_size = 2, stride = 2, pool=P.Pooling.MAX)
```

* Full Connect Layer
```python
net.fc1 = L.InnerProduct(net.pool1, num_output = 512,
                         weight_filler = dict(type = 'xavier'),
                         bias_filler = dict(type = 'xavier'))
```

* Dropout Layer

```python
net.dropout1 = L.Dropout(net.fc1,in_place = False,
                         dropout_param = dict(dropout_ratio = 0.25))
```
* Flatten Layer
```python
net.flatten = L.Flatten(net.dropout1)
```


* Loss Layer
```python
net.loss = L.SoftmaxWithLoss(net.fc1,label)
```

* Accuracy Layer
```python
net.accuracy1 = L.Accuracy(net.fc1,label,
                           include = {'phase': Caffe.TEST},
                           accuracy_param=dict(top_k=5))
```

* Prob Layer

```python
net.prob = L.Softmax(net.fc1)
```

## 3.Write Proto

```python
def create_net(proto_path,train_list,test_list,batch_size,class_num):
    # net specialize
    net = caffe.NetSpec()
    # Train layer
    net.data, net.label = caffe.layers.ImageData(name = 'InputData',
						 include = dict(phase=caffe.TRAIN),
                                                 batch_size = batch_size,
                                                 source = train_list,
                                                 new_height = 64,
                                                 new_width = 64,
                                                 is_color = False,
                                                 transform_param = dict(scale = 1./255),
                                                 ntop = 2)
    write_proto(proto_path,net.to_proto(),'w')
    # Test layer
    net.data, net.label = caffe.layers.ImageData(name = 'InputData',
						 include = dict(phase=caffe.TEST),
                                                 batch_size = 1,
                                                 source = test_list,
                                                 new_height = 64,
                                                 new_width = 64,
                                                 is_color = False,
                                                 transform_param = dict(scale = 1./255),
                                                 ntop = 2)
    # 1 convolution: 32x3x3 kernels
    net.conv1 = caffe.layers.Convolution(net.data,
                                         num_output=32,
                                         kernel_size=3,
                                         weight_filler={"type":"xavier"},
                                         bias_filler={"type":"xavier"})
    # 2 ReLU
    net.relu1 = caffe.layers.ReLU(net.conv1,in_place=True)
    # 3 convolution:32x3x3 kernels
    net.conv2 = caffe.layers.Convolution(net.relu1,
                                         num_output=32,
                                         kernel_size=3,
                                         weight_filler={"type":"xavier"},
                                         bias_filler={"type":"xavier"})
    # 4 ReLU
    net.relu2 = caffe.layers.ReLU(net.conv2,in_place=True)
    # 5 Pooling
    net.pool1 = caffe.layers.Pooling(net.relu2,
                                     kernel_size=2,
                                     stride=2)
    # 6 Dropout
    net.dropout1 = caffe.layers.Dropout(net.pool1, in_place=True,
					dropout_param =dict(dropout_ratio=0.25))
    # 7 Convolution:64x3x3 kernels
    net.conv3 = caffe.layers.Convolution(net.dropout1,
                                         num_output=64,
                                         kernel_size=3,
                                         weight_filler={"type":"xavier"},
                                         bias_filler={"type":"xavier"})
    # 8 ReLU
    net.relu3 = caffe.layers.ReLU(net.conv3, in_place=True)
    # 9 Convolution:64x3x3 kernels
    net.conv4 = caffe.layers.Convolution(net.relu3,
                                         num_output=64,
                                         kernel_size=3,
                                         weight_filler={"type":"xavier"},
                                         bias_filler={"type":"xavier"})
    # 10 ReLU
    net.relu4 = caffe.layers.ReLU(net.conv4, in_place=True)
    # 11 Pooling
    net.pool2 = caffe.layers.Pooling(net.relu4,
                                     kernel_size=2,
                                     stride=2)
    # 12 Dropout
    net.dropout2 = caffe.layers.Dropout(net.pool2, in_place = True,
					dropout_param=dict(dropout_ratio=0.25))
    # 13 Flatten
    net.flatten1 = caffe.layers.Flatten(net.dropout2)
    # 14 FC
    net.fc1 = caffe.layers.InnerProduct(net.flatten1,
                                        num_output=512,
                                        weight_filler=dict(type='xavier'),
                                        bias_filler=dict(type='xavier'))
    # 15 ReLU
    net.relu5 = caffe.layers.ReLU(net.fc1, in_place=True)
    # 16 Dropout
    net.dropout3 = caffe.layers.Dropout(net.relu5, in_place = True,
					dropout_param = dict(dropout_ratio=0.5))
    # 17 FC
    net.fc2 = caffe.layers.InnerProduct(net.dropout3,
                                        num_output=class_num,
                                        weight_filler=dict(type='xavier'),
                                        bias_filler=dict(type='xavier'))
    # 18 SoftMaxwithLoss
    net.loss = caffe.layers.SoftmaxWithLoss(net.fc2,net.label)
    # Test accuracy
    net.top1 = caffe.layers.Accuracy(net.fc2,
				     net.label,
			             include=dict(phase=caffe.TEST))
    net.top5 = caffe.layers.Accuracy(net.fc2,
				     net.label,
				     include=dict(phase=caffe.TEST),
				     accuracy_param=dict(top_k=5))
    write_proto(proto_path,net.to_proto(),'a')
```

在构建网络时，一种方法是将train和test分开，还有一种是将train和test放在一起，通过include参数加以区分。上述两种构建网络的方法在生成sovler时的参数不同

* train和test分开放

```python
def write_solver(proto_file_root,train_prototxt,test_prototxt):
    solver = caffe.proto.caffe_pb2.SolverParameter()
    solver.train_net = train_prototxt      # train.prototxt
    solver.test_net.append(test_prototxt)
    solver.test_iter.append(1)			   # test iter
    solver.test_interval = 3               # test interval
    solver.base_lr = 0.05                  # base learnning rate
    solver.momentum = 0.9                  # dong liang
    solver.weight_decay = 0.0005           # weight decay
    solver.lr_policy = 'step'              # learning policy
    solver.stepsize = 5                    # change frequency of lr 
    solver.gamma = 0.1                     # lr change rate
    solver.display = 5                     # display log time interval
    solver.max_iter = 15                   # max iter
    solver.snapshot = 15                   # save model intreval
    solver.snapshot_prefix = '/home/jojo/face-detection/caffe_model/face'
    solver.solver_mode = caffe.proto.caffe_pb2.SolverParameter.CPU
    solver.type = "SGD"

    with open(proto_file_root, 'w') as f:
        f.write(str(solver))
```

* train和test放一起，见上述creat_net()

```python
def write_solver(proto_file_root,train_val_prototxt):
    solver = caffe.proto.caffe_pb2.SolverParameter()
    solver.net = train_val_prototxt        # train_val.prototxt
    solver.test_iter.append(1)			   # test iter
    solver.test_interval = 3               # test interval
    solver.base_lr = 0.05                  # base learnning rate
    solver.momentum = 0.9                  # dong liang
    solver.weight_decay = 0.0005           # weight decay
    solver.lr_policy = 'step'              # learning policy
    solver.stepsize = 5                    # change frequency of lr 
    solver.gamma = 0.1                     # lr change rate
    solver.display = 5                     # display log time interval
    solver.max_iter = 15                   # max iter
    solver.snapshot = 15                   # save model intreval
    solver.snapshot_prefix = '/home/jojo/face-detection/caffe_model/face'
    solver.solver_mode = caffe.proto.caffe_pb2.SolverParameter.CPU
    solver.type = "SGD"

    with open(proto_file_root, 'w') as f:
        f.write(str(solver))
```



参数说明：

**test_iter**：测试步长，即测试遍历整个测试集需要的步数。test_iter=test_num/test_batch_size。假设测试集有25张图片，测试batch_size = 25，那么test_iter就是25/25=1。

**test_interval**：测试间隔，也就每训练多少步，进行一次测试。通常是训练时将整个训练集遍历一次后，进行一次测试。test_interval = train_num/batch_size。举例训练集有25张图片，测试batch_size=25，那么test_interval=75/25=3。

**max_iter**：训练最大步数

**display**：显示trian log的间隔

**snapshot**：训练时，模型的保存间隔

-----------------------------------------------------------------------------------------------------------------------------------------

下列参数和学习过程有关，参考学习公式推导

**base_lr**：基础学习率，取值分类任务参考VGG、ResNet50、Inception等网络的取值，通常都是0.05

**momentum**：动量，减少波动，摆脱局部最优

**weight_decay**：权重衰减

**step_size**：改变学习率的频率

**gamma**：学习率改变速率

**lr_pilicy**：学习策略，通常是step

## 4.Train

```python
def train(solver_proto_path):
    solver = caffe.SGDSolver(solver_proto_path)
    solver.solve()
```

## 5.Deploy

训练好模型后，会获得***.caffemodel***模型文件，还需要生成**deploy.prototxt**，去除Data层和loss层，以及test相关的accuracy层，添加softmax层和输入data参数。

```python
def creat_deploy(deploy_proto):
    net = caffe.NetSpec()
    net.conv1 = caffe.layers.Convolution(bottom='data',
                                         num_output=32,
                                         kernel_size=3)
    # 2 ReLU
    net.relu1 = caffe.layers.ReLU(net.conv1,in_place=True)
    # 3 convolution:32x3x3 kernels
    net.conv2 = caffe.layers.Convolution(net.relu1,
                                         num_output=32,
                                         kernel_size=3)
    # 4 ReLU
    net.relu2 = caffe.layers.ReLU(net.conv2,in_place=True)
    # 5 Pooling
    net.pool1 = caffe.layers.Pooling(net.relu2,
                                     kernel_size=2,
                                     stride=2)
    # 6 Dropout
    net.dropout1 = caffe.layers.Dropout(net.pool1, in_place=True,
					dropout_param =dict(dropout_ratio=0.25))
    # 7 Convolution:64x3x3 kernels
    net.conv3 = caffe.layers.Convolution(net.dropout1,
                                         num_output=64,
                                         kernel_size=3)
    # 8 ReLU
    net.relu3 = caffe.layers.ReLU(net.conv3, in_place=True)
    # 9 Convolution:64x3x3 kernels
    net.conv4 = caffe.layers.Convolution(net.relu3,
                                         num_output=64,
                                         kernel_size=3)
    # 10 ReLU
    net.relu4 = caffe.layers.ReLU(net.conv4, in_place=True)
    # 11 Pooling
    net.pool2 = caffe.layers.Pooling(net.relu4,
                                     kernel_size=2,
                                     stride=2)
    # 12 Dropout
    net.dropout2 = caffe.layers.Dropout(net.pool2, in_place = True,
					dropout_param=dict(dropout_ratio=0.25))
    # 13 Flatten
    net.flatten1 = caffe.layers.Flatten(net.dropout2)
    # 14 FC
    net.fc1 = caffe.layers.InnerProduct(net.flatten1,
                                        num_output=512)
    # 15 ReLU
    net.relu5 = caffe.layers.ReLU(net.fc1, in_place=True)
    # 16 Dropout
    net.dropout3 = caffe.layers.Dropout(net.relu5, in_place = True,
					dropout_param = dict(dropout_ratio=0.5))
    # 17 FC
    net.fc2 = caffe.layers.InnerProduct(net.dropout3,
                                        num_output=100)
    # 18 SoftMax
    net.prob = caffe.layers.Softmax(net.fc2)

    with open(deploy_proto,'w') as f:
        f.write('input:"data"\n')
        f.write('input_dim:1\n')  # 这个参数一般是1
        f.write('input_dim:1\n')  # 这个是图像的维度，灰度为1，彩色为3
        f.write('input_dim:64\n')
        f.write('input_dim:64\n')
        f.write(str(net.to_proto()))
```

## 6.Forward inferrence

```python
def test(caffe_model,deploy_proto,image):
    net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape}) 
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data',255)
    #transformer.set_channel_swap('data',(2,1,0)) # 彩色图取消注释这行

    im = caffe.io.load_image(image,color=False) # 彩色图color=True
    net.blobs['data'].data[...] = transformer.preprocess('data',im)

    out = net.forward()
    prob = net.blobs['prob'].data[0].flatten() # 'prob'改为用户的deploy.prototxt中Sofmax层的名字
    order = prob.argsort()[-1]
    print(order)
```



