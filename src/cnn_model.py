#-*- coding: UTF-8 -*-
#!/usr/bin/python
import caffe
import cv2
import os

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
       

def write_proto(proto_file_root, net_proto,mode):
    with open(proto_file_root,mode) as f:
    	f.write(str(net_proto))

def write_solver(proto_file_root,train_prototxt):
    solver = caffe.proto.caffe_pb2.SolverParameter()
    solver.net = train_prototxt      # train.prototxt
    solver.test_iter.append(25)
    solver.test_interval = 3               # test interval
    solver.base_lr = 0.05                  # base learnning rate
    solver.momentum = 0.9                  # dong liang
    solver.weight_decay = 0.0005             # weight decay
    solver.lr_policy = 'step'              # learning policy
    solver.stepsize = 5                    # change frequency of lr 
    solver.gamma = 0.1                     # lr change rate
    solver.display = 5                    # display log time interval
    solver.max_iter = 15                # max iter
    solver.snapshot = 15                 # save model intreval
    solver.snapshot_prefix = '/home/jojo/face-detection/caffe_model/face'
    solver.solver_mode = caffe.proto.caffe_pb2.SolverParameter.CPU
    solver.type = "SGD"

    with open(proto_file_root, 'w') as f:
        f.write(str(solver))

def create_img_list(img_root,img_list,start,end):
    with open(img_list, 'w') as f:
        for i in range(start,end):
            f.write(img_root+str(i)+'.jpg 9\n')

def train_net(solver_proto):
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()

def resize_train_pic(img_root,width,height):
    lists = os.listdir(img_root)
    for pic in lists:
        image = cv2.imread(img_root+pic)
        image = cv2.resize(image,(width,height),0,0, cv2.INTER_LINEAR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_root+pic,image)
    print("pic resize is done")
    
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
        f.write('input_dim:1\n')
        f.write('input_dim:1\n')
        f.write('input_dim:64\n')
        f.write('input_dim:64\n')
        f.write(str(net.to_proto()))

if __name__ == '__main__':
    prj_root = "/home/jojo/face-detection/"
    train_list = prj_root + "data/train.txt"
    test_list = prj_root + "data/test.txt"
    train_val_proto = prj_root + "caffe_model/train_val.prototxt"
    test_proto = prj_root + "caffe_model/test.prototxt"
    solver_proto = prj_root + "caffe_model/solver.prototxt"
    deploy_proto = prj_root + "caffe_model/deploy.prototxt"

    #train_val_proto = create_net(train_list,25,50)
    #net_test_proto = create_net(test_list,25,50,include_acc = 1)

    #create_net(train_val_proto,train_list,test_list,25,100)
    #write_solver(solver_proto, train_val_proto)
    #create_img_list(prj_root+'data/',train_list,0,75)
    #create_img_list(prj_root+'data/',test_list,75,100)
    creat_deploy(deploy_proto)
    #train_net(solver_proto)
