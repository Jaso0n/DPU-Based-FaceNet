import caffe
import numpy as np

def test(caffe_model,deploy_proto,image):
    net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape}) 
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data',255)
    #transformer.set_channel_swap('data',(2,1,0))

    im = caffe.io.load_image(image,color=False)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)

    out = net.forward()
    prob = net.blobs['prob'].data[0].flatten()
    order = prob.argsort()[-1]
    print(order)
    
if __name__ == '__main__':
    prj_root = '/home/jojo/face-detection/'
    caffe_model = prj_root + 'caffe_model/face_iter_15.caffemodel'
    deploy_proto = prj_root + 'caffe_model/deploy.prototxt'
    image = prj_root + 'data/xuchang/10.jpg'

    test(caffe_model,deploy_proto,image)
    
