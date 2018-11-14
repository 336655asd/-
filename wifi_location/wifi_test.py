# -*- coding: utf-8 -*-
"""

@author: cc
"""
#y[0][0],y[0][1]为返回的横纵坐标
#需要输入参照坐标x,y
#运行,在当前路径下 python beidou_wifi.py x y

import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("x", type=int, help="横坐标x")
parser.add_argument("y", type=int, help="纵坐标y")
args = parser.parse_args()
#当前位置
location=np.zeros([1,2])
#地图形状mr*mr
mr=512
#地图
map=np.zeros([mr,mr])
#搜索半径，50米
r=200
#站点
#模拟每隔50米，一个AP
map_ap=np.ones([mr,mr])
for i in range(16):
    x=i*mr/16
    for j in range(16):
        y=j*mr/16
        map_ap[x][y]=10
        
#矩阵输出
def P(array):
    for i in range(mr):
        for j in range(mr):
            if array[i][j]!=0:
                
                print '{}-{}'.format(i,j),array[i][j]
#计算信号大小
                
def RSSI(location,i,j):
    #k=1
    #b=0
    rssi=np.sqrt(np.square(location[1]-j)+np.square((location[0]-i)))
    return rssi
    
def Judge(ap_return,i,j,rssi):
    if -1<i and i<mr:
        if -1<j and j<mr:
            ap_return[i][j]=rssi
#计算信号分布图    
def Discovery(map_ap,r,location):
    #信号分布图，作为输入
    ap_return=np.zeros([mr,mr])  
    i0=location[0]-r
    j0=location[1]-r
    i0=i0.astype(int)
    j0=j0.astype(int)
    #print i0,j0
    for i in range(i0,i0+2*r):
        if -1<i and i<mr:
            for j in range(j0,j0+2*r):
                if -1<j and j<mr:
                    
                    if map_ap[i][j]==10:
                       # print '{}-{}'.format(i,j)
                        #print '[{}-{}],rssi:{}'.format(i,j,RSSI(location,i,j))
                        rssi=RSSI(location,i,j)                        
                        Judge(ap_return,i-1,j-1,rssi)
                        Judge(ap_return,i-1,j,rssi)
                        Judge(ap_return,i-1,j+1,rssi)
                        Judge(ap_return,i,j-1,rssi)
                        Judge(ap_return,i,j,rssi)
                        Judge(ap_return,i,j+1,rssi)
                        Judge(ap_return,i+1,j-1,rssi)
                        Judge(ap_return,i+1,j,rssi)
                        Judge(ap_return,i+1,j+1,rssi)
    return ap_return

def Batch(size):
    
    label=np.zeros([size,2],dtype=float)
    return_ap=np.zeros([size,512,512])
    for i in range(size):
        label[i][0]=np.random.randint(mr)
        label[i][1]=np.random.randint(mr)
        return_ap[i]=Discovery(map_ap,r,label[i])
        
    return label,return_ap
#tensorflow相关函数
def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)
    #print 'weight初始化'
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial=tf.ones(shape);
    #print 'bias初始化'
    return tf.Variable(initial,name=name)
    
def conv2d(x,w,name):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME',name=name)

def max_pool_2x2(x):
     return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def wifi_cnn(ap):
    
    data=tf.reshape(ap,[-1,mr,mr,1])
    #print '开始卷积'
    #第一层卷积
    
    with tf.name_scope('CNN'):
        with tf.name_scope('conv1'):
            w_conv1=weight_variable([3,3,1,16],'w_conv1')
            b_conv1=bias_variable([16],'b_conv1')
            r_conv1=tf.nn.relu(conv2d(data,w_conv1,'conv1')+b_conv1)
            tf.summary.histogram('w_conv1',w_conv1)
            tf.summary.histogram('b_conv1',b_conv1)
    #第一层池化
            r_pool1=max_pool_2x2(r_conv1)
    #print '第1层shape',r_pool1.shape
    #第二层卷积
        with tf.name_scope('conv2'):
            w_conv2=weight_variable([3,3,16,32],'w_conv2')
            b_conv2=bias_variable([32],'b_conv2')
            r_conv2=tf.nn.relu(conv2d(r_pool1,w_conv2,'conv2')+b_conv2)
    #第二层池化
            r_pool2=max_pool_2x2(r_conv2)
            tf.summary.histogram('w_conv2',w_conv2)
            tf.summary.histogram('b_conv2',b_conv2)
    #print '第2层shape',r_pool2.shape
    #第三层卷积
        with tf.name_scope('conv3'):
            w_conv3=weight_variable([3,3,32,32],'w_conv3')
            b_conv3=bias_variable([32],'b_conv3')
            r_conv3=tf.nn.relu(conv2d(r_pool2,w_conv3,'conv3')+b_conv3)
    #第三层池化
            r_pool3=max_pool_2x2(r_conv3)
            tf.summary.histogram('w_conv3',w_conv3)
            tf.summary.histogram('b_conv3',b_conv3)
    #print '第3层shape',r_pool3.shape
    #第四层卷积
        with tf.name_scope('conv4'):
            w_conv4=weight_variable([3,3,32,64],'w_conv4')
            b_conv4=bias_variable([64],'b_conv4')
            r_conv4=tf.nn.relu(conv2d(r_pool3,w_conv4,'conv4')+b_conv4)
    #第四层池化
            r_pool4=max_pool_2x2(r_conv4)
            tf.summary.histogram('w_conv4',w_conv4)
            tf.summary.histogram('b_conv4',b_conv4)
   # print '第四层shape',r_pool4.shape
    #print '全连接层'
    #当前图像为，63*63*64，4000000
    #全连接层FC，（250*250*64，1024）
    with tf.name_scope('Full_connection'):
        with tf.name_scope('fc1'):
            w_fc=weight_variable([32*32*64,64],'w_fc1')
            b_fc=bias_variable([64],'b_fc1')
            r_pool4_flat=tf.reshape(r_pool4,[-1,32*32*64])
            r_fc1=tf.nn.relu(tf.matmul(r_pool4_flat,w_fc)+b_fc)
            tf.summary.histogram('w_fc1',w_fc)
            tf.summary.histogram('b_fc1',b_fc)
        """
    #drop层
    print 'drop'
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(r_fc1, keep_prob)
    """
        #print '结果'
    #结果：将64映射到[x,y]
        with tf.name_scope('fc2'):
            w_fc2=weight_variable([64,2],'w_fc2')
            b_fc2=bias_variable([2],'b_fc2')
            r_fc2=tf.matmul(r_fc1,w_fc2)+b_fc2
            tf.summary.histogram('w_fc2',w_fc2)
            tf.summary.histogram('b_fc2',b_fc2)
    return r_fc2
    
#    return r_fc2,keep_prob

if __name__=='__main__':
    
    x=tf.placeholder(tf.float32,[None,mr,mr])
    y_=tf.placeholder(tf.float32,[None,2])
    #tf.summary.histogram('real_location',y_)
    #y_conv,keep_prob=wifi_cnn(x)
    y_conv=wifi_cnn(x)
    #tf.summary.histogram('pred_location',y_conv[0])
    with tf.name_scope('Train'):
        with tf.name_scope('cross_entropy'):
            cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('loss'):
            loss=tf.reduce_mean(tf.pow(tf.subtract(y_conv,y_),2))
            length=tf.sqrt(loss*2)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('length', length)
        with tf.name_scope('adamoptimizer'):
            train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
    #loss=tf.pow(tf.subtract(y_,y_conv[0]),2.0)
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    #print 'session 开始'
    with tf.Session() as sess:
        saver.restore(sess,"D:\wifi_model\wifi_location.ckpt")
        #saver.restore(sess,"model\wifi_location.ckpt")
        #merge = tf.summary.merge_all() 

        #sess.run(init_op)
        #print '变量初始化完成'
        timei=1
        timej=1
        batch_size=20
        for i in range(timei):
            for j in range(timej):
                #print '[{},{}]'.format(i,j) 
                #print '训练误差'
                location[0][0]=args.x
                location[0][1]=args.y
                
                #print '随机完成'
                r_ap=Discovery(map_ap,r,location[0]) 
                return_ap=np.zeros([1,mr,mr])
                for i1 in range(mr):
                    for j1 in range(mr):
                        return_ap[0][i1][j1]=r_ap[i1][j1]
                        
            train_length=sess.run(length,feed_dict={x:return_ap,y_:location})
            train_loss=sess.run(loss,feed_dict={x:return_ap,y_:location})
            
            y=sess.run(y_conv,feed_dict={x:return_ap,y_:location})
            print ('目标坐标[{},{}]'.format(location[0][0],location[0][1]))
            print ('预测坐标[{},{}]'.format(y[0][0],y[0][1]))    
            print ('this is the turn :{}-{}, the loss is {},the length is {}'.format(i+1,j+1,train_loss,train_length))
#y[0][0],y[0][1]为最终预测坐标
                
       
    
    
    


