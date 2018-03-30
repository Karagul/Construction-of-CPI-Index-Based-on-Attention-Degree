#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:25:28 2018

@author: wangyikai
"""
import tensorflow as tf

#定义神经网络的输入输出结点，每个样本为1*315维，以及输出分类结果
INPUT_NODE=1800
OUTPUT_NODE=1

#定义两层隐含层的神经网络，一层300个结点，一层100个结点
LAYER1_NODE=500
LAYER2_NODE=200

#定义学习率，学习率衰减速度，正则系数，训练调整参数的次数以及平滑衰减率
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.9
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=100000
MOVING_AVERAGE_DECAY=0.99


#定义整个神经网络的结构，也就是向前传播的过程，avg_class为平滑可训练量的类，不传入则不使用平滑
def inference(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
    if avg_class==None:
        #第一层隐含层，输入与权重矩阵乘后加上常数传入激活函数作为输出
        layer1=tf.nn.sigmoid(tf.matmul(input_tensor,w1)+b1)
        #第二层隐含层，前一层的输出与权重矩阵乘后加上常数作为输出
        layer2=tf.nn.sigmoid(tf.matmul(layer1,w2)+b2)
        #返回 第二层隐含层与权重矩阵乘加上常数作为输出
        return tf.matmul(layer2,w3)+b3
    else:
        #avg_class.average()平滑训练变量，也就是每一层与上一层的权重
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
        layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))
        return tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3)


#定义输出数据的地方，None表示无规定一次输入多少训练样本,y_是样本标签存放的地方
x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

#依次定义每一层与上一层的权重，这里用随机数初始化，注意shape的对应关系
w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))

w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

#输出向前传播的结果
y=inference(x,None,w1,b1,w2,b2,w3,b3)

#每训练完一次就会增加的变量
global_step=tf.Variable(0,trainable=False)

#定义平滑变量的类，输入为平滑衰减率和global_stop使得每训练完一次就会使用平滑过程
variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#将平滑应用到所有可训练的变量，即trainable=True的变量
variable_averages_op=variable_averages.apply(tf.trainable_variables())

#输出平滑后的预测值
average_y=inference(x,variable_averages,w1,b1,w2,b2,w3,b3)

#定义交叉熵和损失函数，但为什么传入的是label的arg_max(),就是对应分类的下标呢，我们迟点再说
#cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
#计算交叉熵的平均值，也就是本轮训练对所有训练样本的平均值
#cross_entrip_mean=tf.reduce_mean(cross_entropy)
mse_mean = tf.reduce_mean((y-y_)**2)

#定义正则化权重，并将其加上交叉熵作为损失函数
#regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#regularization=regularizer(w1)+regularizer(w2)+regularizer(w3)
#loss=cross_entrip_mean+regularization
#loss = mse_mean+regularization
#定义动态学习率，随着训练的步骤增加不断递减
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,900,LEARNING_RATE_DECAY)
#定义向后传播的算法，梯度下降发，注意后面的minimize要传入global_step
#train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_mean,global_step=global_step)
#管理需要更新的变量，传入的参数是包含需要训练的变量的过程
train_op=tf.group(train_step,variable_averages_op)

#正确率预测
#correct_prediction=tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#temp = tf.square(average_y,y_)
#accuracy = tf.reduce_mean(temp)
with tf.Session() as sess:
    #初始所有变量
    tf.global_variables_initializer().run()
    #训练集输入字典
    train_feed={x:data_train_x[:18],y_:wyk_y[:18]}
    #测试集输入字典
    validate_feed={x:data_train_x[18:],y_:wyk_y[18:]}

    for i in range(TRAINING_STEPS):
        if i%100==0:
            validate_acc=sess.run(mse_mean,feed_dict=validate_feed)
            #validate_acc=sess.run(mse_mean,feed_dict=validate_feed)
            print(sess.run(y,feed_dict=validate_feed))
            print(sess.run(y_,feed_dict=validate_feed))
            print("After %d training step(s),validation accuracy using average model is %g"%(i,validate_acc))
            #print(sess.run(y, feed_dict=validate_feed))
        #每一轮通过同一训练集训练，由于样本太少，没办法了
        sess.run(train_op,feed_dict=train_feed)
    #用测试集查看模型的准确率
    #test_acc=sess.run(accuracy,feed_dict=test_feed)
    #print("After %d training step(s),test accuracy using average model is %g"%(TRAINING_STEPS,test_acc))
