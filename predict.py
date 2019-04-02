# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:20:46 2018

@author: 远望-杨了
"""

import tensorflow as tf
import numpy as np
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
#
#import os.path
import glob

start = time.clock()
# 模型目录
CHECKPOINT_DIR = './runs/1553736359/checkpoints'  #训练后生成的检查点文件夹，在当前工程下。
INCEPTION_MODEL_FILE = './models/tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
#path = 'E:/CR雷达回波图像数据/MixedCloud/20170813_152727.00.010.000_R4.png' #这里选择一张图片用于测试
#path = 'E:/CR雷达回波图像数据/ConvectiveCloud/20170920_140404.00.010.000_R1.png'
#path = 'E:/CR雷达回波图像数据/StratiformCloud/20171109_184141.00.010.000_R2.png'
PngPath = 'E:/CR雷达回波图像数据/StratiformCloud/*.png'
#PngPath = 'E:/CR雷达回波图像数据/e/*.png'
#类别字典
RainEcho_dict={0:'对流性降水',1:'混合性降水',2:'非降水回波',3:'层状云降水'}
IndexNum = []
# 读取数据
for PngPic in glob.glob(PngPath):
    image_data = tf.gfile.FastGFile(PngPic, 'rb').read()

    # 评估
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
    
            # 读取训练好的inception-v3模型
            with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
    
            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    
            # 使用inception-v3处理图片获取特征向量
            bottleneck_values = sess.run(bottleneck_tensor,{jpeg_data_tensor: image_data})
            # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
            bottleneck_values = [np.squeeze(bottleneck_values)]
    
            # 加载图和变量（这里我选择的是step=900的图，使用的是绝对路径。）
            saver = tf.train.import_meta_graph('E:/TensorFlowWorkPlace/Transfer_Learning/runs/1553736359/checkpoints/model-9000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('E:/TensorFlowWorkPlace/Transfer_Learning/runs/1553736359/checkpoints/'))
    
            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]
    
            # 我们想要评估的tensors
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]
    
            # 收集预测值
            all_predictions = []
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})
    
    
            # 打印出预测结果
            index=str(all_predictions)[1]
            index=int(index)
#            print(PngPic+' '+'预测为：'+RainEcho_dict[index])
#            
   #给图片添加文字（水印）
    im1 = Image.open(PngPic)
    draw = ImageDraw.Draw(im1)
    font = ImageFont.truetype('simsun.ttc', 24)
    #画图
    draw = ImageDraw.Draw(im1)
    draw.text((160, 0), RainEcho_dict[index], (255, 255, 255), font=font)    #设置文字位置/内容/颜色/字体
    draw = ImageDraw.Draw(im1)

    #另存图片
    im1.save(PngPic)

        # IndexNum.append(index)
        
# 统计元素个数
# set = set(IndexNum)
# dict = {}
# for item in set:
#     dict.update({item:IndexNum.count(item)/len(IndexNum)})
# print(dict)
#
# end = time.clock()
# print("final is in", end-start)
        
        
        