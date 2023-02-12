import os
import random

trainval_percent = 0.8  # 所有数据中用来训练的比例 trainval/all  trainval=train+val
train_percent = 0.9     # trainval用来训练的比例 train/trainval
xmlfilepath = './VOCdevkit/VOC2007/Annotations'
txtsavepath = './VOCdevkit/VOC2007/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open(txtsavepath+'/trainval.txt', 'w+')	# 训练集数据+验证集数据
ftest = open(txtsavepath+'/test.txt', 'w+')	# 测试集
ftrain = open(txtsavepath+'/train.txt', 'w+')	# 训练集
fval = open(txtsavepath+'/val.txt', 'w+')	# 验证集

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print('Finished!')

