# YOLOv1 你只需要看一眼

这应该是第一个在论文中提出,把目标检测当做一个回归问题,没有region proposal和分类器,也是第一个实时的目标检测方法.

YOLO v1有很多缺陷,但是它的地位还是很高.下面是YOLO 检测目标的pipeline,很简单.

![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/yolov1/1.png)

## 检测模型

YOLO是将整个图像输入到网络中,进行端到端的检测.

具体做法写的很形象,但是在网络中使用全连接层体现的,刚开始看论文的时候,就怀疑他是怎么实现的.

将整个图像分割成S\*S的网格,每一个网格负责检测中心点在这个网格中的物体.如下图所示:

每个网格检测B个BoundingBox和这些个BoundingBox的置信分值,置信分值计算是当前BoundingBox的存在目标的概率与IOU的乘积;
对于每个BoundingBox,预测其四个坐标值;
对于一个cell,还要预测一个概率向量,是当前cell中目标是所有类别的概率值.

每个BoundingBox的置信分值计算公式如下(如果没有目标在当前的cell,那么这个分值就是0):

![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/yolov1/2.png)

每个cell计算概率分值计算公式如下:

![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/yolov1/5.png)

在测试阶段,把两者相乘,就会得到具体一类的分值

关于输出.在VOC数据库上网格大小是7\*7,并且每个cell预测2个BoundingBox,这样输出应该是一个7\*7\*(20 + 2\*(4+1))的一个张量,但是由于网络是全卷积层,所以是一个向量的形式.

## 网络

网络结构是给予Google Net但是做了修改.结构图如下所示
![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/yolov1/3.png)

在后面添加了4个卷积层,和两个全连接层.最后的全连接层用来预测概率和BoundingBox的4个方位.

## 目标函数

目标函数包含两个部分,BoundingBox 方位损失和概率损失,如下图所示.

![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/yolov1/4.png)

那个类似与数字"1"的函数是只是函数,表示在第一个cell中的第j个BoundingBox是否存在一个object;
这个目标函数的前两行分别表示中心点误差和长宽的误差,增加权重,用来权衡坐标误差与概率误差.
第三行和第四行表示置信分值的误差,其中第三行表示有目标的时候的损失,第四行表示没有目标的损失,没有目标的损失降低了权重系数,降低没有目标置信分值为0的,对网络学习的影响.由于置信分值的计算是包含IOU的计算的,所以这里也可以看做是IOU的损失.
最后一行表示概率损失,这设一个向量的比较.


## 缺陷

1. 由于每个cell预测多个BoundingBox,可能导致一个目标会被多个BoundingBox(都在一个cell内)预测,但是只需要一个,所以作者就选择IOU最大的那个来预测. 这一点就不如SSD,SSD是每个目标可能会有多个default Box匹配.

2. 由于每个cell预测多个BoundingBox,但是每个cell只能预测一类,当有多个目标在同一个cell中,就有可能造成其他的类别检测不出来

3. 没有对不同宽高比的考虑

4. 虽然把场合宽都映射到0到1之间了,但是大的物体的BoundingBox和小的物体的BoundingBox之间还是有较大的差异,用相同的权重存在不均衡的问题, 比如相同的误差,对于打的物体,这点误差不算啥,但是对于小的物体,误差就很大了.,这一点在SSD中是用的相对比值,能很大成都上避免这样的问题.



