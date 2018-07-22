## 简介

目标检测相对于分类任务更复杂：
1. 目标检测首先要做proposal，计算所candidate
2. 然后根据这些proposal进行location的fine tune。

相对于之前基于卷积神经网络的检测方法，Fast RCNN在单一的训练阶段完成proposal和fine tune。

结果比之前的速度更快（训练和测试），精度更高。

RCNN和SPPnet的弊端：
1. RCNN是分多个阶段：首先，基于卷积网络进行初步分proposal，实际上是fine tune;然后，在这些卷积特征上使用SVM分类，而不是softmax 分类，进行object detector，最后对目标的BBox进行回归，调整好位置。
2. 在训练SVM和BBox回归的时候，对于每一张图像的每一个proposal的特征都要存盘，在时间上和空间上的代价比较大；
3. 在检测的时候，时间代价也很大。

RCNN在每个image的每一个proposal都进行独立的卷积网络，没有共享计算，SPPnet利用的共享计算，所以就快一点。

SPPnet先对正张图像进行卷积，计算出卷积的feature map，在feature map上进行max pooling提取每个proposal的特征，并且进行固定大小的输出，对于不同尺度的输出，构成一个空间金字塔池，因为这样的计算每个feature map，所以训练时间快了100倍，测试速度快了3倍。

SPPnet的缺点是多阶段，提取卷积特征，进行fitting，训练SVM，BBoxfitting。而且特征也要存盘，但是在空间金字塔上的池化层的卷积层不更新。

Fast RCNN的优势：
准确率更高，速度更快；
单一训练阶段；
不存盘；
更新所有的卷积层；

## Fast RCNN
整个网络的输入是整张图像，和object proposal，先对整个图像计算feature map，然后对每个proposal计算一个固定长度的特征向量，这个特征向量输入到一系列的全卷积层，然后有两个分支的输出，一个是计算softmax的概率，一个是坐标值。

实际上的使用ROI Polling 层，取代金字塔Pooling层。
