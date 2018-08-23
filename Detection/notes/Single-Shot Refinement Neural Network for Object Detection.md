# 

## 摘要

Faster-RCNN在检测精度上有优势，SSD算法在效率上有优势，RefineNet是把两个算法的有点合并在一起，并克服两个算法的缺点，既要检测精度高，又要高效。

在RefineNet中，有两个联系的模块，分别是anchor refine 模块和object detection 模块：

Anchor Refine 模块有两个作用：过滤太多的没有用的anchor，降低检索空间；另外，粗略的调整anchor的位置和尺度。
Object Detection模块，在anchor refine的基础上，位置回归和类别预测。

同时，RefineNet另外一个重要贡献是有一个transfer connection block，用来把anchor refine的特征转换成object  detection阶段的位置预测、尺度、类别。整个网络可以实现端到端的训练。

## 简介
在深度神经网络结构中，目标检测算法取得了很大的进展。目前，常用的目标检测算法分为两类：基于两步法和一步法，两步法中，以Faster RCNN作为代表，第一步选出候选的目标，第二步对这些目标进行回归和分类。两步法的精度要比较高一些。
一部分以SSD和YOLO为代表，其优势是效率高，使用方便，但是存在正负样本不平衡的问题导致精度不高。

后续有人针对一步法的正负样本不平衡问题，来解决一步法精度不够的问题。

作者认为两步法的算法有三个主要的优点：1. 使用采样启发式的方法来解决正负样本不平衡问题；2. 使用级联的方式回归目标BBox；3. 使用两步特征（对于RPN，是二分类特征，判断是否是目标，对于检测，是多目标特征，判断目标类型）。

RefineNet是对两种类型目标检测算法的改进，一方面，结合两种算法的优势，另一方面，克服两种算法的不足。

RefineNet有两个个部分组成： ARM，anchor refine module；ODM， Object Detection Module；前者负责anchor box的预处理，后者在此基础上进行检测，看上去还是两步法的操作，但是refineNet使用了TCM transfer connection block，把两者联系在一起。在cost function使用了multi-task方式，最终训练过程可以做到端到端。

这两个模块和两个模块之间的连接是RefineNet最大的创新。

## 相关研究
1. 经典的方法都是基于滑动窗口的，密集的采集样本，提取人工设计的特征，比如haar特征，然后训练分类器，准确性还行，效率一般；后来有DPM，能够使用不同的尺度，这种方法持续了一段时间，但是在CNN出现之后，这些方法有些黯然失色了。

2. 两步法，第一步Region Proposal，有很多种方法，比如Selective Search，EdgeBoxs, DeepMask, RPN等，第二步，进行位置回归和多分类。典型的代表有Faster-RCNN等

3. 一步法，就是把Region Proposal和回归分类都放在一个网络完成，真正实现端到端，典型的代表有YOLO和SSD，YOLOv2是对原始YOLO的改进，添加了BN和Anchor Box，用高分辨率训练，但是精度仍然是个问题，作者认为是正负样本不均衡造成的。

两种算法都有针对性的改进。

## 网络架构

首先ARM对anchor box进行refine，减少负样本，并调整anchor box，然后DPM实现对目标分类和对于refined anchor的偏移回归。

### Transfer Connection Block
ARM与ODM两者结合的网络如上图所示，这个TCB是这样的。

TCM是用来把ARM中的不同的feature map转换成DPM中使用，这样，DPM和ARM实际上是公用卷积特征，具体怎么转换：
1. TCB只对在Anchor Box在ARM的特征进行转换，而不是对整个Feature Map全不转换；
2. 集成大尺度的上下文信息：通过把高一级的特征添加到转换的feature中，增加检测准确率；
3. 为了匹配高一级别的特征与转换的feature之间的维度，对高一级别的特征进行去卷积，增大它的维度，然后与转换后的feature进行元素之间的相加操作；
4. 然后在相加后的特征上添加一个卷积层，增加特征的分类判别能力。

整个TCB的结构如下图所示

![]()

### 两步级联回归
之前的一步法，都是采用一步回归，在这一步回归中，预测目标的位置和尺度。但是这种方法在一些挑战性的环境中，特别是对于小目标，很容易检测不到。RefineNet使用两步级联的方式回归策略，对目标的位置和位置进行回归。

第一级回归：在ARM阶段，对所有的anchor box进行回归，调整位置和尺度，但是是一个粗调整，提供给ODM一个好的初始化。具体的做法有些跟SSD类似，先把每个feature map分割成网格，么个网格有n个anchor box，然后对于每个anchor box，预测与“平铺的anchor”的偏移，有两个分值用来表示这两个box表达有前景目标的置信度。具体的怎么构建还是看看后面的cost function。
经过第一级回归，每个网格产生n个refined anchor box。

第二级回归：经过ARM进行回归后的每个refined anchor box传递给ODM，ODM进行分类和精确位置尺度回归。每个Anchor Box预测4个坐标信息和c个类别信息。

### 负anchor 滤波
为了提前去除那些被很容易确定为负样本的anchor box，对每个anchor box计算negative 分值，若分值大于一个预先设定的阈值，那么不会传递到ODM，在推理的时候，也不会进行推理。直接忽略掉。

## 训练和推理

### 数据增强
参考文章，使用很多数据增广手段，扩张，反转，切片等等，增强模型的鲁棒性。

### 基础网络
RefineNet使用的是VGG16和ResNet101.

### Anchor设计和匹配

不管基础网络怎么样，feature map的步长分别是8 16 32 64，每个feature map对应一个尺度的anchor，每个anchor还有不同的宽高比（0.5，1，2.0）.

在训练的时候，要用anchor box与ground truth进行匹配，首先按重叠分值最高的选择anchor box， 然后再选择重叠超过0.5的anchor box。

### hard negative anchor选择

跟SSD一样，负样本与正样本比例是3：1，需要注意的是这里是hard  negative anchor，那些明显的负样本直接被去除了。

### 损失函数

损失函数有点类似与SSD，但是还不太一样，也是分了两部分，一部分是优化arm，一部分是优化ODM。如下图

### 优化

主要是一些优化参数的设定，使用SGD优化，有动量和weight decay。

### 推理

利用了NMS。

## 实验

