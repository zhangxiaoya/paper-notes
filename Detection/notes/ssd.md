# SSD 目标检测算法

## 摘要
SSD,翻译过来就是瞄一眼,就能用很多个Box检测出图像中所有的物体.不同于以往的RCNN方法,SSD使用一个端到端的神经网络,在基础网络的上,计算出一些feature map,然后把这些feature map进行离散化不同宽高比和尺度的default box,这些default box相当于在Faster RCNN中的anchor Box, 用来做region proposal(功能类似,论文里没有这么说).

相对与其他的方法,SSD还是相对简单,它没有单独的RPN,没有专门的分类器,而是把RPN和分类都在一个网络的一次端到端的传递中做了.因此,它的实现难度和计算难度都相对比较小,在预测的时候,速度也很快,具备实时的效果.

## 简介

之前的方法的套路是这样的,先进行候选区域选择,比如使用选择搜索(SS),或者是边缘回归等方法,产生一些candidate regions;第二步是针对这些candidates进行采样和特征提取;第三步是训练一个高精度的分类器.

这样一个复杂的过程下来,region proposal决定了目标的BoundingBox, 特征选择和分类器决定了目标的置信度.实际的精度并不是很高,而且网络设计复杂,计算量大.
比如在region proposal阶段就消耗了大量的时间, 在Faster Rcnn之前,还是用非卷积网络的来做region proposal,虽然Faster RCNN使用了RPN,提出了用anchor box来解决region proposal消耗时间太多的问题(实际上利用共享卷积特征),但是网络结构还是很复杂,训练比较麻烦,还是不能达到实时的效果.

SSD并没有区分region和proposal,而是把region proposal和classification用一个端到端的训练解决,全部用回归的方式解决定位和分类的问题.这种方式也在YOLO中使用了,应该是YOLO是第一篇文章,提出用端到端的训练,一些解决定位和分类问题,但是YOLO v1存在明显的缺陷(虽然有缺陷,但是仍然具有跨时代的意义,而且这些缺陷在后续的版本中都有解决了).

## 2 SSD算法

首先介绍SSD模型,然后介绍这样的一个Model是如何训练的,然后是SSD的目标函数是怎样的.

### 2.1 SSD的Model

SSD跟YOLO的结构类似,也是在一个基础的分类网络上,添加了自定义的网络,最终网络输出一个预测的BoundingBox,每个Box都有一个对应分类的置信分值,最后用NMS,去除多余的框,得到最终的结果.

与YOLO不同的是,这里有几个特点:
1. 使用多尺度特征图, 与YOLO不同,SSD在基础的分类网络后添加了不同尺度的卷积层,用来计算不同尺度的feature map,以适应检测不同尺度的目标;
2. 最终预测是使用卷积网络,而在YOLO中使用的是全连接层.
3. 使用default box和不同的宽高比. 这里有点结合YOLO的网格思想和anchor box的意思, 把每个feature map也是分割成若干cell,每个cell产生不同宽高比的default box,这样,每个default box是与每个尺度和宽高比一一对应的.

具体的网络如下图所示.
![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/ssd/2.png)

注意这里是跟YOLO v1的网络结构做了对比.在SSD的网络结构中:
1. 前面是给予VGG16的分类网络, 其输出也被用到了的检测了,其往后传递的卷积核大小是(3\*3\*(4\*(class+4))
2. 在分了网络后面又有两个卷基层,分别是卷积核是3*3和1*1,这里的1*1卷积核的深度与前面的3*3的卷积核的深度一样,应该是做了一次非线性映射,这两层的也通过一个卷积核传递后最后的结果了;
3. 接下来就是各种尺度的feature  map 层了.每个feature map提取的时候,先用1*1卷积核进行降维,然后用3*3的卷积核处理,最后用步长为2的pooling层降采样.从图上可以看出,做了4个降维尺度的feature map,每个map的结果直接传递给结果层.

从代码看一下这里是怎么实现的:

from_layer 表示这一层的输入是从哪里来的, 从函数的第一层网络(10*10)的可以看出,是从基础网络的最后一层输出结果来.
outlayer,表示这一层网络的名字,用来给下一层输入.
可以看出,每个特征图都是由两个卷积组成,前一个卷积层是1*1的卷积核,用来进行降维,然后后面的卷积层是3*3的卷积核,这里不能显示的看出使用步长为2进行pooling降采样,但是从参数和函数定义可以看出来.

``` Python
# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net
```

再看看整个网络的输出:
每个feature map是有若干个cell组成的,比如看上面代码的第一个feature map大小是10*10.每个cell其中的default box的位置偏移和对应每一类的概率置信分值.(这里描述的太抽象)

具体来说,就是对于每个位置(cell),有k个default box,预测每个box,相对于原始的default的4个坐标和c个类别.所以对于一个feature map,它的卷积核的深度是(c+4)\*k.最终,每个feature map经过卷积层输出的应该是m\*n\*c+4)\*k.通过这里就明白上面的图中网络的卷积核的深度为什么定义成那个样子.

下面看看 feature map是怎么用的:

先看图
![](https://github.com/zhangxiaoya/paper-notes/blob/master/Detection/notes/ssd/1.png)

在论文中关于每个feature map上使用卷积核处理输出时,这样解释的:
> For a feature layer of size m × n with p channels, the basic element for predicting parameters of a potential detection is a 3 × 3 × p small kernel that produces either a score for a category, or a shape offset relative to the default box coordinates. 

这里不应该是and的意思吗,每个卷积核处理后应该输出位置偏移和置信分值啊,这里为什么是个or呢!!!

在上面的图上发现,a表示对应的GroundTruth,b和c分别表示两个feature map,分别是8*8和4*4,每个cell产生一些不同宽高比的default Box,对于每个default box预测位移和置信值(对于每一类).
在实际的训练过程中,每个default box要与GroundTruth进行匹配,从上面的图看出,在8*8的feature map中,有两个default boxp匹配成功,在4*4的feature map中,有1个default box匹配成功,这些匹配成功的default box当做正样本,其他的default box当做负样本.

最终在损失函数中,有两部分组成,一部分是位置的损失(这一部分算是回归),另一部分是置信参数(softmax回归).

### 2.2 训练模型
SSD与那些需要region proposal的检测算法最关键的不同是,GroundTruth需要与检测算法的输出进行关联.在YOLO,Faster-RCNN和multibox也有需要.一旦确定了这样的关系,那么损失函数就确定了.
另外,在训练过程中,还需要选择default box和尺度用来检测,还有负样本的选择和数据增强.

#### 2.2.1 匹配策略
在训练过程中,需要将default box与GroundTruth进行匹配,只有匹配成功的作为正样本. 这里的匹配策略与multibox算法类似,都是使用jaccard 重叠率,但是在这里,只要覆盖率大于0.5就认为匹配成功,这样可以用多个box进行拟合.

#### 2.2.2 目标函数
用下面的公式表示匹配标志(Github不支持公式了,用图片吧)
![]()
整个目标函数包含两部分,分别表示位置损失和置信分值损失,如下图所示.
![]()
其中N表示所有匹配成功的default Box的数量,如果N等于0,那么损失函数等于0.

对于方位损失,采用的是smooth L1损失,这里与Faster RCNN类似,回归相对于GroundTruth中心点\宽和高.具体的计算公式如下所示:
![]()
对于置信分值的参数损失是使用的交叉熵,具体如下:
![]()

#### 2.2.3 给default Box选择宽高比和尺度

对于尺度的选取,作者从其他的方法那里受到一些启发,这里的尺度设定如下公式所示(假设有m个feature map,每个map的尺度)
![]()

对于不同的宽高比,设置如下
![]()

同事每个default Box的宽和高的计算如下所示
![]()
另外,对于宽高比为1的default的宽和高相等,计算公式如下:
![]()

每个cell的中心点计算公式如下所示,其中f表示feature map的尺寸.



#### 2.2.4 负样本挖掘

对于有很多default box的情况,这时候出现样本不均衡的问题,在SSD的解决方案是按照负样本的置信分值进行从高到低进行排序, 然后选排在前面的样本作为负样本,保证负样本数量与正样本数量是3;1

#### 2.2.5 数据增强

这里是为了使训练的模型更鲁棒性,对数据进行适当的处理,对于每个训练的图像,随机进行下面的处理:
- 用整个图进行训练
- 随机的选择一块,使得交叉覆盖比是0.1,0.3 , 0.5, 0.7, 0.9
- 随机的选择一块

选择的一块大小是原始图像大小的[0.1, 1]倍数之间,宽高比是[0.5, 2].

如果一个物体的中心正好在选择的一块图像中,那么要保证他的整个GroundTruth 部分都在整个块中.

图像要经过尺寸处理, 整理为固定的大小,并以0.5的概率进行水平翻转.

## 3 实验
SSD是在VGG16基础上进行训练的,把VGG16的全连接层6和7都改成了卷基层,把池化层5的参数修改了,,然后移除了dropout层和全连接层8.

在整个基础上进行fine tune,参数设置如下:
1. SGD优化
2. 学习率:0.001
3. 动量:0.9
4. 权重退化率:0.0005
5. batch Size 32

后续对于不同的数据库,这些参数略有调整.

## 最后

总结里面写了算法的贡献和性能,最后还提到了整个网络结构简单,可以与RNN结合,用来解决视频中的目标跟踪问题.


作者好像是对caffe进行大大的修改.
