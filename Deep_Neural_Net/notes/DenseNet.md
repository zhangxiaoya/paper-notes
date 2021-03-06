# 稠密链接卷积神经网络

## 摘要

为了一个目的:越深越好,而且还要效率高.

于是就提出了一个"全连接"的卷积神经网络,由于是"全连接",链接很稠密,所以就起了个"DenseNet"名字.

这里的"全连接"是指的层与层之间是全连接(实际上是半全有向,有向无环图),前面的所有层的输出都连接到当前层,当前层的输出后面所有的层.

输入层不算,后面的是L层.总数是(L*(L+1))/2连接. 

总之,这个DenseNet有几个优点:
1. 缓解了权重消失的问题
2. 强化了特征传递
3. 特征重用
4. 大大降低了参数总量

## 简介

一句话卷及神经网络很好.

但是,之前的网络大多数都是逐层传递的,存在梯度消失的问题.为了解决这个问题,产生了比如ResNet这样的神奇设计--隔层传递.然后还有好多这样的设计,总结起来就是"抄近路传递特征".

DenseNet比他们所有的算法都狠--"全连接"(有向无环图的方式),即便是这么密集的链接,DenseNet仍然有两个优势:
1. 与想象的恰恰相反,计算的参数反而更少
2. 所有层的信息在整个网络传递,信息更多.

下面是一个简单的DenseNet示意图

![](https://github.com/zhangxiaoya/paper-notes/blob/master/Deep_Neural_Net/notes/DenseNet/1.png)

上图示意的网络中的很多标注,在后续的内容有介绍,比如学习率k是个什么鬼.

## 相关工作

列举了一些类似使用"抄近路"方式的网络

1. 以前的一些不是卷积神经网络的多连接方式
2. Highway Network
3. ResNet
4. GoogleLenet
5. Network In Network
6. Deeply Supervised Network (DSN)
7. Ladder Networks 
8. Deeply-Fused Nets (DFNs)

## DenseNet

这里需要注意的是, ResNet是把从前面抄近路传递过来的feature,直接与当前层输出的feature进行加运算.而DenseNet是把之前所有的feature连接起来,组成一个feature map,然后像普通的网络层一样.
一些表示:
1. 用Hl(.)表示第l层的非线性映射.
2. 用xl表示低l层的输出;
3. xl = Hl(xl-1)
4. Hl() 表示了多种操作的集合,比如BN,ReLU,Polling, Conv

在ResNet中是这样做的
xl = Hl(xl-1) + xl-1

是加法操作,容易损失信息.
在DenseNet中是这样做的
xl = Hl([x0, x1, ... , xl-1])

从上面的公式能明显看出与ResNet的区别.

另外,在这里,Hl()被定义成3个操作的集合:BN+ReLU+Conv
没有Pooling.Pooling有其他用途.

一个很深的网络结构是由若干个Dense 块组成的,整体的网络结构如下图所示
![](https://github.com/zhangxiaoya/paper-notes/blob/master/Deep_Neural_Net/notes/DenseNet/2.png)

块内是"全连接的",块之间是用一种被叫做"transition Layer"连接的.
每个transition Layer是由三个操作构成的:BN+1\*1的Conv + Pooling

**增长率**: 每一层都往后面的所有层传递一个k深度的feature map,这个k就是增长率,在第l层,其输入的feature深度是k0 + k × (l − 1).
作者就是控制这个k很小,才将参数总数降低,这里的k才是12,平时见到的网络都是128开始,然后到512.

**瓶颈层**: 虽然k很小,但是越积累越多啊,于是就产生了瓶颈层,这么个玩意儿,其实就是把3*3的卷积层换成了1*1的卷积层,降低产生的feature map的深度.

**压缩**: 瓶颈层是在一个Dense块中的,这里的压缩是在Dense块之间的transition层. 同样的方法压缩featuremap数量.

**实现细节**:
1. 进入Dense块之前, 用卷积层先做处理
2. Dense块内的卷积层是3\*3的卷积核,用零值填充边缘,保持feature map的大小
3. transition 层用1\*1的卷积层,后面跟一个2\*2的Pooling层
4. Feature Map的大小是32\*32,16\*16,8\*8

训练过程是使用SGD.




