# 深度残差神经网络

## 摘要
随着深度神经网络的深度增加,训练难度也越来越大,这篇文章提出了一种方法,在增加网络深度的同时,不会增加网络的计算复杂度,并且增加网络的精度.

作者将网络的多个layer(2个或者3个),构建成一个关于这些layer的输入x的一个残差函数,这样整个网络变成由若干个残差块(多个普通的Layer)构成的网络,整个网络学习的就是关于这些显式定义残差函数的学习,而不是传统的非引用式学习.论文中的这个句子看了半天才明白.
> We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. 

剩下的部分,作者开始长篇大论的讲解并证明,这种使用残差块的方式与原始的方式是等价的,而且效果更好.

## 简介
首先,肯定一下神经网络的作用,在图像处理领域,特别是在图像分类,识别,检测方面都有很多突破性进展,神经网络,把提取特征(低级,中级,高级),分类等任务放在一个端到端的网络中,并且,随着网络层次的增加,提取的特征越高级.

但是神经网络存在的问题也暴露出来,就是随着网络深度的增加,梯度会消失. 这种问题,虽然可以通过在卷基层与激活函数之间增加BN来解决(或者增加正则化层).但是随着网络层次深度的增加,会出现误差增加的的现象,而且不幸的是这种问题不是由过拟合导致的.

误差曲线如下图所示
![]()

为了解决这个问题,这篇文章提出了一种残差学习的框架: 之前几个层叠的网络是直接映射(潜在的一个期望映射),这里是把这些层叠网络层进行残差映射. 比如定义之前这几个网络层的映射是 H(x),其中x是这几个网络层的输入,把这几个网络定义成残差映射就是F(x)=H(x)-x.那么原始的潜在映射就是H(x) = F(x) + x.

这个新的原始映射H(x) = F(x) + x.可以通过一个带有shortcut connection的前馈网络来实现.这里所谓的shortcut connection是跳跃几层的网络连接方式.结果如下图所示:
![]()

这个shortcut connection就是所谓的“同等映射”，同等映射的输出与之间的层叠网络的输出加在一起。跟上面图显示的一致。添加这个同等映射，并没有增加参数，也没有增加计算复杂度，其训练过程还是可以使用SGD进行优化，同样使用现在的深度学习工具进行开发。

1. 与同等深度的“plain”网络进行对比，误差很小
2. 即使深度增加很多，相对于之前的网络，精度要比其他的网络更好

作者在ImageNet、MSCOCO、CIFAR上进行测试，效果很好。

## 残差神经网络

### 残差学习

用H(x)表示一个层叠的网络层（一般是几层，比如2层，3层等，不会是整个网络），x表示这些层叠在一起的网络的输入数据--这是问题的基本形式。

然后是一个假设：如果多个非线性的网络层，能逐渐的逼近一个复杂的函数，那么一个同等的假设，多个非线性的网络层，能逐渐的逼近一个残差，比如上面提到的H(x)-x（这里是简写，假设输入和输出有相同的维度）。

根据这个假设，我们不在期望，这些层叠一起的网络逐渐的逼近一个复杂的函数输出H(x)，而是，期待它能逐渐逼近一个残差F(x)=H(x)-x。那么这个层叠网络的原始逼近的函数H(x)=F(x)+x。虽然逼近效果是一样的，但是学习的效果是不一样的。

在文章还有这样一段话：
> As we discussed in the introduction, if the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart. The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers. With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

大概的意思是：如果增加深度，把增加的这些网络层叠组成一个同等映射，这个同等映射的训练误差要比对应的副本误差要小。在之前的网络训练过程出现的degradation现象说明，用这些层叠的网络逼近同等映射是困难的，但是通过定义残差，网络学习时优化同等映射，使得同等映射达到最优，那么solver就会驱动这些层叠网络的权重逼近0，来逼近同等映射。

### 利用shortcut connection实现同等映射
通常会对少量的网络层进行残差学习，这些网络层就称作是残差块。一个残差块的定义如下：

$$ y = F(x, {W_i}) + x $$

对于含有两个网络层的残差块定义
$$F = {W_2}\theta({W_1}x)$$

上面公式都是假设F(x)与x的维度是相同，但是不同的时候，就需要对x进行一个映射。

### 网络结构
比较网络参数计算量优化

#### plain 网络
卷积网络层的卷积核大小是3*3,降采样的规则:
1. 如果输出相同的feature map,那么x下面一层还是具有相同的滤波器;
2. 如果feature map的大小折半,那么滤波器的数量double.

这些规则是给予VGG网络.

#### 残差网络
如上图的右侧,如果identity map的尺寸不一致了,有两种方案,一种是扩充填0;另外一种方案是使用映射矩阵映射.

### 实验
1. 图像预处理,选择较短的边进行降采样,并随机的从其中选择一个224*224的warp
2. 像素值进行减除均值处理,(相等于正则化)
3. 在每个卷积层之后,激活函数之前,实行BN
4. 参数进行随机初始化
5. 优化方法是使用的SGD,最小batch大小是100
6. 学习率是0.1,如果遇到错误停滞,学习率再除以10;
7. 训练总迭代次数是600000次
8. 权重衰减率是0.0001;
9. 动量是0.8
10. 没有使用droout