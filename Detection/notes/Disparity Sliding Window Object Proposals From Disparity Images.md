# 视差滑动窗口：从视差图中进行目标提议

## 摘要


 在目标检测算法中，经常会用到Object Proposal，相当于在进行检测分类之前的“目标提议”，这一步从整张图像进行“Object Investigation”，确定目标的region和目标的location。虽然，在DL大浪潮下，基于DL的Object detection是主流，但是很多DL的方法还是需要进行Object Proposal，经典“两步法”就是先进性Object Proposal，然后进行分类。

 相对于传统的基于形状、边缘、颜色的目标检测方法，对目标进行一次Overlooking能够显著的降低漏检的风险。但是，当object proposal产生大量的“提议”时，对于后续步骤进行的分类运算，增加了相当多的计算成本。

 这篇文章提出的基于视差图的Object Proposal，是使用深度相机的深度图信息，能显著的降低“Proposal”的数量，并且不会对目标识别的准确性造成太大的影响。

