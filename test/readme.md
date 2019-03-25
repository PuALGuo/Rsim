我怕我忘了 做个备注

bogus_vgg 指的是keras给出的那份类似vgg的样例 
my_bogus_vgg_weight bogus_vgg的权重训练结果   //精度76%左右
bogus_vgg_2 即为bvgg，对原来的bogus_vgg做了深度和训练数据的迭代生成的改进，激活函数为elu
model bvgg的权重训练数据 // 精度为90%左右
bogus_vgg_3 为进行实验模拟对bvgg进行修改，去除了batchnormalization()，将激活函数改为tanh，结果收敛速度骤降，在数据增强后的结果中用原数据模拟出现了loss回升的问题，难以下降（可能是momentum的问题），简而言之，实验结果令人很不满意
bvgg_weight_2 bvgg_3的训练结果 // 精度为79%左右
//接下来还是用原来的bvgg训练吧，将数据进行归一化进行计算
//batchnormalization()还要不要？
//l2范式的正则化好像是必要的，回想起自己vgg的疯狂过拟合
//数据的拟定重训练用原数据吧，就不要用数据增强了
bvgg_4 添加了一层dense，添加了归一化，去除了最开始的z-score改为最为原始的归一化
//算了 不加dense了，巨他妈多的训练权重，爆炸
bvgg_4_1.h5  bvgg_4的训练结果 //精度83%左右
train loss:0.6209 acc:83.25%
test  loss:0.6294 acc:83.21%
//83这道坎我觉得过不去了  真的悲惨
bvgg_4_2.h5  bvgg_4的训练结果 迭代250次 //精度84%左右
train loss:0.5989 acc:84.01%
test  loss:0.6158 acc:84.13%
//滚回原数据，用sgd进行训练，貌似会过拟合，val_loss波动剧烈，落泪了
//放弃滚回原数据，在原有数据增强下用sgd进行优化，期望能有更好结果
//突破84大关，来到了85，落泪了，希望不是过拟合
bvgg_4_3.h5
train loss:0.5390 acc:85.59%
test  loss:0.5869 acc:84.86%
//fighting!!!!!!
//失败了，并不能带来提高
bvgg_4_4.h5
train loss:0.5560 acc:85.24%
test  loss:0.5918 acc:84.80%
//陷入僵局
//干 我知道我错在哪里了  最大最小归一化写错了
