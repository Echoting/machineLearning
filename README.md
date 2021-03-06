## 机器学习demos

### 运行项目：使用parcel 构建项目

如果未安装parcel

```
npm install -g parcel-bundler
```

parcel构建文件，例如：运行线性回归文件夹的demo

```
 parcel li*/*html
```

parcel支持通配符

### 使用链接
- [playground tensorflow](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.95037&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


### 零星的知识点

神经网络：

1、每个神经元中存储了若干权重，偏置和一个激活函数

2、输入乘以权重加上偏置，经过激活函数就得到输出

3、激活函数用以添加一些非线性变化



什么是神经网络的训练？

1、给大量的输入和输出，算出神经网络里所有神经元的权重，偏置，然后给定新的输入，可以算出新的输出

2、在机器学习里面输入输出被称为特征和标签，大量输入输出被称为训练集



如何训练神经网络：

1、初始化：随机生成一些权重和偏置

2、计算损失：给定特征，计算出标签，得到它与真实标签差的有多远

3、优化：微调权重和偏置，使损失变小

重复第二步和第三步，直到损失尽量小



前向传播与反向传播：

1、前向传播：将训练数据的特征送入网络，得到标签

2、反向传播：计算损失并优化



如何计算损失：

- 使用损失函数

- 本课程涉及的损失函数：均方误差、对数损失、交叉熵....

- 了解原理即可，工作中可以从库函数中调用

    

如何优化

- 使用优化器

-  本课程涉及的优化器：随机梯度下降（SGD）、Adam

- 了解原理即可，工作中可以从第三方库中调用

    

Tensowflow.js是什么？

- 是一个用JavaScript实现的机器学习库

- 可以直接在浏览器和Node.js中使用机器学习技术了

    

Tensowflow.js具体功能

- 运行现有模型

- 重新训练现有模型

- 使用JavaScript开发机器学习模型


