# 自定义数据集及Reader

本文仅介绍一种飞桨在推荐领域内广泛使用的数据存储格式、Reader定义和调试方法，其实飞桨支持各种数据类型并且不限定存储方法，也可参考 https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/data_preparing/static_mode/index_cn.html

本文以CTR-DNN模型为例，给出了从数据集整理，变量定义，Reader写法，调试的完整历程。


## 数据集整理

我们推荐用户将数据预处理为`slot:feasign`这种格式，每行样本以`slot:feasign`存储，以`\n`作为每行的结尾，可存储为明文或者是tgz的形式。

> Slot : Feasign 是什么？
>
> Slot直译是槽位，在推荐工程中，是指某一个宽泛的特征类别，比如用户ID、性别、年龄就是Slot，Feasign则是具体值，比如：12345，男，20岁。
> 
> 在实践过程中，很多特征槽位不是单一属性，或无法量化并且离散稀疏的，比如某用户兴趣爱好有三个：游戏/足球/数码，且每个具体兴趣又有多个特征维度，则在兴趣爱好这个Slot兴趣槽位中，就会有多个Feasign值。
>
> PaddleRec在读取数据时，每个Slot ID对应的特征，支持稀疏，且支持变长，可以非常灵活的支持各种场景的推荐模型训练。

## 数据格式说明

假如你的原始数据格式为

```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```

其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。
并且每个特征有一个特征值。
```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔。

假设这13个连续特征（dense slot）的name如下：

```
D1 D2 D3 D4 D4 D6 D7 D8 D9 D10 D11 D12 D13
```

这26个离散特征（sparse slot）的name如下：
```
S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S20 S21 S22 S23 S24 S25 S26
```

那么下面这条样本（1个label + 13个dense值 + 26个feasign）
```
1 0.1 0.4 0.2 0.3 0.5 0.8 0.3 0.2 0.1 0.5 0.6 0.3 0.9 60 16 91 50 52 52 28 69 63 33 87 69 48 59 27 12 95 36 37 41 17 3 86 19 88 60
```

可以转换成：
```
label:1 D1:0.1 D2:0.4 D3:0.2 D4:0.3 D5:0.5 D6:0.8 D7:0.3 D8:0.2 D9:0.1 D10:0.5 D11:0.6 D12:0.3 D13:0.9 S14:60 S15:16 S16:91 S17:50 S18:52 S19:52 S20:28 S21:69 S22:63 S23:33 S24:87 S25:69 S26:48 S27:59 S28:27 S29:12 S30:95 S31:36 S32:37 S33:41 S34:17 S35:3 S36:86 S37:19 S38:88 S39:60
```

注意：上面各个slot:feasign字段之间的顺序没有要求，比如```D1:0.1 D2:0.4```改成```D2:0.4 D1:0.1```也可以。



关于数据的tips：
1. 数据量：

    飞桨的分布式训练专门面向大规模数据设计，可以轻松支持亿级的数据读取，工业级的数据读写api：`dataset`在搜索、推荐、信息流等业务得到了充分打磨。
2. 文件类型:

    支持任意直接可读的文本数据，`dataset`同时支持`.gz`格式的文本压缩数据，无需额外代码，可直接读取。数据样本应以`\n`为标志，按行组织。

3. 文件存放位置：

    文件通常存放在训练节点本地，但同时，`dataset`支持使用`hadoop`远程读取数据，数据无需下载到本地，为dataset配置hadoop相关账户及地址即可。
4. 数据类型

    Reader处理的是以行为单位的`string`数据，喂入网络的数据需要转为`int`,`float`的数值数据，不支持`string`喂入网络，不建议明文保存及处理训练数据。
5. Tips

    Dataset模式下，训练线程与数据读取线程的关系强相关，为了多线程充分利用，`强烈建议将文件合理的拆为多个小文件`，尤其是在分布式训练场景下，可以均衡各个节点的数据量，同时加快数据的下载速度

## 在模型组网中加入输入占位符

Reader读取文件后，产出的数据喂入网络，需要有占位符进行接收。占位符在Paddle中使用`fluid.data`或`fluid.layers.data`进行定义。`data`的定义可以参考[fluid.data](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/data_cn.html#data)以及[fluid.layers.data](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/data_cn.html#data)。

加入您希望输入三个数据，分别是维度32的数据A，维度变长的稀疏数据B，以及一个一维的标签数据C，并希望梯度可以经过该变量向前传递，则示例如下：

数据A的定义：
```python
var_a = fluid.data(name='A', shape= [-1, 32], dtype='float32')
```

数据B的定义，变长数据的使用可以参考[LoDTensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#cn-user-guide-lod-tensor)：
```python
var_b = fluid.data(name='B', shape=[-1, 1], lod_level=1, dtype='int64')
```

数据C的定义：
```python
var_c = fluid.data(name='C', shape=[-1, 1], dtype='int32')
var_c.stop_gradient = False
```

至此，我们完成了在组网中定义输入数据的工作。


# TODO
Reader写法及调试
