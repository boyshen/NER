## 简述

中文细粒度命名实体识别。识别文本中的人名、地名、机构名、游戏、电影、景点等多个实体。

模型采用 BiLSTM + CRF

参考：

● [中文命名实体识别总结](https://www.jianshu.com/p/34a5c6b9bb3e)

● [中文细粒度命名实体识别](https://zhuanlan.zhihu.com/p/103034432?utm_source=wechat_session)

● [NLP-中文命名实体识别](https://blog.csdn.net/MaggicalQ/article/details/88980534)

● [命名实体总结](https://www.cnblogs.com/nxf-rabbit75/archive/2019/04/18/10727769.html)

### 数据集

项目数据集来自 [中文细粒度命名实体识别数据集](https://www.cluebenchmarks.com/introduce.html)。

该数据集主要包括 train.json、test.json、dev.json。

● train.json 训练数据集。包含text、label。可以进行训练

● test.json 测试数据集。该数据集没有提供 label、无法进行评分。详细参考[官网](https://www.cluebenchmarks.com/introduce.html)

● dev.json 验证数据集。包含 text 和 label。可以进行测试、验证。

项目中采用 train.json 做训练和验证数据集。dev.json 做测试数据集。数据集中包括多个实体，每个实体的语料数量各不相同。

标签类别：
数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

### 评估

使用 dev 数据集进行评估。

|              | RECALL | PRECISION | F1-score |
| ------------ | ------ | --------- | -------- |
| name         | 0.4710 | 0.7349    | 0.5740   |
| organization | 0.4796 | 0.8186    | 0.6048   |
| game         | 0.6034 | 0.7982    | 0.6873   |
| book         | 0.4545 | 0.7955    | 0.5785   |
| position     | 0.5012 | 0.8750    | 0.6373   |
| government   | 0.4130 | 0.8095    | 0.5469   |
| movie        | 0.6490 | 0.7481    | 0.6950   |
| address      | 0.2359 | 0.5605    | 0.3321   |
| company      | 0.4815 | 0.8198    | 0.6067   |
| scene        | 0.2679 | 0.6914    | 0.3862   |

### 使用

1. 环境

   安装 pytorch  > 1.1.0  。其他相关包参考 requirements/requirement.txt

2. 配置

   ```
   {
     "PATH_DATA_TRAIN": "/data/cluener_public/train.json",
     "PATH_DATA_DEV": "/data/cluener_public/dev.json",
     "PATH_DATA_TEST": "/data/cluener_public/test.json",
     "PATH_WORD_DICT": "/NER/model/BiLSTM_CRF_torch/word_dict.pickle", # 词典索引保存路径
     "MODEL": {
       "embedding_dim": 128,
       "hidden_size": 384,
       "learning_rate": 0.001,
       "weight_decay": 1.0e-4,
       "dropout": 0.5,
       "batch_size": 4,
       "shuffle": true,
       "gradient_normal": 5,
       "epochs": 3,  # 训练轮次
       "validation_rate": 0.1,  # 验证数据集比例，从训练数据集中划分0.1比例的数据作为验证集
       "factor": 0.2,  # 指定学习率衰减因子，在验证损失没有减少的情况下减少学习率。new_lr = factor * lr
       "patience": 2,  # 在指定的评估指标2次没有变化之后，修改学习率
       "eps": 1.0e-8,  # 应用于lr的最小衰减。如果新旧lr之间的差异小于eps，则忽略该更新
       "checkpoint": "/NER/model/BiLSTM_CRF_torch/BiLSTM_CRF.pth", # 权重保存
       "config": "/NER/model/BiLSTM_CRF_torch/BiLSTM_CRF.json" # 模型配置信息保存
     }
   }
   ```

3. 执行

```
# 训练：使用训练数据集进行训练
    example: python run.py train

# 评估：将使用 dev 数据集进行评估。输出Recall、Precise、F1-score
    example: python run.py eval

# 测试：将使用测试数据集，将测试数据集识别结果写入文件
    example：python run.py test --file "predict_result.json"
            --file          指定写入的文件名
# 预测：
    example: python run.py prediction --sentence "《加勒比海盗3：世界尽头》的去年同期成绩死死甩在身后，后者则即将赶超《变形金刚》，"
            --sentence      指定预测的句子
```

### 总结

1. 根据评估指标。可以看出实体 position 的 F1-score 得分最高，实体 scene 的 F1-score 评分最低。这里一方面是由于 position 实体数据比较多，而 scene 数据实体比较小。

   可由下看出训练数据中实体数量：

   ```
   name : 2847
   company : 2215
   game : 1897
   organization : 1894
   movie : 779
   address : 2090
   position : 2464
   government : 1461
   scene : 946
   book : 908
   ```

提高F1-score一方面可以增加数据，一方面可以尝试使用其他模型。

