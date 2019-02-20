## 手撕机器学习

手撕机器学习系列模型，手推公式与numpy实现部分模型。

已实现模型全部使用minist数据集测试验证，并与sklearn做了对比：
<p align = 'center'>
<img src = 'images/accuracy.png' height = '400px'>
</p>

特殊说明：

* 所有模型训练之前都对minist数据做了LDA压缩，为了保证特征分解过程非奇异，对数据加入了少量白噪声，所以每次训练结果都有微小差异；
* 每种模型适用不同的数据维度，例如神经网络、逻辑回归适用于高维数据，而朴素贝叶斯和决策树适用于低维数据，所以每种模型使用的特征维度并不一定一样；
* 同种模型在手撕实现和sklearn之间都使用相同的参数配置，只有优化过程的细节差异。

### Requirements
适用于python2.7与python3.6，依赖包：
- numpy，实现模型结构的主要库
- scipy，稀疏矩阵与部分高级线性算法库
- tqdm，神经网络训练进度条显示

## 推断模型

### 神经网络
numpy手撕神经网络，适用于二分类、多分类，手推梯度公式，支持dropout，使用Adam优化器，可显示训练进度条。
神经网络特性：
* 采用迭代的方式更新参数，复杂度随数据维度线性增加，所以可以适用于高维数据；
* 基于梯度更新的优化方式，所以对数值的尺度比较敏感，所以输入的规范化、参数的初始化、初始学习率的设置影响较大；
* 采用min_batch的方式迭代更新，平衡GD与SGD的优劣，dropout是一种成本很低的高效正则手段，每轮迭代随机更新部分参数。
运行示例：
	python neural_network.py --batch 200 --epochs 10 --dropout 0.5

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/caishiqing/manual/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
