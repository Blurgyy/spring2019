# A Digit Recognizer Based on Linear Classifier (MNIST Dataset)







| 姓名 | 张高阳 |
|:---:|:---:|
| 学号 | 16020021054 |
| 专业 | 电子信息科学与技术 |
| 日期 | 2019 年 6 月 19 日 |

<div style="page-break-after:always"></div>

## Table of Contents

[TOC]

## 1. Abstract
本文是「数字图像处理」课程的课程设计报告。本文通过线性分类器实现 `MNIST` 手写数字数据集的识别，并试图达到较高的准确度。

## 2. Background
### 2.1 MNIST 数据集
`MNIST` 手写数字数据集[^mnist]中，训练集含有 `60,000` 张图片，测试集有 `10,000` 张图片，每个手写数字都被规范化到一个 $28\times 28$ 的图像的中心。

### 2.2 图像分类
图像分类任务，就是对于一个给定的图像，预测它所属于的那个分类标签，或是给出该图像属于一系列不同标签的可能性。由于现实中图像的复杂性，图像分类算法显然不能通过在代码中直接写明各类物体看起来是什么样的来实现，它的实现方法应当是数据驱动的，即通过给计算机很多数据，然后实现学习算法，让计算机自行学习如何区分每个类别。

### 2.3 线性分类器 (Linear Classifier)
线性分类器由两部分组成：**评分函数(score function)** 和 **损失函数(loss function)**，这种方法可以自然地延伸到神经网络和卷积神经网络上。其中， **评分函数** 可以看作是原始图像数据到各类别分数值的映射，而 **损失函数** 是用来量化预测分类标签的得分（**评分函数**）与真实标签的一致性的。线性分类器的训练可以转化为一个最优化问题，在最优化过程中通过更新**评分函数**的参数来最小化**损失函数**的值。

### 2.4 损失函数 (Loss Function)
本文使用 $softmax$ 函数与交叉熵（ $cross\ entropy$ ）损失函数及 $L_2$ 正则损失（最小平方误差， $LSE$ ）函数作为即将实现的线性分类器的损失函数。

输入数据 $x$ 通过线性分类器 $f(*)$ 得到 $score=f(x)$ ，这里的 $score_i$ 即为线性分类器对第 $i$ 类打出的分数。 $softmax$ 函数取 $score_i$ 作为其输入，输出 $s_i=e^{score_i}$ ，这样输出的向量 $\{s_i\}$ 可以看作是未经归一化的对数概率。交叉熵损失函数取这些 $s_i$ 作为输入，输出 
$$
L = -log(\frac{s_{y_i}}{\sum_{j=1}^{n}s_j})
$$
作为线性分类器的打分的损失函数。其中 $y_i$ 为正确分类， $n$ 为种类总数。

### 2.5 梯度下降 (Gradient Descent)
梯度下降法是一种最优化算法，通过求出目标函数在当前点的梯度，然后沿梯度反向规定步长进行迭代，可以找到目标函数的一个局部最优。

### 2.6 反向传播 (Backpropagation)
反向传播是神经网络和深度神经网络的基础，通过与梯度下降法结合使用，可以计算出损失函数关于网络中各部分的梯度。反向传播的基本思想是对网络中各部分使用微积分的链式法则，即对于函数 $f(u(x))$,有
$$
\frac{df}{dx}=\frac{df}{du}\times \frac{du}{dx}
$$



## 3. Algorithm Description

### 3.1 概述
定义线性映射 $f(w,x)=wx$ ，取图像 $x$ 作为输入。此处的输入图像大小为 $28\times 28$ ，预处理图像为一个 $784\times 1$ 的向量，要通过一次线性变换做出十个类别的打分，则映射中的参数 $w$ （即权重）应为一个 $10\times 784$ 的矩阵。本文使用的损失函数是 $softmax$ 函数结合交叉熵损失加上正则化损失，通过最小化这个损失函数，每次沿梯度反向更新权重 $w$ ，从而达到学习权重的目的（训练过程对同一个训练集打乱顺序进行多次训练，以使权重收敛）。检测过程使用训练出的权重矩阵 $w$ 对输入图像进行打分，得分最高的类别即为线性分类器的预测结果。

### 3.2 反向传播梯度的推导
打分函数和损失函数式：
$$
f = wx
$$
$$
L = -log(\frac{f_{y_i}}{\sum_{j}e^{f_j}})
$$
要根据梯度更新权重 $w$ ，则要求的是 $\frac{dL}{dw}$ ，分两步求导。
####  计算 $\frac{dL}{df}$ 
首先有
$$
\frac{dL}{df}=[\frac{dL_1}{f_1}, \frac{dL_2}{f2},...,\frac{dL_n}{f_n}]
$$
而上面的损失函数有等价形式：
$$
L=-f_{y_i}+log(\sum_{j}e^{f_j})
$$
则有
$$
\frac{dL}{df}=\begin{cases}\\
-1+\frac{e^{f_j}}{\sum_{k}e^{f_k}},     & {j=y_i}\\
\\
\frac{e^{f_j}}{\sum_{k}e^{f_k}},        & {j\neq y_i}\\
\end{cases}.
$$
####  计算 $\frac{df}{dw}$ 
由于 $f=wx$ ，则有
$$
\frac{df}{dw_k}=x_{k,1}
$$
其中 $k$ 为 $w$ 的列标，即 $\frac{df}{dw}$ 在 $w$ 的同一列上有相同的值。

完成以上两步后，即可得到**真实的梯度**为
$$
\frac{dL}{dw_k}=\frac{dL}{df}\frac{df}{dw}=\begin{cases}\\
-x_{k,1}+\frac{e^{f_j}}{\sum_{k}e^{f_k}}x_{k,1}, & {j=y_i}\\
\\
\frac{e^{f_j}}{\sum_{k}e^{f_k}}x_{k,1},    & {j\neq y_i}\\
\end{cases}
$$

### 3.3 代码实现的注意事项
在求 $softmax$ 函数梯度时，要求 $\frac{e^{f_j}}{\sum_{j}e^{f_j}}$ ，这时由于分子和分母求的是指数，非常容易造成溢出，直接计算的数值不稳定。为了保持精度，根据恒等变换
$$
\begin{aligned}
&&&\frac{e^{f_j}}{\sum_{j}e^{f_j}}\\
=&&&\frac{Ce^{f_j}}{C\sum_{j}e^{f_j}}\\
=&&&\frac{e^{f_j+logC}}{\sum_{j}e^{f_j+logC}}\end{aligned}
$$
取 $logC=-\max_j\{f_j\}$ ，可以使得 $e$ 的指数最大值为 $1$ ，从而将向量 $\{e^f\}$ 归一化到区间 $[0,1]$ 上，实现数值稳定。

### 3.4 代码

代码实现过程中，为了防止每次开始训练都要重新读取一遍图片，我先写了一个脚本用来读取图片并把读取在内存中的数据保存在磁盘中，此后每次需要读取训练数据集或测试数据集时将磁盘中预先读取好的数据加载到内存中即可，可以提高训练和测试效率。下面是核心代码（略去了读取数据函数、初始化权重、备份数据等函数）。

使用 `python` 的 `numpy` 库实现。完整代码可以在[**这里**][mygit]查看 

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import sys
import pickle 
import numpy as np 
from PIL import Image 
import random 


def update_weights(w, x, score, learning_rate, backup = False, ):
    fn_name = "update_weights";
    try:
        X = x[0].reshape(1, -1);
        gt = x[1];
        prob = score;
        prob -= np.max(prob);
        prob = np.exp(prob) / np.sum(np.exp(prob));
        dL_dw = np.dot(prob, X);
        dL_dw[gt] = dL_dw[gt] - X;
        w -= learning_rate * dL_dw;
        if(backup):
            backup_weights(w);
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def train(training_imgs, w, learning_rate, ):
    fn_name = "train";
    try:
        YES = 0;
        NO = 0;
        size = len(training_imgs);
        for i in range(size):
            score = np.dot(w, training_imgs[i][0]);
            
            idx = np.argmax(score);
            if(idx == training_imgs[i][1]):
                YES += 1;
            else:
                NO += 1;
            ratio = 100 * YES/(YES+NO);
            if(i % 1000 == 999):
                print("\rtrained \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '     ');
                # print("\rtrained \033[1;34m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%, (%g, %g)" % (YES + NO, YES, NO, ratio, np.amax(w), np.amin(w)), end = '     ');
            update_weights(w, training_imgs[i], score, learning_rate, i % 1000 == 999);
        return YES / size;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def main():
    fn_name = "main";
    try:
        w = init_weights();
        epoch = 100;
        if("--epoch" in sys.argv):
            try:
                epoch = int(sys.argv[sys.argv.index("--epoch")+1]);
            except Exception as e:
                pass;
        print("\033[1;37mstarting training\033[0m: epoch = %d" % (epoch));
        refresh_log();
        for i in range(epoch):
            training_imgs = load_training_set();
            # learning_rate = 1; # constant
            # learning_rate = (epoch - i) / (epoch); # linear
            # learning_rate = 1 / (i + 1); # hyperbola
            learning_rate = 1 / (1 + np.exp(i + 1 - epoch / 2)); # sigmoid #3 best
            # learning_rate = (np.arctan(-(i+1 - epoch/2)) + np.pi/2) / np.pi; # arctan
            print("\ntraining epoch %d/%d with learning_rate=%f" % (i+1, epoch, learning_rate));
            precision = train(training_imgs, w, learning_rate);
            backup_weights(w);
            log_precision(precision);
            print();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));


if(__name__ == "__main__"):
    main();

```



## 4. Conclusion

* 最初使用常数 $l=1$ 作为学习率，最终达到的测试集准确率在 $89%$ 左右；然后尝试让学习率 $l$ 随训练迭代次数增加而下降，分别尝试了常数、线性、反比、$sigmoid$、反正切函数作为 $l$ 的值，实验后发现 $sigmoid$ 函数可以达到最高的测试集准确率。下面是不同的函数在训练过程中的预测准确度的变化曲线：
  
  * 常数：
    $$
    l=1
    $$
    ![constant][constant]
    
  * 线性下降： 
    $$
    l=\frac{(epoch-i+1)}{epoch}
    $$
    ![linear][linear]
    
  * 反比：
    $$
    l=\frac{1}{i}
    $$
    ![hyperbola][hyperbola]
    
  *  $sigmoid$ ：  
    $$
    l=\frac{1}{1+e^{i-\frac{epoch}{2}}}​
    $$
    ![sigmoid][sigmoid]
    
  * 反正切：
    $$
    l=\frac{-(i-\frac{epoch}{2})+\frac{\pi}{2}}{\pi}
    $$
    ![arctan][arctan]
  
  五种训练模型的最终准确率如下：
  
  ||常数|线性下降|反比|$sigmoid$|反正切|
  |:---:|:---:|:---:|:---:|:---:|:---:|
  |训练集准确率|$89.31\%$|$92.64\%$|$91.27\%$|$93.22\%$|$92.58\%$|
  |测试集准确率|$89.83\%$|$90.11\%$|$90.03\%$|$90.63\%$|$89.97\%$|
  
  可见，在测试的五种函数中，如果学习率 $l$ 按 $sigmoid$ 函数规律下降，达到的准确率不论是在训练集还是在测试集上都是最优的。
  
  > 在实际训练过程中观察到，学习率根据 $sigmoid$ 函数下降的训练模型仅在第 $50$ 次迭代左右下降明显，在前 $35$ 次迭代时的学习率几乎等于 $1$ ，并且在第 $65$ 次之后，学习率几乎变为 $0$ ，此后的预测准确率都不再变化。
  
* 线性分类器通过给输入图像的每一个像素分配不同的权重，对不同的可能类别进行分别打分而达到图像分类目的，这种方式可以自然地推广到比手写数字识别更加复杂的图像分类任务上，但是由于现实生活中大多任务都不是线性的，而线性分类器只能作出线性变换，所以其准确率并不能达到很高。

* 线性分类器可以被用于神经网络中，叫做全连接层 (Fully Connected Layer, FC) ，卷积神经网络的最后一层大都是全连接层，多层全连接层配合非线性的激活函数理论上可以模拟任何非线性变换。[^FC]





[^mnist]: [<font face="courier" color="black">http://yann.lecun.com/exdb/mnist/</font>](http://yann.lecun.com/exdb/mnist/)
[^FC]:全连接层的作用： [<font face="courier" color="black">https://www.zhihu.com/question/41037974/answer/150522307</font>](<https://www.zhihu.com/question/41037974/answer/150522307>)
[mygit]: https://github.com/Blurgyy/spring2019/tree/master/DIP/mnist


[^constant]: $l=1$ 
[^linear]: $l=\frac{(epoch-i+1)}{epoch}$ 
[^hyperbola]: $l=\frac{1}{i}$ 
[^arctan]: $l=\frac{-(i-\frac{epoch}{2})+\frac{\pi}{2}}{\pi}$ 
[^sigmoid]: $l=\frac{1}{1+e^{i-\frac{epoch}{2}}}$ 


[constant]: http://106.14.194.215/imghost/mnist_linear_classifier/constant.png	"Constant"
[linear]: http://106.14.194.215/imghost/mnist_linear_classifier/linear.png	"Linear"
[hyperbola]: http://106.14.194.215/imghost/mnist_linear_classifier/hyperbola.png	"Hyperbola"

[sigmoid]: http://106.14.194.215/imghost/mnist_linear_classifier/sigmoid.png	"Sigmoid"
[arctan]: http://106.14.194.215/imghost/mnist_linear_classifier/arctan.png	"Arctan"

