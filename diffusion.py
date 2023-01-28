#!/usr/bin/env python
# coding: utf-8

# <h1>
# 	The Annotated Diffusion Model
# </h1>
#
#
# <div class="author-card">
#     <a href="/nielsr">
#         <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/48327001?v=4" width="100" title="Gravatar">
#         <div class="bfc">
#             <code>nielsr</code>n
#             <span class="fullname">Niels Rogge</span>
#         </div>
#     </a>
#     <a href="/kashif">
#         <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/8100?v=4" width="100" title="Gravatar">
#         <div class="bfc">
#             <code>kashif</code>
#             <span class="fullname">Kashif Rasul</span>
#         </div>
#     </a>
#
# </div>

# <script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>
#
# 在这篇博文中，我们将深入了解 Denoising Diffusion Probabilistic Models 去噪扩散概率模型(也称为DDPM、扩散模型、基于分数的生成模型或简单的[自动编码器](https://benanne.github.io/2022/01/31/diffusion.html))，目前研究人员已经能够在（非）条件图像/音频/视频生成方面取得显著的结果。（在编写本报告时）现有的比较受欢迎的的例子包括由OpenAI主导的[GLIDE](https://arxiv.org/abs/2112.10741)和[DALL-E 2](https://openai.com/dall-e-2/)、由海德堡大学主导的[潜在扩散](https://github.com/CompVis/latent-diffusion)和由Google Brain主导的[图像生成](https://imagen.research.google/)。
#
# 我们将在MindSpore中逐步学习并复现DDPM的原论文([Ho et al., 2020](https://arxiv.org/abs/2006.11239))，本篇博文是基于Phil Wang的[实现](https://github.com/lucidrains/denoising-diffusion-pytorch)，而它本身是基于[原始TensorFlow实现](https://github.com/hojonathanho/diffusion)。请注意，生成建模的扩散概念实际上已经在([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585))中介绍过。然而，直到（[Song et al., 2019](https://arxiv.org/abs/1907.05600))（斯坦福大学）和([Ho et al., 2020](https://arxiv.org/abs/2006.11239))（在Google Brain）才各自独立地改进了这种方法。
#
# 请注意，有关于扩散模型的[几个观点](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw)。本文我们采用离散时间（潜在变量模型）的观点，但请务必查看其他观点。

# 让我们开始探索吧！

# ![Image](https://drive.google.com/uc?id=11C3cBUfz7_vrkj_4CWCyePaQyr-0m85_)
#
#  我们将首先安装并导入所需的库(假设您已经安装了[MindSpore](https://mindspore.cn/install)、dataset、matplotlib以及tqdm)。

# In[1]:


import math
from functools import partial

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from multiprocessing import cpu_count
from download import download

import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, ms_function, Tensor, Parameter, ms_class
import mindspore.common.dtype as mstype
from mindspore.dataset.vision import Resize, Inter, CenterCrop, ToTensor, RandomHorizontalFlip, ToPIL
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out

inf = float('inf')
context.set_context(device_target="GPU", mode=ms.PYNATIVE_MODE)


# ## 什么是扩散模型？
#
# 如果将（去噪）扩散模型与其他生成模型（如Normalizing Flows、GAN或VAE）进行比较，它并没有那么复杂：它们都将噪声从一些简单分布转换为数据样本。这也是从纯噪声开始**一个神经网络学习逐步去噪**的例子。
# 对于图像的更多细节，设置包括2个过程：
#  *  我们选择的固定（或预定义）正向扩散过程$q$，它逐渐将高斯噪声添加到图像中，直到最终得到纯噪声
#  *  一个学习的反向去噪的扩散过程$p_\theta$，其中神经网络被训练以从纯噪声开始逐渐对图像去噪，直到最终得到一个实际的图像。
#
# ![Image](https://drive.google.com/uc?id=1t5dUyJwgy2ZpDAqHXw7GhUAp2FE5BWHA)
#
# 由 \\(t\\) 索引的正向和反向过程都发生在某些有限时间步长\\(T\\)（DDPM作者使用\\(T=1000\\))。您从\\(t=0\\)开始，在数据分布中采样真实图像 \\(\mathbf{x}_0\\) （假设来自ImageNet的猫图像），正向过程在每个时间步长\\(t\\)从高斯分布中采样一些噪声，将添加到上一个时刻的图像中。给定一个足够大的\\(T\\)和一个在每个时间步长添加噪声的良好时间表，您最终会在\\(t=T\\)通过渐进的过程得到所谓的[各向同性的高斯分布](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic)。
#
# ## 更精确的形式
#
# 让我们更正式地把这一点写下来，因为最终我们需要一个可控制的损失函数，运用神经网络进行优化。
#
# 设 \\(q(\mathbf{x}_0)\\) 是真实数据分布，比如"真实图像"。我们可以从这个分布中采样以获得图像， \\(\mathbf{x}_0 \sim q(\mathbf{x}_0)\\). 我们定义了前向扩散过程 \\(q(\mathbf{x}_t | \mathbf{x}_{t-1})\\) ，根据已知的方差 \\(0 < \beta_1 < \beta_2 < ... < \beta_T < 1\\) 在每个时间步长\\(t\\)添加高斯噪声，因此，
# $$
# q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}).
# $$
# 回想一下，正态分布（也称为高斯分布）由两个参数定义：平均值\\(\mu\\)和方差 \\(\sigma^2 \geq 0\\). 基本上，在时间步长\\(t\\)处的每个新的（轻微噪声）图像都是从**条件高斯分布**中绘制的，其中 \\(\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}\\) 且 \\(\sigma^2_t = \beta_t\\). 我们可以通过采样 \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) 然后设置 \\(\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}\\)。
#
# 请注意， \\(\beta_t\\) 在每个时间步长\\(t\\)（因此是下标）不是恒定的---事实上，我们定义了一个所谓的**"方差计划"**，可以是线性的、二次的、余弦的等，正如我们将进一步看到的那样（有点像学习速率计划）。
#
# 因此，从 \\(\mathbf{x}_0\\)开始，我们最终得到 \\(\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T\\)，其中，如果我们适当设置时间表， \\(\mathbf{x}_T\\) 是纯高斯噪声。
#
# 现在，如果我们知道条件分布\\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\)，我们可以反向运行这个过程：通过采样一些随机高斯噪声 \\(\mathbf{x}_T\\)，然后逐渐"去噪"它，这样我们就能得到实分布\\(\mathbf{x}_0\\)中的样本。
#
# 但是，我们不知道 \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\)。这很棘手，因为需要知道所有可能图像的分布，才能计算这个条件概率。因此，我们将利用神经网络来**近似（学习）这个条件概率分布**，可以称之为 \\(p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)\\), \\(\theta\\)是神经网络的参数，通过梯度下降更新。
#
# 好吧，我们需要一个神经网络来表示反向过程的（条件）概率分布。如果我们假设这个反向过程也是高斯的，那么请记住，任何高斯分布都由2个参数定义：
#  * 由 \\(\mu_\theta\\)参数化的平均值；
#  * 由 \\(\mu_\theta\\)参数化的方差；
#
# 因此，我们可以将过程参数化为
# $$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
# 其中平均值和方差也取决于噪声水平\\(t\\)。

# ## 什么是扩散模型？
#
# 如果将（去噪）扩散模型与其他生成模型（如Normalizing Flows、GAN或VAE）进行比较，它并没有那么复杂：它们都将噪声从一些简单分布转换为数据样本。这也是从纯噪声开始**一个神经网络学习逐步去噪**的例子。
# 对于图像的更多细节，设置包括2个过程：
#  *  我们选择的固定（或预定义）正向扩散过程$q$，它逐渐将高斯噪声添加到图像中，直到最终得到纯噪声
#  *  一个学习的反向去噪的扩散过程$p_\theta$，其中神经网络被训练以从纯噪声开始逐渐对图像去噪，直到最终得到一个实际的图像。
#
# ![Image](https://drive.google.com/uc?id=1t5dUyJwgy2ZpDAqHXw7GhUAp2FE5BWHA)
#
# 由 \\(t\\) 索引的正向和反向过程都发生在某些有限时间步长\\(T\\)（DDPM作者使用\\(T=1000\\))。您从\\(t=0\\)开始，在数据分布中采样真实图像 \\(\mathbf{x}_0\\) （假设来自ImageNet的猫图像），正向过程在每个时间步长\\(t\\)从高斯分布中采样一些噪声，将添加到上一个时刻的图像中。给定一个足够大的\\(T\\)和一个在每个时间步长添加噪声的良好时间表，您最终会在\\(t=T\\)通过渐进的过程得到所谓的[各向同性的高斯分布](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic)。
#
# ## 更精确的形式
#
# 让我们更正式地把这一点写下来，因为最终我们需要一个可控制的损失函数，运用神经网络进行优化。
#
# 设 \\(q(\mathbf{x}_0)\\) 是真实数据分布，比如"真实图像"。我们可以从这个分布中采样以获得图像， \\(\mathbf{x}_0 \sim q(\mathbf{x}_0)\\). 我们定义了前向扩散过程 \\(q(\mathbf{x}_t | \mathbf{x}_{t-1})\\) ，根据已知的方差 \\(0 < \beta_1 < \beta_2 < ... < \beta_T < 1\\) 在每个时间步长\\(t\\)添加高斯噪声，因此，
# $$
# q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}).
# $$
# 回想一下，正态分布（也称为高斯分布）由两个参数定义：平均值\\(\mu\\)和方差 \\(\sigma^2 \geq 0\\). 基本上，在时间步长\\(t\\)处的每个新的（轻微噪声）图像都是从**条件高斯分布**中绘制的，其中 \\(\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}\\) 且 \\(\sigma^2_t = \beta_t\\). 我们可以通过采样 \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) 然后设置 \\(\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}\\)。
#
# 请注意， \\(\beta_t\\) 在每个时间步长\\(t\\)（因此是下标）不是恒定的---事实上，我们定义了一个所谓的**"方差计划"**，可以是线性的、二次的、余弦的等，正如我们将进一步看到的那样（有点像学习速率计划）。
#
# 因此，从 \\(\mathbf{x}_0\\)开始，我们最终得到 \\(\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T\\)，其中，如果我们适当设置时间表， \\(\mathbf{x}_T\\) 是纯高斯噪声。
#
# 现在，如果我们知道条件分布\\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\)，我们可以反向运行这个过程：通过采样一些随机高斯噪声 \\(\mathbf{x}_T\\)，然后逐渐"去噪"它，这样我们就能得到实分布\\(\mathbf{x}_0\\)中的样本。
#
# 但是，我们不知道 \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\)。这很棘手，因为需要知道所有可能图像的分布，才能计算这个条件概率。因此，我们将利用神经网络来**近似（学习）这个条件概率分布**，可以称之为 \\(p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)\\), \\(\theta\\)是神经网络的参数，通过梯度下降更新。
#
# 好吧，我们需要一个神经网络来表示反向过程的（条件）概率分布。如果我们假设这个反向过程也是高斯的，那么请记住，任何高斯分布都由2个参数定义：
#  * 由 \\(\mu_\theta\\)参数化的平均值；
#  * 由 \\(\mu_\theta\\)参数化的方差；
#
# 因此，我们可以将过程参数化为
# $$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
# 其中平均值和方差也取决于噪声水平\\(t\\)。

# 因此，我们的神经网络需要学习/表示均值和方差。然而，DDPM的作者决定**保持方差固定，让神经网络只学习（表示）这个条件概率分布的平均值\\(\mu_\theta\\)**。从文章中可知：
#
# > 首先，我们将\\(\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}\\)设置为未训练的时间相关常数。实验上，\\(\sigma^2_t = \beta_t\\) and \\(\sigma^2_t  = \tilde{\beta}_t\\)（见论文）都有相似的结果。
#
# 这点后来在[改进的扩散模型](https://openreview.net/pdf?id=-NEXDKk8gZ)论文中继续深入研究，其中神经网络还学习了这个反向过程的方差，除了平均值。
#
# 所以我们继续，假设神经网络只需要学习/表示这个条件概率分布的平均值。
#
#
# ## 定义目标函数（通过重新参数化平均值）
#
# 为了导出一个目标函数来学习反向过程的平均值，作者观察到\\(q\\)和 \\(p_\theta\\) 的组合可以被视为变分自动编码器(VAE)[（Kingma等人，2013年）](https://arxiv.org/abs/1312.6114)。因此，**变分下界**（也称为ELBO）可用于最小化地面真值数据样本\\(\\mathbf\{x\}_0\\)的负对数似然性（有关ELBO的详细信息，请参阅VAE论文）。结果表明，该过程的ELBO是每个时间步长\\(t\\),\\(L=L_0+L_1+...+L_T\\)的损失之和。通过构造正向\\(q\\)过程和反向过程，损失的每个项（除了 \\(L_0\\))实际上是**2个高斯分布之间的KL发散**，可以明确地写为相对于均值的L2-loss！
#
# 构建的正向过程的直接结果\\(q\\)，如Sohl-Dickstein等人所示，是我们可以在任何任意噪声水平上采样\\(\mathbf{x}_t\\)，条件是 \\(\mathbf{x}_0\\) (因为高斯和也是高斯）。这非常方便：我们不需要重复应用\\(q\\)就可以采样\\(\mathbf{x}_t\\)。使用\\(\alpha_t := 1 - \beta_t\\)和\\(\bar{\alpha}t := \Pi_{s=1}^{t} \alpha_s\\)，我们有
#
# $$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$ 让我们把这个等式称为"好属性"。这意味着我们可以采样高斯噪声并适当地缩放它，然后将其添加到\\(\mathbf{x}_0\\)中，直接获得\\(\mathbf{x}_t\\)。请注意，\\(\bar{\alpha}_t\\)是已知\\(\beta_t\\)方差计划的函数，因此也是已知的，可以预先计算。这允许我们在训练期间**优化损失函数\\(L\\)的随机项** . 或者换句话说，在训练期间随机采样\\(t\\)并优化\\(L_t\\)。

# 在这里， \\(\mathbf{x}_0\\) 是初始（真实，未损坏）图像，我们看到由固定正向过程给出的直接噪声水平 \\(t\\) 样本。 \\(\mathbf{\epsilon}\\) 是在时间步长\\(t\\)采样的纯噪声，\\(\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)\\)是我们的神经网络。神经网络是使用真实噪声和预测高斯噪声之间的简单均方误差（MSE）进行优化的。
#
# 训练算法现在如下所示：
#
# ![Image](https://drive.google.com/uc?id=1LJsdkZ3i1J32lmi9ONMqKFg5LMtpSfT4)
#
# 换句话说：
#
#  *  我们从真实未知和可能复杂的数据分布 $q(\mathbf{x}_0)$ 中随机抽取一个样本$\mathbf{x}_0$
#  *  我们均匀地采样 $1$和$T$之间的噪声水平$t$ （即，随机时间步长）
#  *  我们从高斯分布中采样一些噪声，并使用上面定义的尼斯属性在$t$级别损坏输入
#  *  神经网络被训练以基于损坏的图像$\mathbf{x}_t$来预测这种噪声，即基于已知的时间表$\mathbf{x}_t$上施加的噪声
#
# 实际上，所有这些都是在成批数据上完成的，因为人们使用随机梯度下降来优化神经网络。
#
# ## 神经网络
#
# 神经网络需要在特定时间步长接收带噪声的图像，并返回预测的噪声。请注意，预测噪声是与输入图像具有相同大小/分辨率的张量。因此，从技术上讲，网络接受并输出相同形状的张量。我们可以用什么类型的神经网络来做这个？
#
# 这里通常使用的是非常相似的[自动编码器](https://en.wikipedia.org/wiki/Autoencoder)	，您可能还记得典型的"深度学习入门"教程。自动编码器在编码器和解码器之间有一个所谓的"bottleneck"层。编码器首先将图像编码为一个称为"bottleneck"的较小的隐藏表示，然后解码器将该隐藏表示解码回实际图像。这迫使网络只保留bottleneck层中最重要的信息。
#
# 在体系结构方面，DDPM的作者选择了**U-Net**，出自([Ronneberger et al.，2015](https://arxiv.org/abs/1505.04597)	)（当时，它在医学图像分割方面取得了最先进的结果）。这个网络，就像任何自动编码器一样，在中间由一个bottleneck组成，确保网络只学习最重要的信息。重要的是，它在编码器和解码器之间引入了残差连接，极大地改善了梯度流(灵感来自于[He et al., 2015](https://arxiv.org/abs/1512.03385)).
#
# ![Image](https://drive.google.com/uc?id=1_Hej_VTgdUWGsxxIuyZACCGjpbCGIUi6)
#
# 可以看出，U-Net模型首先对输入进行下采样（即，在空间分辨率方面使输入更小），之后执行上采样。
#
# 下面，我们逐步实施这个网络。
#
# ### 网络助手
#
# 首先，我们定义了一些帮助函数和类，这些函数和类将在实现神经网络时使用。重要的是，我们定义了`Residual`模块，只是将输入添加到特定函数的输出中（换句话说，将残留连接添加到特定函数）。

# In[2]:


def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x * y))


# rsqrt 在2.0 报错，无法import
def rsqrt(x):
    rsqrt_op = _get_cache_prim(ops.Rsqrt)()
    return rsqrt_op(x)


def randn_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(x.shape).astype(dtype)


def randn(shape, dtype=None):
    if dtype is None:
        dtype = mindspore.float32
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(shape).astype(dtype)


def randint(low, high, size, dtype=mindspore.int32):
    uniform_int = _get_cache_prim(ops.UniformInt)()
    return uniform_int(size, Tensor(low, dtype), Tensor(high, dtype)).astype(dtype)


# softmax 可能在ascend报错提issue


# In[3]:


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# 我们还定义了上采样和下采样操作的别名。

# In[4]:


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1,
                 group=1, has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias,
                         weight_init='normal', bias_init='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        # self.weight = Parameter(initializer(HeUniform(math.sqrt(5)), self.weight.shape), name='weight')
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


def Upsample(dim):
    return nn.Conv2dTranspose(dim, dim, 4, 2, pad_mode="pad", padding=1)


def Downsample(dim):
    return Conv2d(dim, dim, 4, 2, pad_mode="pad", padding=1)


# ### 位置嵌入
#
# 由于神经网络的参数在时间（噪声水平）上共享，作者使用正弦位置嵌入来编码$t$，灵感来自Transformer([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))。对于批处理中的每一张图像，这使得神经网络"知道"它在哪个特定时间步长（噪声水平）上运行。
#
# `SinusoidalPositionEmbeddings`模块采用`(batch_size, 1)`形状的张量作为输入（即批处理中几个有噪声图像的噪声水平），并将其转换为`(batch_size, dim)`形状的张量，与`dim`是位置嵌入的尺寸。然后，我们将进一步看到将添加到每个剩余块中。

# In[5]:


class SinusoidalPositionEmbeddings(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * - emb)
        self.emb = Tensor(emb, mindspore.float32)

    def construct(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = ops.concat((ops.sin(emb), ops.cos(emb)), axis=-1)
        return emb


# ### ResNet/ConvNeXT块
#
# 接下来，我们定义U-Net模型的核心构建块。DDPM作者使用了一个Wide ResNet块([Zagoruyko et al., 2016](https://arxiv.org/abs/1605.07146))，但Phil Wang决定也添加对ConvNeXT块的支持([Liu et al., 2022](https://arxiv.org/abs/2201.03545))，因为后者在图像领域取得了巨大成功。在最终的U-Net架构中，人们可以选择其中一个或另一个。

# In[6]:


class WeightStandardizedConv2d(Conv2d):
    def construct(self, x):
        eps = 1e-5
        weight = self.weight
        mean = weight.mean((1, 2, 3), keep_dims=True)
        var = weight.var((1, 2, 3), keepdims=True)
        normalized_weight = (weight - mean) * rsqrt((var + eps))
        output = self.conv2d(x, normalized_weight.astype(x.dtype))
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1, pad_mode='pad')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def construct(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ConvNextBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.SequentialCell(nn.GELU(), nn.Dense(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = Conv2d(dim, dim, 7, padding=3, group=dim, pad_mode="pad")
        self.net = nn.SequentialCell(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            Conv2d(dim, dim_out * mult, 3, padding=1, pad_mode="pad"),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            Conv2d(dim_out * mult, dim_out, 3, padding=1, pad_mode="pad"),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def construct(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            condition = condition.expand_dims(-1).expand_dims(-1)
            h = h + condition

        h = self.net(h)
        return h + self.res_conv(x)


# ### 注意模块
#
# 接下来，我们定义注意力模块，DDPM作者将其添加到卷积块之间。注意是著名的Transformer架构([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)	)，在人工智能的各个领域都取得了巨大的成功，从NLP和愿景到[蛋白质折叠](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)。Phil Wang使用了两种注意力变体：一种是常规的多头自我注意力（如Transformer中使用的），另一种是[线性注意变体](https://github.com/lucidrains/linear-attention-transformer)	([Shen et al., 2018](https://arxiv.org/abs/1812.01243))，其时间和内存要求在序列长度上线性缩放，而不是在常规注意力中缩放二次元。
#
# 要想对注意力机制进行广泛的解释，请参照Jay Allamar的[精彩的博文](https://jalammar.github.io/illustrated-transformer/)。

# In[7]:


class Identity(nn.Cell):
    def construct(self, inputs):
        return inputs


class Attention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias=False)
        self.to_out = Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias=True)
        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).split(1, 3)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = q * self.scale

        # 'b h d i, b h d j -> b h i j'

        sim = ops.bmm(q.swapaxes(2, 3), k)
        attn = ops.softmax(sim, axis=-1)
        # 'b h i j, b h d j -> b h i d'
        out = ops.bmm(attn, v.swapaxes(2, 3))
        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = out.swapaxes(-1, -2).reshape((b, -1, h, w))
        # print("Attention to_qkv", self.to_qkv)
        # print("Attention to_out", self.to_out)
        return self.to_out(out)


class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.g = Parameter(initializer('ones', (1, dim, 1, 1)), name='g')

    def construct(self, x):
        eps = 1e-5
        var = x.var(1, keepdims=True)
        mean = x.mean(1, keep_dims=True)
        return (x - mean) * rsqrt((var + eps)) * self.g


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias=False)

        self.to_out = nn.SequentialCell(
            Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias=True),
            LayerNorm(dim)
        )

        self.map = ops.Map()
        self.partial = ops.Partial()
        self.is_ascend = mindspore.get_context('device_target') == 'Ascend'

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).split(1, 3)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = ops.softmax(q, -2)
        k = ops.softmax(k, -1)

        q = q * self.scale
        v = v / (h * w)

        # 'b h d n, b h e n -> b h d e'
        if self.is_ascend:
            context = (k.expand_dims(3) * v.expand_dims(2)).sum(-1)
        else:
            context = ops.bmm(k, v.swapaxes(2, 3))

        # 'b h d e, b h d n -> b h e n'
        if self.is_ascend:
            out = (context.expand_dims(-1) * q.expand_dims(-2)).sum(2)
        else:
            out = ops.bmm(context.swapaxes(2, 3), q)

        out = out.reshape((b, -1, h, w))
        return self.to_out(out)


# ### 组归一化 ###
#
# DDPM作者将U-Net的卷积/注意层与群归一化([Wu et al., 2018](https://arxiv.org/abs/1803.08494))。下面，我们定义一个`PreNorm`类，将用于在注意层之前应用groupnorm。请注意，有一个关于在Transformers中在注意之前还是之后应用归一化的[debate](https://tnq177.github.io/data/transformers_without_tears.pdf)。

# In[8]:


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def construct(self, x):
        x = self.norm(x)
        return self.fn(x)

# ### 条件U-Net
#
# 我们已经定义了所有的构建块（位置嵌入、ResNet/ConvNeXT块、注意力和组标准化），现在需要定义整个神经网络了。请记住，网络\\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\)的工作是接收一批噪声图像+噪声水平，并输出添加到输入中的噪声。更正式的：
#
#  *  网络获取了一批`(batch_size, num_channels, height, width)`形状的噪声图像和一批`(batch_size, 1)`形状的噪音水平作为输入，并返回`(batch_size, num_channels, height, width)`形状的张量。
#
# 网络建设如下：
#
#  *  首先，将卷积层应用于噪声图像批上，并计算噪声水平的位置嵌入
#  *  接下来，应用一系列下采样级。每个下采样阶段由2个ResNet/ConvNeXT块 + groupnorm + attention + 残差连接 + 一个下采样操作组成
#  *  在网络的中间，再次应用ResNet或ConvNeXT块，并与attention交织
#  *  接下来，应用一系列上采样级。每个上采样级由2个ResNet/ConvNeXT块+ groupnorm + attention + 残差连接 + 一个上采样操作组成
#  *  最后，应用ResNet/ConvNeXT块，然后应用卷积层。
#
# 最终，神经网络将层堆叠起来，就像它们是乐高积木一样(但重要的是[了解它们是如何工作的](http://karpathy.github.io/2019/04/25/recipe/))。

# In[9]:

class Unet(nn.Cell):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=1,
            use_convnext=False,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = Conv2d(channels, init_dim, 7, padding=3, pad_mode="pad", has_bias=True)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block_klass = partial(ConvNextBlock, mult=convnext_mult)
        # block_klass = partial(ConvNextBlock)
        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.SequentialCell(
                SinusoidalPositionEmbeddings(dim),
                nn.Dense(dim, time_dim),
                nn.GELU(),
                nn.Dense(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.CellList([])
        self.ups = nn.CellList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.CellList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.CellList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.SequentialCell(
            block_klass(dim, dim), Conv2d(dim, out_dim, 1)
        )

    def construct(self, x, time):
        x = self.init_conv(x)

        r = x.copy()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            # h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        len_h = len(h) - 1
        for block1, block2, attn, upsample in self.ups:
            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        return self.final_conv(x)


# ## 定义正向扩散过程
#
# 正向扩散过程在多个时间步长$T$中，从实际分布逐渐向图像添加噪声。根据**差异计划**进行正向扩散。最初的DDPM作者采用了线性时间表：
#
# > 我们将正向过程方差设置为常数，从$\beta_1 = 10^{−4}$线性增加到$\beta_T = 0.02$。
#
# 但是，它显示在([Nichol et al., 2021](https://arxiv.org/abs/2102.09672))中，当使用余弦调度时，可以获得更好的结果。
#
# 下面，我们定义了$T$时间步的各种时间表，以及我们需要的相应变量，如累积方差。

# In[10]:


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps).astype(np.float32)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps).astype(np.float32) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(-6, 6, timesteps)
    return np.sigmoid(betas) * (beta_end - beta_start).astype(np.float32) + beta_start


# 首先，让我们使用\\(T=200\\)时间步长的线性计划，并定义我们需要的\\(\\β_t\\)中的各种变量，例如方差 \\(\bar{\alpha}_t\\)的累积乘积。下面的每个变量都只是一维张量，存储从\\(t\\)到\\(T\\)的值。重要的是，我们还定义了`extract`函数，它将允许我们提取一批索引的适当\\(t\\)索引。

# In[11]:


timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

sqrt_recip_alphas = Tensor(np.sqrt(1. / alphas))
sqrt_alphas_cumprod = Tensor(np.sqrt(alphas_cumprod))
sqrt_one_minus_alphas_cumprod = Tensor(np.sqrt(1. - alphas_cumprod))

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    return a[t, None, None, None]


# 我们将用猫图像说明如何在扩散过程的每个时间步骤中添加噪音。

# In[12]:


# export http_proxy="http://g00498674:r7%40%231036@172.18.100.92:8080" 这个代理可以成功wget URL

from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image


#
# ![Image](https://drive.google.com/uc?id=17FXnvCTl96lDhqZ_io54guXO8hM-rsQ2)
#
# 噪声被添加到PyTorch张量中，而不是Pillow图像。我们将首先定义图像转换，允许我们从PIL图像转换到PyTorch张量（我们可以在其上添加噪声），反之亦然。
#
# 这些转换相当简单：我们首先通过除以$255$来标准化图像（使它们在 $[0,1]$ 范围内），然后确保它们在 $[-1, 1]$ 范围内。来自DPPM论文：
#
# > 我们假设图像数据由 $\{0, 1, ... , 255\}$ 中的整数组成，线性缩放为 $[−1, 1]$. 这确保了神经网络反向过程在从标准正常先验 $p(\mathbf{x}_T )$开始的一致缩放输入上运行。
#

# In[13]:


from mindspore.dataset import ImageFolderDataset

image_size = 128

transforms = [
    Resize(image_size, Inter.BILINEAR),
    CenterCrop(image_size),
    ToTensor(),
    lambda t: (t * 2) - 1
]

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
path = download(url, './image_cat/jpg', replace=False)
path = './image_cat'
dataset = ImageFolderDataset(dataset_dir=path, num_parallel_workers=cpu_count(),
                             extensions=['.jpg', '.jpeg', '.png', '.tiff'],
                             num_shards=1, shard_id=0, shuffle=False, decode=True)
dataset = dataset.project('image')
transforms.insert(1, RandomHorizontalFlip())
print(type(dataset))
dataset_1 = dataset.map(transforms, 'image')
dataset_2 = dataset_1.batch(1, drop_remainder=True)
x_start = next(dataset_2.create_tuple_iterator())[0]
print(x_start.shape)


# <div class="output stream stdout">
#
#     Output:
#     ----------------------------------------------------------------------------------------------------
#     torch.Size([1, 3, 128, 128])
#
# </div>

#
# 我们还定义了反向变换，它接收一个包含 $[-1, 1]$ 中的值的 PyTorch 张量，并将它们转回 PIL 图像：

# In[14]:


import numpy as np
from mindspore.dataset.transforms import Compose

reverse_transform = [
     lambda t: (t + 1) / 2,
     lambda t: t.permute(1, 2, 0), # CHW to HWC
     lambda t: t * 255.,
     lambda t: t.numpy().astype(np.uint8),
     ToPIL()
]

def compose(transform, x):
    for d in transform:
        x = d(x)
#     x = x.resize((640, 480),Image.Resampling.LANCZOS)
    return x


# 让我们验证一下：

# In[15]:


reverse_image= compose(reverse_transform, x_start[0])
reverse_image.show()


# ![Image](https://drive.google.com/uc?id=1WT22KYvqJbHFdYYfkV7ohKNO4alnvesB)
#
# 我们现在可以定义前向扩散过程，如本文所示：

# In[16]:


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = randn_like(x_start)
    return (
            extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


# 让我们在特定的时间步长上测试它：

# In[17]:


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image= compose(reverse_transform, x_noisy[0])

    return noisy_image


# In[18]:


# take time step
t = Tensor([40])
noisy_image = get_noisy_image(x_start, t)
noisy_image.show()


# ![Image](https://drive.google.com/uc?id=1Ra33wxuw3QxPlUG0iqZGtxgKBNdjNsqz)
#
# 让我们为不同的时间步骤可视化此情况：

# In[19]:


import matplotlib.pyplot as plt

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# In[20]:


plot([get_noisy_image(x_start, Tensor([t])) for t in [0, 50, 100, 150, 199]])


#
# ![Image](https://drive.google.com/uc?id=1QifsBnYiijwTqru6gur9C0qKkFYrm-lN)
#
# 这意味着我们现在可以定义给定模型的损失函数，如下所示：

# In[21]:


def p_losses(unet_model, x_start, t, noise=None):
    if noise is None:
        noise = randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = unet_model(x_noisy, t)

    loss = nn.SmoothL1Loss()(noise, predicted_noise)# todo
    # print("alphas_cumprod",alphas_cumprod.all())
    p2_loss_weight = (1 + alphas_cumprod / (1 - alphas_cumprod)) ** -0.
    p2_loss_weight = Tensor(p2_loss_weight)
    loss = loss.reshape(loss.shape[0], -1)
    loss = loss * extract(p2_loss_weight, t, loss.shape)
    return loss.mean()


# `denoise_model`将是我们上面定义的U-Net。我们将在真实噪声和预测噪声之间使用Huber损失。

# ## 定义PyTorch数据集+ DataLoader
#
# 在这里我们定义一个正则数据集。数据集简单地由来自真实数据集的图像组成，如Fashion-MNIST、CIFAR-10或ImageNet，线性缩放为\\([−1, 1]\\)。
#
# 每个图像的大小都会调整为相同的大小。有趣的是，图像也是随机水平翻转的。根据论文内容：
#
# > 我们在CIFAR10的训练中使用了随机水平翻转；我们尝试了有翻转和没有翻转的训练，并发现翻转可以稍微提高样本质量。
#
# 在这里我们使用[数据集库](https://huggingface.co/docs/datasets/index)轻松加载时尚MNIST数据集[hub](https://huggingface.co/datasets/fashion_mnist)。此数据集由已经具有相同分辨率的图像组成，即28x28。

# In[22]:


from mindspore.dataset import FashionMnistDataset


image_size = 28
channels = 1
batch_size = 128

fashion_mnist_dataset_dir = "./dataset"
dataset = FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir, num_parallel_workers=cpu_count(), shuffle=True, num_shards=1, shard_id=0)


# 接下来，我们定义一个transform操作，将在整个数据集上动态应用该操作。该操作应用一些基本的图像预处理：随机水平翻转、重新调整，最后使它们的值在 $[-1,1]$ 范围内。

# In[23]:


transfroms = [
    RandomHorizontalFlip(),
    ToTensor(),
    lambda t: (t * 2) - 1
]

dataset = dataset.project('image')
dataset = dataset.shuffle(64)
dataset = dataset.map(transfroms, 'image')
dataset = dataset.batch(128, drop_remainder=True)


# In[24]:


x = next(dataset.create_dict_iterator())
print(x.keys())


# <div class="output stream stdout">
#
#     Output:
#     ----------------------------------------------------------------------------------------------------
#     dict_keys(['image'])
#
# </div>

# ## 采样
#
# 由于我们将在训练期间从模型中采样（以便跟踪进度），我们定义了下面的代码。采样在本文中总结为算法2：
#
# ![Image](https://drive.google.com/uc?id=1ij80f8TNBDzpKtqHjk_sh8o5aby3lmD7)
#
# 从扩散模型生成新图像是通过反转扩散过程来实现的：我们从$T$开始，我们从高斯分布中采样纯噪声，然后使用我们的神经网络逐渐去噪（使用它所学习的条件概率），直到我们最终在时间步$t = 0$结束。如上图所示，我们可以通过使用我们的噪声预测器插入平均值的重新参数化，导出一个降噪程度较低的图像$\mathbf{x}_{t-1 }$。请记住，方差是提前知道的。
#
# 理想情况下，我们最终会得到一个看起来像是来自真实数据分布的图像。
#
# 下面的代码实现了这一点。

# In[25]:


@ms_function
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    print("sqrt_recip_alphas_t", )
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + ops.sqrt(posterior_variance_t) * noise


def p_sample_loop(model, shape):
    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = randn(shape, dtype=None)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, ms.numpy.full((b,), i, dtype=mstype.int32), i)
        imgs.append(img.asnumpy())
    return imgs


def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


# 下面，我们定义模型，并将其移动到GPU。我们还定义了一个标准优化器（Adam）。

# In[27]:


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)

name_list = []
for (name, par) in list(model.parameters_and_names()):
    name_list.append(name)
i = 0
for item in list(model.trainable_params()):
    item.name = name_list[i]
    i += 1

optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)


# 让我们开始训练吧！

# In[28]:


from mindspore.amp import DynamicLossScaler
loss_scaler = DynamicLossScaler(65536, 2, 1000)

def forward_fn(data, t, noise=None):
    loss = p_losses(model, data, t, noise)
    loss = loss_scaler.scale(loss)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)


# In[29]:


def _check_dtype(d1, d2):
    if mindspore.float32 in (d1, d2):
        return mindspore.float32
    if d1 == d2:
        return d1
    raise ValueError('dtype is not supported.')


# In[30]:


def grad(fn, pos=None, params=None, has_aux=False):
    value_and_grad_f = ms.value_and_grad(fn, pos, params, has_aux)

    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g

    return grad_f

@ms_class
class Accumulator():
    def __init__(self, optimizer, accumulate_step, total_step=None, clip_norm=1.0):
        # super().__init__()
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        if total_step is not None:
            assert total_step > accumulate_step and total_step > 0
        self.total_step = total_step
        self.map = ops.Map()
        self.partial = ops.Partial()

    def __call__(self, grads):
        success = self.map(self.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            # clip_grads, _ = clip_grad_norm(self.inner_grads, self.clip_norm)
            clip_grads = ops.clip_by_global_norm(self.inner_grads, self.clip_norm)
            success = ops.depend(success, self.optimizer(clip_grads))
            success = ops.depend(success, self.map(self.partial(ops.assign), self.inner_grads, self.zeros))
        success = ops.depend(success, ops.assign_add(self.counter, Tensor(1, mindspore.int32)))

        return success


# In[33]:


from mindspore import amp

accumulator = Accumulator(optimizer, 1)

# @ms_function
def train_step(data, t, noise):
    loss, grads = grad_fn(data, t, noise)
    grads = ops.identity(grads)
    status = amp.all_finite(grads)
    if status:
        loss = loss_scaler.unscale(loss)
        grads = loss_scaler.unscale(grads)
        loss= ops.depend(loss, accumulator(grads))
    loss = ops.depend(loss, loss_scaler.adjust(status))
    return loss


# In[ ]:


epochs = 20

for epoch in range(epochs):
    model.set_train()
    step = 0
    for _, batch in enumerate(dataset.create_tuple_iterator()):
        batch_size = batch[0].shape[0]

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = randint(0, timesteps, (batch_size,), dtype=mindspore.int32)
        loss = train_step(batch[0], t, None)

        if step % 1000 == 0:
            print("Loss:", loss)
        step+=1


# <div class="output stream stdout">
#
#     Output:
#     ----------------------------------------------------------------------------------------------------
#     Loss: 0.46477368474006653
#     Loss: 0.12143351882696152
#     Loss: 0.08106148988008499
#     Loss: 0.0801810547709465
#     Loss: 0.06122320517897606
#     Loss: 0.06310459971427917
#     Loss: 0.05681884288787842
#     Loss: 0.05729678273200989
#     Loss: 0.05497899278998375
#     Loss: 0.04439849033951759
#     Loss: 0.05415581166744232
#     Loss: 0.06020551547408104
#     Loss: 0.046830907464027405
#     Loss: 0.051029372960329056
#     Loss: 0.0478244312107563
#     Loss: 0.046767622232437134
#     Loss: 0.04305662214756012
#     Loss: 0.05216279625892639
#     Loss: 0.04748568311333656
#     Loss: 0.05107741802930832
#     Loss: 0.04588869959115982
#     Loss: 0.043014321476221085
#     Loss: 0.046371955424547195
#     Loss: 0.04952816292643547
#     Loss: 0.04472338408231735
#
# </div>

# ## 采样 (推理)
#
# 要从模型中采样，我们可以只使用上面定义的采样函数：

# In[ ]:


def extract(a, t, x_shape):
    b = t.shape[0]
    out = Tensor(a).gather(t, -1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# In[ ]:


def p_sample(model, x, t, t_index):
    betas_t = extract(Tensor(betas), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + ops.sqrt(posterior_variance_t) * noise


def p_sample_loop(model, shape):
    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = randn(shape, dtype=None)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, ms.numpy.full((b,), i, dtype=mstype.int32), i)
        imgs.append(img.asnumpy())
    return imgs


def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


# In[ ]:


# sample 64 images
model.set_train(False)
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)


# In[ ]:


# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")


# ### ![Image](https://drive.google.com/uc?id=1ytnzS7IW7ortC6ub85q7nud1IvXe2QTE)
#
# 看来这个模型能产生一件漂亮的T-shirt！请注意，我们训练的数据集分辨率相当低（28x28）。
#
# 我们还可以创建去噪过程的gif：

# In[ ]:


import matplotlib.animation as animation

random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()


# ![png](output_74_1.png)
#
#
# <img src="https://drive.google.com/uc?id=1eyonQWhfmbQsTq8ndsNjw5QSRQ9em9Au" width="500" />

# ![Image](https://drive.google.com/uc?id=1eyonQWhfmbQsTq8ndsNjw5QSRQ9em9Au)
#
# # 后续阅读
#
# 请注意，DDPM论文表明，扩散模型是（非）条件图像有希望生成的方向。自那以后，这得到了（极大的）改进，最明显的是文本条件图像生成。下面，我们列出了一些重要的（但远非详尽无遗的）后续工作：
#
#  *  改进的去噪扩散概率模型([Nichol et al., 2021](https://arxiv.org/abs/2102.09672))：发现学习条件分布的方差（除平均值外）有助于提高性能
#  *  用于高保真图像生成的级联扩散模型([[Ho et al., 2021](https://arxiv.org/abs/2106.15282))：引入级联扩散，它包括多个扩散模型的流水线，这些模型生成分辨率提高的图像，用于高保真图像合成
#  *  扩散模型在图像合成上击败了GANs([Dhariwal et al., 2021](https://arxiv.org/abs/2105.05233))：表明扩散模型通过改进U-Net体系结构以及引入分类器指导，可以获得优于当前最先进的生成模型的图像样本质量
#  *  无分类器扩散指南([[Ho et al., 2021](https://openreview.net/pdf?id=qw8AKxfYbI))：表明通过使用单个神经网络联合训练条件和无条件扩散模型，不需要分类器来指导扩散模型
#  *  具有CLIP Latents (DALL-E 2) 的分层文本条件图像生成 ([Ramesh et al., 2022](https://cdn.openai.com/papers/dall-e-2.pdf))：在将文本标题转换为CLIP图像嵌入之前使用，然后扩散模型将其解码为图像
#  *  具有深度语言理解的真实文本到图像扩散模型（ImageGen）([Saharia et al., 2022](https://arxiv.org/abs/2205.11487))：表明将大型预训练语言模型（例如T5）与级联扩散结合起来，对于文本到图像的合成很有效
#
# 请注意，此列表仅包括在撰写本文，即2022年6月7日之前的重要作品。
#
# 目前，扩散模型的主要（也许唯一）缺点是它们需要多次正向传递来生成图像（对于像GAN这样的生成模型来说，情况并非如此）。然而，有[研究正在进行中](https://arxiv.org/abs/2204.13902)	只需要10个去噪步骤就能实现高保真生成。
#
#

# In[ ]:





# 请注意，上面的代码是原始实现的简化版本。我们发现simplification（与论文中的算法2一致）与[原始、更复杂的实现](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py)。

# In[26]:
