## 概率模型



```math

X:data \longrightarrow (x_1,x_2,x_3,\ldots,x_N)^T_{N*P}
=\begin{pmatrix}
    x_{11} & x_{12} & \cdots & x_{1p} \\
    x_{21} & x_{22} & \cdots & x_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{N1} & x_{N2} & \cdots & x_{Np} \\
\end{pmatrix}

\theta:parameter 
```

### 频率派
```math
\theta : \pmb{未知的常量} \quad
X: \pmb{random\ variables}

\theta_{MLE} = {\underset {\theta}{\operatorname {arg\,max} }}\,(\log P(X|\theta))=l(\theta)
```
这里在似然函数加上对数函数。是为了方便计算一下给出原因
```math
首先似然函数

    L(\theta)= \quad \prod_{i=1}^n p(x_i;\theta)

我们计算参数

    X^{iid}  p(x_i;\theta)   \quad =  \quad \prod_{i=1}^n p(x_i;\theta)

这是我们为了简化运算将似然函数前加上对数函数，将乘法转换为加法

    X^{iid} \log p(x_i;\theta) \quad = \quad \sum_{i=1}^n \log p(x_i;\theta)
    
    iid 为 \pmb{独立同分布}
```

### 贝叶斯派

```math
    \theta\ :\ \pmb{random\ variables} \quad,\quad
    \theta\ \sim\ p(\theta)
    
    P(\theta|X)\quad = \quad \frac {P(X|\theta) \cdot  P(\theta)} {P(X)} \quad = \quad \frac {P(X|\theta) \cdot  P(\theta)} {\int_{\theta}{P(X|\theta) \cdot P(\theta)d\theta}} \quad \varpropto \quad P(X|\theta) \cdot  P(\theta)


```

给出如下注解：

- `$P(\theta|X) \quad : \pmb{\quad 后验概率 \quad (posterior)} $`
- `$ P(X|\theta) \quad :\pmb{\quad 最大似然估计 \quad (MLE)}$`
- `$ P(\theta) \quad\quad \  : \quad \pmb{先验概率 \quad (priori)}$`


下面引入一个参数估计方法**MAP**

MAP是让后验概率取得最大的点，来做估计。

```math
    \theta_{MAP} \quad = \quad {\underset {\theta}{\operatorname {arg\,max} }}\, P(\theta|X) \quad = \quad {\underset {\theta}{\operatorname {arg\,max} }}\, P(X|\theta) \ \cdot \ P(\theta)
```
但是最大后验估计并不是真正的贝叶斯估计

真正的贝叶斯估计： `$ p(\theta|X) = \quad  = \quad \frac {P(X|\theta) \cdot  P(\theta)} {\int_{\theta}{P(X|\theta) \cdot P(\theta)d\theta}} $` 是要求积分的

贝叶斯估计实际上就是一个计算一个后验概率分布，有这个后验概率分布又有什么用呢，这里我们介绍贝叶斯预测。

#### 贝叶斯预测

`$ X\ ：\ 已知数据 \quad x_{new}\ ：\ 新数据 $`  

贝叶斯预测

首先要把 需要引入`$\theta $`参数 `$X\ \longrightarrow\ \theta\ \longrightarrow\ X_{new} $`

```math
p(x_{new}|X)= \int_{\theta}{p(x_{new},\theta|X)d\theta} \quad = \quad \int_{\theta}{p(X_{new}|\theta)\cdot p(\theta|X)d\theta}

```
这里做一下注解
`$ p(x_{new}|X)= \int_{\theta}{p(x_{new},\theta|X)d\theta} \quad $`这一步是前者是后者的边缘概率。这样就可以把参数`$\theta $`引入

 `$\int_{\theta}{p(X_{new}|\theta)\cdot p(\theta|X)d\theta} $` 这里我们可以很容易看出后者是后验概率
 
 最后老师说到，我们可以看到贝叶斯估计是一个积分，在整个参数空间上求积分是很复杂的，很难求，在贝叶斯派发展出了概率图模型，贝叶斯角度就是求积分问题，而大部分积分解析解是很难求的，所以我们可以使用蒙托卡罗的方式 比如说MCMC采样方式。
 
  
## 总结一下
 
频率派`$ \longrightarrow $` 统计机器学习 ——优化问题

- 建模
- 设计loss funciton
- 具体的算法 algorithm

贝叶斯派`$ \longrightarrow $` 概率图模型 ——求积分

- MCMC (Monte Carlo Method)