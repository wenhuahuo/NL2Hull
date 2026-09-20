网络出版地址：http://kns.cnki.net/kcms/detail/42.1755.TJ.20170926.1103.028.html

期刊网址：www.ship-research.com

引用格式：张彦儒，林焰，陆丛红，等.轻量化NURBS船体曲面自行设计垂向参数化方法[J].中国舰船研究，2017，12(5)：30-37，45. ZHANG Y R，LIN Y，LU C H，et al. Lightweight hull surface self-design vertical parameterization method based on NURBS[J]. Chinese Journal of Ship Research, 2017, 12(5):30-37,45.

<div align="center">

# 轻量化NURBS船体曲面自行设计垂向参数化方法

</div>

张彦儒 $ ^{1} $，林焰 $ ^{1,2} $ ，陆丛红 $ ^{1} $ ，纪卓尚 $ ^{1} $

1 大连理工大学 船舶工程学院，辽宁大连116085 2 大连理工大学 工业装备结构分析国家重点实验室，辽宁大连116085

摘要：[目的]当前常规的船体曲面设计局限于现有母型船设计空间，并且不能以足够少的参数驱动生成设计船型。为了解决上述问题，[方法]将吃水函数与NURBS方法相结合，提出船舶自行设计垂向参数化方法。以船体水线为基本设计单元，以平底线、设计水线、首尾轮廓线、平边线及最大横剖线为特征约束，以特征参数对应的吃水函数值为设计目标，建立水线逼近模型。可应用进化算法对该逼近模型进行求解，最后通过蒙皮法生成船体曲面。[结果]相关特征线的设计实例表明了该方法的实用性和先进性。[结论]应用该方法可以通过尽可能少的数据量完成船体曲面设计，且更适用于新船型的自行设计。

关键词：NURBS；船体曲面；自行设计；垂向参数化；特征线

中图分类号：U662.2

文献标志码：A

DOI: 10.3969/j.issn.1673-3185.2017.05.004

<div align="center">

# Lightweight hull surface self-design vertical parameterization method based on NURBS

</div>

ZHANG Yanru $ ^{1} $ , LIN Yan $ ^{1,2} $ , LU Conghong $ ^{1} $ , JI Zhuoshang $ ^{1} $

1 School of Naval Architecture Engineering, Dalian University of Technology, Dalian 116085, China 2 State Key Laboratory of Structural Analysis for Industrial Equipment, Dalian University of Technology, Dalian 116085, China

Abstract: [Objectives] At present, conventional design is limited to parent ship design space, and cannot drive ship hull design using as few parameters as possible. In order to solve the above problems, [Methods] by combining the draught function with NURBS, a ship hull surface self-design method based on vertical parameterization is proposed. In this method, the waterline is designated as the basic design unit; the bottom flat end line, designed waterline, stem and stern contours, side flat end line and maximum section line are designated as the characteristic constraints of the ship hull; and the draught function values corresponding to the characteristic parameters are designated as the design objectives. In this way, a waterline approximation model is built, and an evolutionary algorithm can be used to solve the approximation model. Finally, the ship hull surface is generated on the basis of the waterline using the NURBS skinning technique. [Results] The design examples of the characteristic curves of the full-scale ship hull surface indicate the practicable and advanced nature of this method. [Conclusions] The hull surface can be designed with as little data as possible using this method, making it much more suitable for the self-design of new ship forms.

Key words: NURBS; hull surface; self-design; vertical parameterization; characteristic curves

## 0 引 言

船体曲面设计是船舶后续设计的基础，非均匀有理B样条（Non-Uniform Rational B-Spline，NURBS）是当今船体曲面设计的主流方法。目前，船体曲面设计主要是基于型值点的设计。其过程是由型值点插值生成曲面截面线（B样条曲线），再由曲面截面线蒙皮生成船体曲面[1-4]。其中，型值点的确定基本依据某种母型变换法。由于大部分基于NURBS的船体曲面设计方法不考虑权因子的作用，故截面线会退化为B样条，其生成的曲面为插值曲面。插值曲面固有的算法会导致控制顶点数过多，不利于后续曲面的光顺和修改。陆丛红等[5]考虑NURBS的权因子，运用实数编码的遗传算法对船体水线进行了逼近，之后该问题在文献[6]中得到了进一步的改进。但上述文献基本属于船型的表达，还未上升到设计的高度。在船型设计方面，于雁云等[7]提出了一种船体曲面参数化设计新方法，该方法实质上是船体曲面变换方法，即母型变换法。母型变换法虽然使设计船继承了母型船的优点，但也导致船体曲面在原有的设计圈中徘徊而难以创新。因此，研究一种数据量小，并且使设计不局限于已有母型“束缚”的船型自行设计方法具有重要意义。张萍等[8]提出一种光顺曲线的参数化设计技术，研究了基于横剖面面积“形心”的参数化船型设计方法，但其设计变量为型值点，设计截面线选择的是横剖线。由于横剖线凸凹性不同，设计变量的数目和初值的设定难以统一，不利于构建统一的优化设计框架。因此在综合对比船体特征线的形状特点的基础上，本文选择将水线作为参数化设计的基本单元，给出轻量化的船体水线和首尾轮廓线的曲线设计模型，将吃水函数法[9]与NURBS方法相结合，构建基于吃水函数的参数化自行设计系统。

## 1 NURBS

## 1.1 B样条基函数

设 $ U=\{u_{0},\dots ,u_{m}\} $为非递减的实数序列，即 $ u_{i}<u_{i+1}\ (i=0,\dots ,m-1) $ ，称 $ u_{i} $为节点，U为节点向量，则第i个p次B样条基函数 $ N_{i,p}(u) $定义如下 $ ^{[10]} $：

$$
\left\{ \begin{array}{l} N _ {i, 0} (u) = \left[ \begin{array}{l l} 1, & u _ {i} < u < u _ {i + 1} \\ 0, & \text {其 他} \end{array} \right) \\ N _ {i, p} (u) = \frac {u - u _ {i}}{u _ {i + p} - u _ {i}} N _ {i, p - 1} (u) + \frac {u _ {i + p + 1} - u}{u _ {i + p + 1} - u _ {i + 1}} N _ {i + 1, p - 1} (u) ^ {(1)} \\ 0 \% = 0 \end{array} \right.
$$

## 1.2 NURBS曲线定义

p次NURBS曲线定义为如下形式的分段有理参数曲线 $ ^{[10]} $：

$$
C (u) = \frac {\sum_ {i = 0} ^ {n} N _ {i , p} (u) \omega_ {i} P _ {i}}{\sum_ {i = 0} ^ {n} N _ {i , p} (u) \omega_ {i}}; 0 \leqslant u \leqslant 1
$$

式中： $ \{P_{i}\} $为控制点列，其连线形成曲线的控制多边形； $ \{\omega_{i}\} $为对应的权因子序列； $ \{N_{i,p}(u)\} $为定义在非周期非均匀节点矢量 $ U=\{ \underbrace{0,\cdots,0}_{p+1},u_{p+1}, $ $ \cdots,u_{m},\underbrace{1,\cdots,1}_{p+1} $上的p次B样条基函数。

## 2 船体曲面特征线

如图1所示，选择船体曲面上的平边线、首尾轮廓线、平底线、设计水线和最大横剖线作为船体曲面设计的特征线。考虑到平底线、设计水线及其他水线的形状特征类似，在垂向参数化自行设计系统中可以统一为一类水线的设计；而首尾轮廓线、最大横剖线和平边线可以作为待设计水线的特征点控制线来进行处理。因此，将先给出水线类的参数化设计模型，然后再给出首尾轮廓线、最大横剖线和平边线的参数控制模型，分别叙述如下。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_1_1789712182830.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=ejUykeGVU%2BoKCteGPmIGiMjmW3Y%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

图1 船体曲面特征线

</div>

<div align="center">

Fig.1 Characteristic curves of hull surface

</div>

## 2.1 水线类设计模型

首先，简化设计模型。船体水线一般是由首尾圆弧曲线（或直线）、首尾段自由曲线和中部的平直段直线部分构成，如图2所示。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_2_1789712182885.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=LnLM06sJD4VlkT%2Ff9K84JUucqA0%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

Fig.2 Waterline segmentation

</div>

由图2可知，船体水线的特征点为水线首尾端点，以及平直段的起止点，这些特征点可以由其他特征线，如首尾轮廓线和平边线来界定；圆弧半径在设计初期可以由吃水函数初步给定。另外，考虑到船体水线前体与后体的形状类似，可以构

建统一的参数化设计模型。因此，将坐标系原点设在舯横剖面与中纵剖线的交点；X轴定义为中纵剖面与基平面的交线，指向船艏、船艉均为正；Y轴定义为舯横剖面与基平面的交线，指向左舷为正；Z轴定义为中纵剖面与舯横剖面的交线，向上为正。在此坐标系下，给出水线前体的特征参数（对于水线后体的特征参数，只要把下标f变为a即可，下文不再赘述），如图3所示。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_1_1789712182893.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=4vEYmhyOOCJ9TnO9v4tXZ5JFsEE%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

图3 水线前体特征参数

</div>

<div align="center">

Fig.3 Characteristic parameter of waterline forebody

</div>

图中： $ L_{\mathrm{wf}} $为从船舯至船艏的水线前体长度，其值可以由首轮廓线确定； $ L_{\mathrm{pf}} $为从船舯测量的水线前体直线部分的长度，其值可以由平边线确定； $ B_{\mathrm{wf}} $为在船舯的水线前体半宽值，其值由最大横剖线确定； $ C_{\mathrm{wf}} $为水线前体的面积系数； $ CG_{\mathrm{wf}} $为水线前体形心距船舯的距离； $ CG_{\mathrm{bwf}} $为水线前体的形心半宽值； $ I_{\mathrm{f}} $为半进流角（此处指水线自由曲线段起点处的切矢）； $ R_{\mathrm{f}} $为水线起点处圆弧半径。上述值都是随吃水而变化，可以由第3节中的吃水函数确定。当这些值确定后，问题便可转化为求解满足给定特征参数的曲线逼近问题。下面，建立曲线设计模型。

## 2.1.1 设计模型的设计变量

设计变量的多少对参数化设计程序的效率影响较大。为了提高参数化程序的运行效率，经过反复比较分析，证实利用3次NURBS曲线来设计水线前体可以满足工程精度和灵活修改等要求。另外，因为首圆弧部分的形状可以由水线前端点和圆弧半径决定，所以本文在逼近模型中暂时不考虑首圆弧，将前体控制顶点的分布如图4所示进行设置。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_2_1789712182935.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=GzL6%2B%2Fyyvjh2Q3OH3jtdMOQZ5Ss%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

图4 水线前体控制顶点分布

</div>

<div align="center">

Fig.4 Control points setting for the waterline forebody

</div>

如图4所示，图中共有8个控制顶点（6个独立位置），分为以下3类：

1) 边界控制顶点 $ P_{0\mathrm{f}} $ ， $ P_{4\mathrm{f}-6\mathrm{f}} $和 $ P_{7\mathrm{f}} $ 。其中

$ P_{0\mathrm{f}} $为水线首圆弧与自由段之间的切点， $ P_{4\mathrm{f}-6\mathrm{f}} $为水线前体平直段的首端点， $ P_{7\mathrm{f}} $为水线与舯横剖面的交点。

2) 切矢控制顶点 $ P_{1\mathrm{f}} $和 $ P_{3\mathrm{f}} $。其中 $ P_{1\mathrm{f}} $与 $ P_{0\mathrm{f}} $的连线相切于首圆弧， $ P_{3\mathrm{f}} $处于 $ P_{4\mathrm{f}-6\mathrm{f}} $和 $ P_{7\mathrm{f}} $所确定的直线上，以达到自由段曲线与平直段和圆弧部分光顺连接的目的。

3) 形状控制顶点 $ P_{2\mathrm{f}} $ ，其与 $ P_{1\mathrm{f}} $ 和 $ P_{3\mathrm{f}} $ 结合用来控制水线的形状。

假设 $x_{i},y_{i},z_{i}$ 和 $\omega_{i}$ 分别为控制顶点 $P_{if}(i = 0,\dots ,7)$ 的纵向坐标、横向坐标、垂向坐标和权因子。其中， $z_{i}$ 的取值为待设计水线的吃水高度。根据控制顶点的分布特点可知： $y_{7} = y_{4 - 6} = y_{3} = B_{\mathrm{wf}}$ ； $x_{7} = 0$ ， $x_{4 - 6} = L_{\mathrm{pf}}$ ； $\tan (I_{\mathrm{f}}) = (y_{1} - y_{0}) / (x_{1} - x_{0})$ ； $x_{0} = L_{\mathrm{wf}} - R_{\mathrm{f}}(1 - \sin (I_{\mathrm{f}}))$ ， $y_{0} = R_{\mathrm{f}}\cdot \cos (I_{\mathrm{f}})$ 。权因子 $\omega_{0}$ ， $\omega_{4 - 6}$ 和 $\omega_{7}$ 设置为1。因此，水线前体曲线的设计变量为 $[x_{1},x_{2},x_{3},y_{2},\omega_{1},\omega_{2},\omega_{3}]$ 。

## 2.1.2 约束条件

因为水线不能产生迂回曲折现象，根据 NURBS的凸包性质，水线前体曲线的约束设置如下：

$$
\left\{ \begin{array}{l} x _ {4} < x _ {3} < x _ {2} < x _ {1} < x _ {0} \\ y _ {0} < y _ {1} < y _ {2} < y _ {3} = y _ {4} \\ 0 < \omega_ {i} < 5 0; i = 1, 2, 3 \\ \tan \left(I _ {\mathrm {f}}\right) = \left(y _ {1} - y _ {0}\right) / \left(x _ {1} - x _ {0}\right) \end{array} \right.
$$

## 2.1.3 设计目标

设 $ A_{\mathrm{wf}} $为待设计水线前体曲线自由曲线段与 x轴所围面积， $ C G_{x\mathrm{f}} $和 $ C G_{y\mathrm{f}} $分别为其对应形心的 x坐标和 y坐标（这些值可以由水线特征参数确定）， $ A_{\mathrm{wf}}^{\prime} $， $ C G_{x\mathrm{f}}^{\prime} $和 $ C G_{y\mathrm{f}}^{\prime} $分别为通过本文设计方法得到的水线前体自由曲线段的相应参数，则水线前体设计目标函数为

$$
\begin{array}{l} \operatorname {M i n} F (x) = \operatorname {M i n} \left(\left| \left(A _ {\mathrm {w f}} ^ {\prime} - A _ {\mathrm {w f}}\right) / A _ {\mathrm {w f}} \right| + \right. \\ \left| \left(C G _ {y f} ^ {\prime} - C G _ {y f}\right) / C G _ {y f} \right| + \left| \left(C G _ {x f} ^ {\prime} - C G _ {x f}\right) / C G _ {x f} \right|) \\ \end{array}
$$

如上所述，将水线参数化设计问题转化为曲线逼近问题后，该逼近问题便可由进化算法，如人机交互的遗传算法 $ ^{[11]} $进行求解（对于水线后体，只需把下标f变为a即可）。

下面，给出其他特征线的初步确定方法，以便在初始设计阶段快速得到待设计水线的特征参数。

## 2.2 首尾轮廓线控制顶点分布和控制参数

经过反复比较和分析，确定了3次NURBS曲

线逼近首尾轮廓线，对控制顶点作如图5和图6所示的设置，能满足工程精度及修改的灵活性等要求。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_1_1789712182941.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=Cpu1QSuP7hNLGDHiJDixM%2BNvBY0%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

图5 首轮廓线的控制顶点分布

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_2_1789712182948.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=WjQI9nDgmYfbV2I7%2BnJyyxGwJhE%3D&Expires=1790316982' alt='OCR图片'/></div>

$$
V _ {1 9 \mathrm {a}}
$$

<div align="center">

Fig.6 Control points setting for the stern contour

</div>

## 2.2.1 首轮廓线控制顶点分布模型

如图5所示，图中共有14个控制顶点（12个独立位置），分为以下3类：

1) 边界控制顶点 $V_{1\mathrm{f}}$ ， $V_{5\mathrm{f}-6\mathrm{f}}$ ， $V_{10\mathrm{f}-11\mathrm{f}}$ 和 $V_{14\mathrm{f}}$ 。其中 $V_{1\mathrm{f}}$ 与平底线的首端点重合， $V_{14\mathrm{f}}$ 与甲板中心线的首端点重合， $V_{5\mathrm{f}-6\mathrm{f}}$ 控制球艏最前端的延伸长度 $L_{\mathrm{bf}}$ ， $V_{10\mathrm{f}-11\mathrm{f}}$ 控制球艏和首轮廓线悬伸部分的连接。后两者为二重控制顶点，并分别与第2)类中相应的切矢控制顶点控制曲线的切矢。对于特殊的球艏前端较平的首轮廓线，可以将 $V_{5\mathrm{f}-6\mathrm{f}}$ 分开设置，并使其连线平行于Z轴。

2) 切矢控制顶点 $ V_{2\mathrm{f}} $ $ V_{4\mathrm{f}} $ $ V_{7\mathrm{f}} $ $ V_{9\mathrm{f}} $和 $ V_{12\mathrm{f}} $ 。其中 $ V_{2\mathrm{f}} $与 $ V_{1\mathrm{f}} $的连线确定球艏底部的切矢； $ V_{4\mathrm{f}} $ $ V_{7\mathrm{f}} $与重顶点 $ V_{5\mathrm{f}-6\mathrm{f}} $结合，控制球艏最前端曲线的切矢方向与Z轴平行；同理， $ V_{9\mathrm{f}} $ $ V_{12\mathrm{f}} $与重顶点

$ V_{1 0 \mathrm{f}-1 1 \mathrm{f}} $结合，控制球艏与首轮廓线悬伸部分的连接曲线段有一段与Z轴平行。

3) 形状控制顶点 $ V_{3\mathrm{f}} $ $ V_{8\mathrm{f}} $和 $ V_{13\mathrm{f}} $ 。其中 $ V_{3\mathrm{f}} $和 $ V_{8\mathrm{f}} $与 $ V_{2\mathrm{f}} $ $ V_{4\mathrm{f}} $ $ V_{7\mathrm{f}} $ $ V_{9\mathrm{f}} $结合，可以通过调整得到多种形状的球艏轮廓线； $ V_{13\mathrm{f}} $与 $ V_{12\mathrm{f}} $和 $ V_{14\mathrm{f}} $结合，控制悬伸部分曲线的形状。

## 2.2.2 尾轮廓线控制顶点分布模型

如图6所示，图中共有19个控制顶点（12个独立位置），分为以下4类：

1) 边界控制顶点 $V_{1\mathrm{a}}$ ， $V_{4\mathrm{a}-6\mathrm{a}}$ ， $V_{7\mathrm{a}-9\mathrm{a}}$ ， $V_{12\mathrm{a}-13\mathrm{a}}$ 和 $V_{19\mathrm{a}}$ 。其中 $V_{1\mathrm{a}}$ 与平底线的尾端点重合， $V_{19\mathrm{a}}$ 与甲板中心线的尾端点重合， $V_{4\mathrm{a}-6\mathrm{a}}$ 和 $V_{7\mathrm{a}-9\mathrm{a}}$ 控制尾轴出口形状， $V_{12\mathrm{a}-13\mathrm{a}}$ 控制球艉最尾段延伸长度 $L_{\mathrm{ba}}$ ，并控制球艉和尾轮廓线悬伸部分的连接。

2) 切矢控制顶点 $V_{2\mathrm{a}}$ ， $V_{11\mathrm{a}}$ 和 $V_{14\mathrm{a}}$ 。其中 $V_{1\mathrm{a}}$ 与 $V_{2\mathrm{a}}$ 的连线确定球艉底部的切矢； $V_{11\mathrm{a}}$ 和 $V_{14\mathrm{a}}$ 与 $V_{12\mathrm{a}-13\mathrm{a}}$ 结合，控制球艉和尾轮廓线悬伸部分的连接曲线段有一段与Z轴平行。

3) 形状控制顶点 $V_{3\mathrm{a}}$ ， $V_{10\mathrm{a}}$ 和 $V_{15\mathrm{a}}$ 。其中 $V_{3\mathrm{a}}$ 和 $V_{10\mathrm{a}}$ 与 $V_{2\mathrm{a}}$ ， $V_{4\mathrm{a}-6\mathrm{a}}$ ， $V_{7\mathrm{a}-9\mathrm{a}}$ ， $V_{11\mathrm{a}}$ 相结合，可以通过调整得到多种形状的球艉轮廓线； $V_{14\mathrm{a}}$ 和 $V_{15\mathrm{a}}$ 与 $V_{16\mathrm{a}-18\mathrm{a}}$ 结合，控制悬伸部分曲线的形状。

4) 艉封板平面控制顶点 $ V_{1 6 \mathrm{a}-1 8 \mathrm{a}} $和 $ V_{1 9 \mathrm{a}} $。三重控制顶点 $ V_{1 6 \mathrm{a}-1 8 \mathrm{a}} $与艉封板最低点重合，其与 $ V_{1 9 \mathrm{a}} $决定了艉封板最低点以上部分的轮廓线是直线，目的是保证船体曲面设计时艉封板为平面。

## 2.2.3 首尾轮廓线控制参数

首轮廓线的主要控制参数如下：

球艏长度 $ L_{\mathrm{bf}} $球艏高度 $ H_{\mathrm{bf}} $首轮廓线悬伸部分长度 $ O H_{\mathrm{f}} $尾轮廓线的主要控制参数如下：艉轴中心线高度 H艉柱轴毂高度 h球艉长度 $ L_{\mathrm{ba}} $球艉高度 $ H_{\mathrm{ba}} $尾轮廓线悬伸部分长度 $ O H_{\mathrm{a}} $

## 2.2.4 首尾轮廓线的确定方法

首先，根据首尾轮廓线的控制参数确定轮廓线的边界控制顶点位置，然后再调整切矢和形状控制顶点的分布，以得到满足设计意图的首尾轮廓线。首尾轮廓线确定后，待设计水线的特征参数 $ L_{\mathrm{wf}} $也随之确定。

## 2.3 最大横剖线

常见的最大横剖线形式按舭部形状的不同主要分为圆舭型、斜底型+圆舭型、椭圆舭型，它们形状简单，可以通过先确定其形状参数，然后再根据形状特征给出以吃水为变量的分段函数来确定形状。下面以圆舭型最大横剖线为例来说明函数形式，如图7所示。

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260918141537ea50f2a0a8884acf%2Fcrop_1_1789712182986.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=Kyv%2FFF7faUK%2Fuag3%2FU5yZfHzWaQ%3D&Expires=1790316982' alt='OCR图片'/></div>

<div align="center">

图7 圆舭型舯剖面

</div>

<div align="center">

Fig.7 Round bilge type of midship section

</div>

圆舭型最大横剖线的形状参数只有一个，就是图7所示的圆弧半径 R。其半宽关于吃水的函数可以写成

$$
B _ {\mathrm {w f}} (z) = \left\{ \begin{array}{l l} B / 2, & R \leqslant z \leqslant T \\ \sqrt {2 R z - z ^ {2}} + (B / 2) - R, & 0 \leqslant z < R \end{array} \right.
$$

式中：B为型宽；T为设计吃水。其他形式的最大横剖线也可以通过给出以吃水为自变量的函数形式来确定 $ B_{\mathrm{wf}} $ ，这里不再赘述。

## 2.4 平边线

平边线与水线前体的交点对应水线前体的平直段 $L_{\mathrm{pf}}$ 。平直段长度对船型的光顺性影响不大，因此在初步设计阶段可以按参数多项式表示为 $[9]$ $L_{\mathrm{pf}}(z) = L_{\mathrm{pf0}} + 0.5\left(L_{\mathrm{pfd}} - L_{\mathrm{pf0}}\right)\left(3(z / T) - 2(z / T)^{3} + (z / T)^{4}\right)$

其中形状参数为：平底线首部平行中体长度 $ L_{\mathrm{pf0}} $和设计水线首部平行中体长度 $ L_{\mathrm{pfd}} $ 。在设计时，可以先按照公式值绘出平边线，然后再逐渐调整平边线以满足设计要求。

## 3 吃水函数

吃水函数的选择对船型的光顺性影响很大，文献[9]在文献[12]的基础上对吃水函数进行了引申。本文先以文献[9]中设计的吃水函数的值作为基础值，绘制吃水函数曲线，并在此基础上，根据待设计的船体型线形状特点调整吃水函数曲线形状，反复调整，直至得到满意的型线。其吃水函数的数学公式如下所示。

## 3.1 进去流角 i

进去流角的垂向函数采用二次多项式方程 $ ^{[9]} $：

$$
i (z) = c _ {1} + c _ {2} \left(z / T\right) + c _ {3} \left(z / T\right) ^ {2}
$$

确定系数 $c_{1}, c_{2}, c_{3}$ 的形状控制参数为： $i = i_{j}$ （ $j = 1,2,3$ ），并且 $i(z) = \left(L_{\mathrm{wf}}(z) - L_{\mathrm{pf}}(z)\right) I_{\mathrm{f}}(z) / B_{\mathrm{wf}}(z)$ 。

## 3.2 圆弧半径r

首圆弧半径曲线垂向函数采用三次多项式方程 $ ^{[9]} $：

$$
r (z) = c _ {1} + c _ {2} \left(z / T\right) + c _ {3} \left(z / T\right) ^ {2} + c _ {4} \left(z / T\right) ^ {3}
$$

确定系数 $ c_{1}, c_{2}, c_{3}, c_{4} $的形状控制参数为： $ r=r_{j} $ （ $ j=1,2,3,4 $ ），并且 $ r(z)=(L_{\mathrm{wf}}(z)-L_{\mathrm{pf}}(z))R_{\mathrm{f}}(z) / B_{\mathrm{wf}}^{2}(z) $ 。

## 3.3 水线面系数 $ C_{\mathrm{w}} $

水线面系数垂向函数是对光顺性能影响较大的吃水函数，可以按下面的方程粗略处理 $ ^{[9]} $：

$$
C _ {\mathrm {w f}} (z) = c _ {1} + c _ {2} (z / T) + c _ {3} (z / T) ^ {2}
$$

确定系数 $ c_{1} $ $ c_{2} $ $ c_{3} $ 的形状控制参数为：平底线的水线面系数 $ C_{\mathrm{wf0}} $ 和设计水线的水线面系数 $ C_{\mathrm{wfd}} $ ，以及

$$
C _ {\mathrm {b f}} = \int_ {0} ^ {T} \frac {2 C _ {\mathrm {w f}} (z) L _ {\mathrm {w f}} (z) B _ {\mathrm {w f}} (z)}{L _ {\mathrm {p p f}} B T} \mathrm {d} z
$$

式中： $ C_{\mathrm{b f}} $为船舶前体的方形系数； $ L_{\mathrm{p p f}} $为艏垂线至船舯的距离。

## 3.4 水线面形心位置

## 1）水线面形心距船舯的距离。

水线面形心的垂向函数是对光顺性能影响较大的吃水函数，可以按下面的方程粗略处理 $ ^{[9]} $：

$$
c g _ {\mathrm {w f}} (z) = c _ {1} + c _ {2} (z / T) + c _ {3} (z / T) ^ {2}
$$

确定系数 $ c_{1}, c_{2}, c_{3} $的形状控制参数为：平底线的形心距船舯的距离 $ C G_{\mathrm{f0}} $和设计水线的形心距船舯的距离 $ C G_{\mathrm{fd}} $，以及

$$
L _ {\mathrm {c b f}} = \frac {1}{C _ {\mathrm {b f}}} \int_ {0} ^ {T} \frac {2 c g _ {\mathrm {w f}} (z) C _ {\mathrm {w f}} (z) L _ {\mathrm {w f}} ^ {2} (z) B _ {\mathrm {w f}} (z)}{L _ {\mathrm {p p f}} ^ {2} B T} \mathrm {d} z
$$

式中， $ L_{\mathrm{cbf}} $ 为船舶前体的浮心距船舯的距离除以艏垂线至船舯的距离 $ L_{\mathrm{ppf}} $ ，并且 $ cg_{\mathrm{wf}}(z)=CG_{\mathrm{wf}}(z) / L_{\mathrm{wf}}(z) $ 。

## 2）半水线面形心距中线的距离。

半水线面形心的垂向函数是对光顺性能影响较大的吃水函数，可以按下面的方程粗略处理：

$$
c g _ {\mathrm {b w f}} (z) = c _ {1} + c _ {2} (z / T) + c _ {3} (z / T) ^ {2}
$$

确定系数 $c_{1}, c_{2}, c_{3}$ 的形状控制参数为：平底线的形心半宽 $CG_{\mathrm{bwf0}}$ 和设计水线的形心半宽 $CG_{\mathrm{bwfd}}$ ，以及任一水线的形心半宽，并且 $cg_{\mathrm{bwf}}(z) =$

$ C G_{\mathrm{bwf}}(z) / B_{\mathrm{wf}}(z) $

## 4 设计流程

船体曲面的生成分为以下几个步骤：

1) 根据确定的主尺度和最大横剖线类型，确定最大横剖线形状；

3) 根据第2.1节所述，确定平底线和设计水线形状；

2) 根据第2.2节所述，确定首尾轮廓线形状；

4) 根据第2.4节所述，确定平边线形状；

5) 由前4步确定的特征线，确定第3节中各吃水函数的系数；

6) 由确定的吃水函数，得到特定吃水下水线的几何设计目标和几何约束信息，并根据第2.2节所述建立曲线逼近模型，运用进化算法进行求解；

7) 根据步骤6)得到的水线族蒙皮生成船舶曲面。

## 5 设计结果

图8所示为7000t散货船设计水线（7000WL）的设计结果以及控制顶点的分布。表1列出了设计目标值（水线自由曲线段面积和水线形心位置）、设计逼近数据以及逼近水线与设计目标间的偏差，其中 $ |Error| $为行为设计目标与近似水线特征参数的相对误差。表2列出了逼近水线的定义数据（控制顶点坐标及权因子），其中x行和y行分别为控制顶点的x，y坐标， $ \omega $行对应控制点的权值。由相对误差可知，运用本文的逼近模型进行求解，在满足工程精度要求的前提下可以得到满足设计人员设计意图的设计结果。

<div align="center">

图8设计水线（7000WL）设计结果及控制顶点分布

</div>

<div align="center">

Fig.8 Designed waterline（7000 WL）design results and control points distribution

</div>

<div align="center">

表1 设计水线（7000WL）的设计目标、设计逼近数据及其相对误差

</div>

<div align="center">

Table 1 Design objective,approximate design data and errors of the designed waterline (7000 WL)

</div>

<table border="1"><tr><td></td><td colspan="3">Forebody</td><td colspan="3">Afterbody</td></tr><tr><td></td><td>Awf/m2</td><td>CGxf/m</td><td>CGyf/m</td><td>Awa/m2</td><td>CGxa/m</td><td>CGya/m</td></tr><tr><td>Design objective</td><td>168.123867</td><td>33.957682</td><td>3.556745</td><td>187.357002</td><td>37.655673</td><td>3.514651</td></tr><tr><td>Approximate design data</td><td>168.123879</td><td>33.953542</td><td>3.556738</td><td>187.361427</td><td>37.655992</td><td>3.514643</td></tr><tr><td>|Error|×1000</td><td>0.0001</td><td>0.1219</td><td>0.0020</td><td>0.0236</td><td>0.0085</td><td>0.0023</td></tr></table>

<div align="center">

表2 设计水线（7000WL）的控制顶点坐标及权因子

</div>

<div align="center">

Table 2 Control points and weights of the designed waterline (7000 WL)

</div>

<table border="1"><tr><td colspan="3">Forebody</td><td colspan="3">Afterbody</td></tr><tr><td>x</td><td>y</td><td>ω</td><td>x</td><td>y</td><td>ω</td></tr><tr><td>50.764</td><td>0.405</td><td>1</td><td>53.799</td><td>3.348</td><td>1</td></tr><tr><td>46.868</td><td>2.715</td><td>17.5</td><td>45.618</td><td>5.463</td><td>33.116</td></tr><tr><td>39.153</td><td>6.894</td><td>38.466</td><td>40.420</td><td>7.300</td><td>31.292</td></tr><tr><td>28.916</td><td>8.6</td><td>18.161</td><td>28.793</td><td>8.6</td><td>29.995</td></tr><tr><td>23.257</td><td>8.6</td><td>1</td><td>25.5</td><td>8.6</td><td>1</td></tr><tr><td>0</td><td>8.6</td><td>1</td><td>0</td><td>8.6</td><td>1</td></tr></table>

图9给出了根据本文方法得到的船舶水线及其控制顶点分布，表3给出了整个船体型线的逼近设计结果，表4给出了整个船体水线控制顶点的坐标及对应的权因子。

图10给出了根据图9截面线蒙皮生成的船体曲面。

从图10可以看出，由本文方法得到的曲面是初光顺的，造成其些微不光顺的原因是吃水函数中形状参数的取值不够协调。因此，下一步的研究工作将考虑如何快速得到协调的形状参数，或更科学的吃水函数。

<div align="center">

（a）Perspective view

</div>

<div align="center">

(b) Front view

</div>

<div align="center">

图9 水线设计结果及其控制点分布

</div>

<div align="center">

Fig.9 Waterline design results and control points distribution

</div>

<div align="center">

图10 船体曲面

</div>

<div align="center">

Fig.10 Hull surface

</div>

<div align="center">

表3 整个船体型线的设计目标、设计逼近数据及其相对误差

</div>

<div align="center">

Table 3 Design objective, approximate design data and errors of the designed hull lines

</div>

<table border="1"><tr><td rowspan="2"></td><td rowspan="2">Waterline height/m</td><td colspan="3">Forebody</td><td colspan="3">Afterbody</td></tr><tr><td>$A_{wf}/m^{2}$</td><td>$CG_{xf}/m$</td><td>$CG_{yf}/m$</td><td>$A_{wa}/m^{2}$</td><td>$CG_{xa}/m$</td><td>$CG_{ya}/m$</td></tr><tr><td rowspan="3">0</td><td>Design objective</td><td>143.331317</td><td>22.774</td><td>2.547</td><td>140.0489</td><td>22.071</td><td>2.656</td></tr><tr><td>Approximate design data</td><td>143.461342</td><td>22.773</td><td>2.550</td><td>140.052</td><td>22.093</td><td>2.650</td></tr><tr><td>|Error|/%</td><td>0.091</td><td>0.004</td><td>0.113</td><td>0.002</td><td>0.1</td><td>0.19</td></tr><tr><td rowspan="3">0.5</td><td>Design objective</td><td>202.543392</td><td>24.669</td><td>3.139</td><td>193.1818</td><td>23.585</td><td>3.245</td></tr><tr><td>Approximate design data</td><td>202.514694</td><td>24.601</td><td>3.154</td><td>193.5315</td><td>23.595</td><td>3.266</td></tr><tr><td>|Error|/%</td><td>0.014</td><td>0.272</td><td>0.489</td><td>0.18</td><td>0.04</td><td>0.66</td></tr><tr><td rowspan="3">1</td><td>Design objective</td><td>227.761633</td><td>25.501</td><td>3.356</td><td>209.3561</td><td>23.887</td><td>3.449</td></tr><tr><td>Approximate design data</td><td>227.855187</td><td>25.455</td><td>3.358</td><td>209.2084</td><td>23.890</td><td>3.454</td></tr><tr><td>|Error|/%</td><td>0.041</td><td>0.183</td><td>0.046</td><td>0.07</td><td>0.01</td><td>0.14</td></tr><tr><td rowspan="3">2</td><td>Design objective</td><td>254.651539</td><td>26.553</td><td>3.554</td><td>224.0142</td><td>24.329</td><td>3.624</td></tr><tr><td>Approximate design data</td><td>255.067469</td><td>26.376</td><td>3.564</td><td>225.0527</td><td>24.351</td><td>3.610</td></tr><tr><td>|Error|/%</td><td>0.163</td><td>0.666</td><td>0.290</td><td>0.46</td><td>0.09</td><td>0.39</td></tr><tr><td rowspan="3">3</td><td>Design objective</td><td>202.169826</td><td>31.044</td><td>3.419</td><td>158.1443</td><td>29.42</td><td>3.442</td></tr><tr><td>Approximate design data</td><td>203.211047</td><td>31.022</td><td>3.452</td><td>158.0617</td><td>29.417</td><td>3.443</td></tr><tr><td>|Error|/%</td><td>0.515</td><td>0.071</td><td>0.960</td><td>0.05</td><td>0.01</td><td>0.02</td></tr><tr><td rowspan="3">4</td><td>Design objective</td><td>193.667993</td><td>32.434</td><td>3.448</td><td>149.2842</td><td>31.001</td><td>3.478</td></tr><tr><td>Approximate design data</td><td>193.600563</td><td>32.245</td><td>3.452</td><td>149.3225</td><td>30.918</td><td>3.490</td></tr><tr><td>|Error|/%</td><td>0.035</td><td>0.585</td><td>0.128</td><td>0.03</td><td>0.27</td><td>0.33</td></tr><tr><td rowspan="3">5</td><td>Design objective</td><td>179.187594</td><td>32.550</td><td>3.540</td><td>151.0786</td><td>32.908</td><td>3.431</td></tr><tr><td>Approximate design data</td><td>179.175368</td><td>32.594</td><td>3.521</td><td>150.8809</td><td>32.865</td><td>3.440</td></tr><tr><td>|Error|/%</td><td>0.007</td><td>0.136</td><td>0.539</td><td>0.13</td><td>0.13</td><td>0.27</td></tr><tr><td rowspan="3">6</td><td>Design objective</td><td>171.418285</td><td>33.361</td><td>3.506</td><td>172.988189</td><td>35.939</td><td>3.351</td></tr><tr><td>Approximate design data</td><td>171.472260</td><td>33.341</td><td>3.511</td><td>172.987932</td><td>35.928</td><td>3.355</td></tr><tr><td>|Error|/%</td><td>0.031</td><td>0.062</td><td>0.149</td><td>0.00</td><td>0.03</td><td>0.12</td></tr></table>

<div align="center">

表4 水线控制顶点坐标及权因子

</div>

<div align="center">

Table 4 Control points and weights of the waterline

</div>

<table border="1"><tr><td colspan="2">Waterline height/m</td><td colspan="7">Control points and weights</td></tr><tr><td rowspan="7">0</td><td rowspan="3">Forebody</td><td>x</td><td>46.175</td><td>41.575</td><td>24.842</td><td>18.917</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.265</td><td>0.950</td><td>5.099</td><td>6.6</td><td>6.6</td><td>6.6</td></tr><tr><td>ω</td><td>1</td><td>16.2442</td><td>42.9988</td><td>4.1526</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>40.311</td><td>34.476</td><td>31.086</td><td>19.353</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>1.006</td><td>2.752</td><td>3.765</td><td>6.6</td><td>6.6</td><td>6.6</td></tr><tr><td>ω</td><td>1</td><td>47.8423</td><td>21.0103</td><td>5.564</td><td>1</td><td>1</td></tr><tr><td rowspan="6">0.5</td><td rowspan="3">Forebody</td><td>x</td><td>50.524</td><td>45.355</td><td>27.806</td><td>21.868</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.321</td><td>1.301</td><td>6.349</td><td>7.923</td><td>7.923</td><td>7.923</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>45.119</td><td>40.596</td><td>33.483</td><td>21.924</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.505</td><td>1.932</td><td>5.056</td><td>7.923</td><td>7.923</td><td>7.923</td></tr><tr><td>ω</td><td>1</td><td>23.1767</td><td>33.8524</td><td>7.3927</td><td>1</td><td>1</td></tr><tr><td rowspan="6">1</td><td rowspan="3">Forebody</td><td>x</td><td>51.906</td><td>49.449</td><td>32.952</td><td>24.832</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.321</td><td>0.864</td><td>5.629</td><td>8.332</td><td>8.332</td><td>8.332</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>46.348</td><td>37.270</td><td>33.755</td><td>19.019</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.495</td><td>3.399</td><td>6.309</td><td>8.332</td><td>8.332</td><td>8.332</td></tr><tr><td>ω</td><td>1</td><td>38.1003</td><td>26.0365</td><td>19.6532</td><td>1</td><td>1</td></tr><tr><td rowspan="6">2</td><td rowspan="3">Forebody</td><td>x</td><td>53.017</td><td>50.209</td><td>31.775</td><td>17.057</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.266</td><td>1.030</td><td>8.005</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>48</td><td>47.823</td><td>44.151</td><td>28.688</td><td>10.2</td><td>0</td></tr><tr><td>y</td><td>0.4</td><td>0.455</td><td>0.953</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>13.5384</td><td>31.3506</td><td>8.887</td><td>1</td><td>1</td></tr></table>

<div align="center">

表4（续）

</div>

<table border="1"><tr><td colspan="2">Waterline height/m</td><td colspan="7">Control points and weights</td></tr><tr><td rowspan="7">3</td><td rowspan="3">Forebody</td><td>x</td><td>53.280</td><td>44.236</td><td>34.814</td><td>24.662</td><td>17.786</td><td>0</td></tr><tr><td>y</td><td>0.246</td><td>3.698</td><td>6.900</td><td>8.600</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>47.514</td><td>43.146</td><td>32.692</td><td>21.046</td><td>19.089</td><td>0</td></tr><tr><td>y</td><td>0.015</td><td>1.794</td><td>7.669</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>15.1163</td><td>37.1464</td><td>41.1599</td><td>1</td><td>1</td></tr><tr><td rowspan="6">4</td><td rowspan="3">Forebody</td><td>x</td><td>52.954</td><td>52.061</td><td>49.354</td><td>28.506</td><td>19.652</td><td>0</td></tr><tr><td>y</td><td>0.210</td><td>0.612</td><td>2.577</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>47.402</td><td>45.539</td><td>36.730</td><td>25.694</td><td>21.275</td><td>0</td></tr><tr><td>y</td><td>0.015</td><td>0.794</td><td>6.486</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>44.6423</td><td>43.8399</td><td>12.6476</td><td>1</td><td>1</td></tr><tr><td rowspan="6">5</td><td rowspan="3">Forebody</td><td>x</td><td>51.492</td><td>48.154</td><td>38.673</td><td>28.815</td><td>21.131</td><td>0</td></tr><tr><td>y</td><td>0.212</td><td>1.974</td><td>6.287</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>49.714</td><td>49.708</td><td>36.819</td><td>26.510</td><td>23.008</td><td>0</td></tr><tr><td>y</td><td>0.015</td><td>0.017</td><td>7.096</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>32.9863</td><td>36.8404</td><td>14.0571</td><td>1</td><td>1</td></tr><tr><td rowspan="6">6</td><td rowspan="3">Forebody</td><td>x</td><td>50.845</td><td>44.121</td><td>35.617</td><td>27.522</td><td>22.289</td><td>0</td></tr><tr><td>y</td><td>0.268</td><td>4.156</td><td>7.448</td><td>8.600</td><td>8.600</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>29.8216</td><td>22.8074</td><td>2.4959</td><td>1</td><td>1</td></tr><tr><td rowspan="3">Afterbody</td><td>x</td><td>53.752</td><td>50.878</td><td>46.074</td><td>32.243</td><td>24.366</td><td>0</td></tr><tr><td>y</td><td>1.753</td><td>2.601</td><td>3.948</td><td>8.6</td><td>8.6</td><td>8.6</td></tr><tr><td>ω</td><td>1</td><td>27.0966</td><td>36.2071</td><td>8.1059</td><td>1</td><td>1</td></tr></table>

## 6 结语

通过分析船舶各个特征线的形状特征，提出了基于NURBS的垂向参数化船型自行设计方法。该方法给出了轻量化的首尾轮廓线和水线的NURBS逼近模型，并在此基础上以吃水函数为纽带，将船型设计转化为特征参数设计和特征线设计。该方法实现了用较少参数来驱动生成设计船型的目的。设计实例表明在船舶初始设计阶段该设计方法可行。后续的研究可以考虑如何在参数化设计船舶特征线时，选择或改进进化算法，以提高设计效率和设计稳定性，也可以总结各个船型的型线特征，给出系列船的吃水函数回归公式，以指导吃水函数曲线形状的调整。

## 参考文献：

[1] PÉREZ F, CLEMENTE J A. Constrained design of simple ship hulls with B-spline surfaces [J]. Computer-Aided Design, 2011, 43(12):1829-1840.

[2] WANG H, ZOU Z J. Geometry modeling of ship hull based on non-uniform B-spline [J]. Journal of Shanghai Jiaotong University (Science), 2008, 13(2): 189-192.

[3] PEREZ-ARRIBAS F. Parametric generation of planning hulls[J]. Ocean Engineering, 2014, 81:89-104.

[4] 钱宏，刘敏，贺庆，等. 基于NURBS曲面插值的船体曲面重构[J]. 中国造船，2016，57(1)：138-148. QIAN H, LIU M, HE Q, et al. Reconstruction of ship hull based on NURBS surface interpolation [J]. Shipbuilding of China, 2016, 57(1):138-148 (in Chinese).

[5] 陆丛红，林焰，纪卓尚. 基于缩减控制顶点数和自适应遗传算法的船体水线NURBS拟合[J].大连理工大学学报，2007，47(6)：846-852. LU C H，LIN Y，JI Z S. NURBS approximation of hull waterline based on reduced control points number with adaptive genetic algorithm[J]. Journal of Dalian University of Technology,2007,47(6):846-852（in Chinese).

[6] LU C H,ZHANG Y R,LIN Y. Representation for characteristic curves of hull form using immune genetic algorithm based on NURBS with path curve parameterization method [C]//Proceedings of the 24th International Ocean and Polar Engineering Conference. Busan, Korea: ISOPE, 2014:535-540.

[7] 于雁云，林焰，纪卓尚. 船体曲面参数化设计新方法 [J]. 中国造船，2013(1):21-29. YU Y Y, LIN Y, JI Z S. A new method for parametric design of hull surface [J]. Shipbuilding of China, 2013 (1):21-29 (in Chinese).

[8] 张萍，朱德祥，何术龙. 参数化的船型设计方法[J]. 中国造船，2008，49(4)：26-35.

（下转第45页）

[5] 李军，李樱，罗白璐. 基于FORAN的结构模型管理方法[J]. 船舶标准化工程师，2012，45(1)：28-31. LI J，LI Y，LUO B L. Method for structure models management based on FORAN[J]. Ship Standardization Engineer，2012，45(1)：28-31(in Chinese).

[6] 李海峰，吴慧中，陈卫东.三维CAD环境下的产品数据管理方法[J].计算机辅助设计与图形学学报，2006，18(3)：464-469. LI H F，WU H Z，CHEN W D. An associative approach to 3D CAD product data management[J]. Journal of Computer-Aided Design and Computer Graphics，2006，18(3)：464-469（in Chinese).

[7] 张荣霞，张树生，周竞涛，等. 基于MBD的零件制造模型管理[J]. 制造业自动化，2011，33(8)：6-9. ZHANG R X，ZHANG S S，ZHOU J T，et al. Part manufacturing model management based on the MBD [J]. Manufacturing Automation，2011，33(8)：6-9 (in Chinese).

[8] 许松林，龚文秀，王惠玲. 基于模块的飞机产品结构管理[J]. 航空工程进展，2013，4(2)：219-225. XU S L，GONG W X，WANG H L. Product structure management of aircraft based on modularity [J]. Advances in Aeronautical Science and Engineering, 2013，4(2)：219-225（in Chinese).

[9] 闵志坤，王时龙，任亨斌，等.一种新型BOM检索方法的研究与实现[J].重庆工学院学报（自然科学），2009，23(1)：109-112. MIN Z K, WANG S L, REN H B, et al. Research on and implementation of a new bom retrieval method[J]. Journal of Chongqing Institute of Technology（Natural Science），2009，23(1)：109-112（in Chinese).

[10] LEE J H, KIM S H, LEE K. Integration of evolutional BOMs for design of ship outfitting equipment [J]. Computer-Aided Design, 2012, 44(3): 253-273.

[11] 马登武，叶文，李瑛. 基于包围盒的碰撞检测算法综述[J]. 系统仿真学报，2006，18（4）：1058-1061，1064.MADW，YEW，LIY.Survey of box-based algorithms for collision detection[J].Journal of System Simulation，2006，18（4）：1058-1061，1064（in Chinese).

[12] 陈阳平，谢强，于春江，等. 基于AABB层次树的数字样机空间区域计算与搜索方法[J]. 南京航空航天大学学报，2009，41(4)：540-544. CHEN Y P, XIE Q, YU C J, et al. Spatial region calculation and search method for digital mock-up based on axis aligned bounding box（AABB）hierarchical tree[J]. Journal of Nanjing University of Aeronautics and Astronautics，2009，41(4)：540-544（in Chinese).

[13] 陶松桥. 面向设计的三维CAD模型搜索技术研究 [D]. 武汉：华中科技大学，2012. TAO S Q. Research on design oriented CAD model retrieval [D]. Wuhan：Huazhong University of Science and Technology，2012（in Chinese).

[14] 王占松，田凌. 基于功能的三维模型检索系统[J].计算机辅助设计与图形学学报，2013，25(12)：1877-1885.WANG Z S，TIAN L. Function-based 3D model retrieval system[J].Journal of Computer-Aided Design and Computer Graphics，2013，25(12)：1877-1885（in Chinese).

||||||||||

（上接第37页）

ZHANG P, ZHU D X, HE S L. The parametric design method of hull form [J]. Shipbuilding of China, 2008, 49(4):26-35 (in Chinese).

[9] 李彦本. 数学船型的设计研究[D]. 大连：大连理工大学，1997. LI Y B. Mathematical hull-line design research [D]. Dalian: Dalian University of Technology, 1997 (in Chinese).

[10] PIEGL L, TILLER W. The NURBS book [M]. London:

Springer-Verlag, 1995:133-158.

[11] 王运龙，王晨，彭飞，等. 基于人机结合遗传算法的船舶管路三维布局优化设计[J]. 中国造船，2015， 56(1):196-202. WANG Y L, WANG C, PENG F, et al. Intelligent layout design of ship pipes based on genetic algorithm with human-computer cooperation [J]. Shipbuilding of China, 2015, 56(1):196-202 (in Chinese).

[12] KUIPER G. Preliminary design of ship lines by mathematical methods[J]. Journal of Ship Research, 1970, 14(1):52-66.