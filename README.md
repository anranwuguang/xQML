# xQML-SZ
## 1. 安装与声明
### 1.1 软件安装
我们的代码主要依赖于healpix, xQML, ymaster软件包, 对于xQML软件包这里我们给出了适用的版本.

Please install HEALPix, NaMaster and xQML code first. For xQML code we have given the source code of the appropriate version.

Documents:
- NaMaster: https://namaster.readthedocs.io/en/latest/
- xQML:     https://gitlab.in2p3.fr/xQML/xQML
- HEALPix:  https://healpix.sourceforge.io/

### 1.2 源码错误与修改:

/xQML-master/src/libcov.c
Line 221,Line 229 在计算TE, TB部分P02的表达式中缺少了 $\frac{2l+1}{4\pi}$ 这一因子，应做如下修改:

           P02 = -d20[l]; → P02 = norm*(-d20[l]);
