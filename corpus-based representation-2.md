Review:
```
1. Atomic-symbols : one-hot represnetation -> High dimension
2. Low dimension dense word vector
   ├── 2-(1) dimension reduction
   ├── 2-(2) directly learn low-dimension word vectors
```
在前一篇 Atomic-symbols 及他遇到的困難及解決方式，而此篇主要討論 Low dimension dense word vector。
------------------------------------------------------------------------------
2-(1) Dimension reduction
<br>降維的方法有很多種(例:PCA、SVD、tsne等等...)，而此處要討論的是利用SVD，將前一篇所建構的 Co-occurence matrix 進行維度縮減。

SVD(Singular Value Decomposition): 奇異值分解
```bash
SVD 能夠對'任意矩陣'拆解為3個矩陣的乘積。X 拆解為 U、S、V三者，其中 U、V 為 orthogonal matrix，而 S 為對角矩陣。

在對角矩陣 S 中，Singular value 會按照大小以遞減排序，Singular value 可當作是對應基底(新的座標軸)的重要程度。
故我們可由S矩陣之倒數幾項開始刪減一些相對較不重要的元素。
```
