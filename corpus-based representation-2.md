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

在對角矩陣 S 中，Singular value 會按照大小以遞減排序，Singular value 可當作是對應基底(新的座標軸)的重要程度，我們可由S矩陣之倒數幾項開始刪減後方一些相對較不重要的元素。
```

對上篇的 Co-occurence matrix 進行 SVD 拆解並降維(ex:降到2維)，我是直接使用 numpy 中的 linalg(linear algebra)模組做SVD。
```python
U, S, V = np.linalg.svd(co_matrix)
```

```bash
# vector of 'is' in Co-occurence matrix
>>> is_vec = co_matrix[token_to_id['is']]
>>> print(is_vec)

[0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0.]

# vector of 'is' in SVD matrix
>>> print(U[1])

[-1.00000000e+00  0.00000000e+00  1.32799596e-16  0.00000000e+00
  6.16297582e-33  0.00000000e+00 -4.62223187e-33  0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00]

由以上，可發現利用SVD，把原本較為稀疏的 is_vec 轉換成較稠密的 U[1]。
若要將稠密向量進行降維，可以如以上所說的取前幾個維度即可(此處取n=2)，
並可依降維後的結果投影到二維，觀察字詞間的相關性。
```
