Review:
```
1. Atomic-symbols : one-hot represnetation -> High dimension
2. Low dimension dense word vector
   ├── 2-(1) dimension reduction
   ├── 2-(2) directly learning low-dimension word vectors -> word embedding
```
此篇主要是紀錄 2-(2) 之主題，前面有提到，使用：
* Knowledge based 會有高成本、更新慢、難以量化的問題。
* Corpus based 中之 one-hot representation 容易有維度過大問題且因為向量相互獨立而無法計算相似度。
* Corpus based 中之 dimension reduction 則有計算複雜度高的問題，且若更改矩陣將必須重複計算。


```bash
基本想法：
透過 Network，我們可以直接從資料中學出 word representation，而此representation是一個相對較低維度的vector，
能對有效表達此 word。
```

```bash
Word Embedding:
此類型方法之所以稱為 word embedding，是因為我們會在Vector space(Embedding space，ex: 300維)尋找一個點，
而此點就代表了這個 word， 即把 word 嵌(embedded)在此space上。
```

Word Embedding 之使用主要可分為：
1. 由設定隨機之向量(矩陣)出發，並根據 target-test 不段更新參數而學得對應之word embedding。
2. 使用 Pretrained word embedding，此 word embedding 則相對來說較為通用(generic)。
3. 結合 1&2 之概念，在初始即使用 Pretrained 之 word embedding，並同時根據 target-test 去更新參數，此稱作 fine-tuning。

------------------------------------------------------------------------------
(1) Learn by target-test:

此種方法是根據你當初建構 Network 時為了解決的問題(ex: Sentimental Analysis, Seq2Seq...)去學習參數，在深度學習框架中，都提供很方便的 API 可以幫我們完成 word embedding。

```bash
假設我們現在想表達一個簡短的句子:"I love deep learning"，且假設在 Corpus 中，共有100個相異的單字。
故我們可以建構一個簡單的 word_to_index dictionary。
大略可表示成：
{
  "I":1,
  "am":2,
  "love":3,
  .....
}
其中，'I love deep learning'四字分別被對應到[1, 3, 5, 7]之value。
接著，我們可將各個字由 index 藉由 word embedding weight matrix 進行轉換。
```
這邊示範一下在 Keras、Pytorch 下使用 word embedding：

In Keras:
```python
>>> import numpy as np
>>> from keras.models import Sequential
>>> from keras.layers import Embedding

>>> input_array = np.array([[1, 3, 5, 9]])
>>> model = Sequential() # Build the model
>>> model.add(Embedding(input_dim=100, output_dim=5)) # Add the Embedding layer
>>> model.predict(input_array)

array([[[ 0.00102774, -0.00858252, -0.01779232,  0.01967026, -0.0355065]],
       [[-0.03129957, -0.02744168, -0.03758506, -0.02883364, 0.03076198]],
       [[ 0.02777207,  0.04753191, -0.01752094,  0.00322759, 0.04345156]],
       [[ 0.00882687,  0.04097025, -0.04970786, -0.03205182, 0.03969387]]], dtype=float32)
```

```bash
稍微紀錄一下上面的結果：

Arguments:
(1)input_dim: 此為 Corpus 中單字的數量，即為 vocab_size。
(2)output_dim: 此為你設定的 embedding dimension，即你想用幾維度去有效表達word。

由此例，因我們的Corpus共有100個相異單字，故設定input_dim = 100。
Note:
這邊要特別注意，作為 model 輸入的input index值是不可小於input_dim的。

查看模型結果:
output size == (1, 4, 5)
分別的意義：
1 -> 共有一個row(我們只有一句話'I love deep learning')
4 -> 一個row中，透過embedding weights轉換的字數(len([1, 3, 5, 9]) == 4)
5 -> 即為 embedding dimension

Note:
* 還有另一個經常被使用的參數為: input_length，其實他相當直覺，即為你input中每一個row的長度。
  故經常會將 array 在作為 input前，由keras.preprocessing 中的 sequence.pad_sequences 先進行預處理。

  假設我將 Embedding 設定為 Embedding(input_dim=100, output_dim=5, input_length=7)，
  因len([1, 3, 5, 9]) == 4 而會導致 ValueError。

* 在此皆是以'numpy.ndarray'物件作為 Keras model的input。
```

In Pytorch:
```python
>>> import torch
>>> import torch.nn as nn

>>> embedding = nn.Embedding(100, 5)
>>> input_torch = torch.tensor([[1, 3, 5, 7]])
>>> embedding(input_torch)

tensor([[[-0.6092, -0.9798, -1.6091, -0.7121,  0.3037],
         [ 0.4676, -0.6970, -1.1608,  0.6995,  0.1991],
         [-0.1759, -2.2456, -1.4465,  0.0612, -0.6177],
         [-0.7735,  0.1991,  0.0457,  0.1530, -0.4757]]],
         grad_fn=<EmbeddingBackward>)
```

```bash
在這邊也用了與 Keras 中一樣的範例。
僅有一個特別要注意的是:在 Pytorch 中是以 tensor 作為運算，但其概念跟 ndarray 其實非常像。
```
