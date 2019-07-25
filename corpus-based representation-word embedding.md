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
假設我們現在想表達一個簡短的句子:"I love deep learning"
且假設在 Corpus 中，共有100個相異的單字。
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
<br>In Keras:
```python
>>> import numpy as np
>>> from keras.models import Sequential
>>> from keras.layers import Embedding

>>> input_array = np.array([1, 3, 5, 9])
>>> model = Sequential() # Build the model
>>> model.add(Embedding(input_dim=100, output_dim=5)) # Add the Embedding layer
>>> model.predict(input_array)

array([[[-0.04232415,  0.04877395]],
       [[-0.04796009,  0.04976574]],
       [[ 0.04478942,  0.00876119]],
       [[ 0.01540852, -0.03079422]]], dtype=float32)
```
