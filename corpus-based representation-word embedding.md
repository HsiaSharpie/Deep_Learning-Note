Review:
```
1. Atomic-symbols : one-hot represnetation -> High dimension
2. Low dimension dense word vector
   ├── 2-(1) dimension reduction
   ├── 2-(2) directly learning low-dimension word vectors -> word embedding
```
此篇主要是紀錄 2-(2) 之主題。
<br>前面有提到，使用：
* Knowledge based 會有高成本、更新慢、難以量化的問題。
* Corpus based 中之 one-hot representation 容易有維度過大問題且因為向量相互獨立而無法計算相似度。
* Corpus based 中之 dimension reduction 則有計算複雜度高的問題，且若更改矩陣將必須重複計算。

基本想法：
```bash
透過 Network，我們可以直接從資料中學出 word representation，而此word representation是一個相對較低維度的vector，
能對 word 有效表達。
```

Word Embedding ?
```bash
此類型方法之所以稱為 word embedding，是因為我們會在Vector space(Embedding space，ex: 300維)尋找一個點，
而此點就代表了這個 word， 即把 word 嵌(embedded)在此space上。
```
