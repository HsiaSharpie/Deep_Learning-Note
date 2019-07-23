Review:
```
1. Atomic-symbols : one-hot represnetation -> High dimension
2. Low dimension dense word vector
   ├── 2-(1) dimension reduction
   ├── 2-(2) directly learning low-dimension word vectors -> word embedding
```
此篇主要是紀錄 2-(2) 之主題。
<br>前面有提到，使用：
* Knowledge based 會有高層本、更新慢、難以量化的問題。
* Corpus based 中之 one-hot representation 容易有維度過大問題且因為向量相互獨立而無法計算相似度。
* Corpus based 中之 dimension reduction 則有計算複雜度高的問題。

在 2-(2)主題中，最著名的就是NLP大神Tomas Mikolov在2013年所提出的word2vec & 2014年Pennington 提出的Glove。

基本想法：
```bash
透過 NN，我們可以直接 learn 出 word representation，而此 word representation是一個相對較低維度的 vector。
```

Word Embedding ?
```bash
此類型方法之所以稱為 word embedding，是因為我們會在Vector space(Embedding space，ex: 300維)尋找一個點，
而此點就代表了這個 word -> 把 word 嵌(embedded)在此 space上。
```
