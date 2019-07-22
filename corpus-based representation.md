(2)Corpus-based representation:
```bash
Corpus(語料庫)其實就是大量的文本資料。
```
Corpus 中儲存的文章為由人類撰寫而成，故在其中涵蓋大量自然語言的常識 & 知識。故我們可從 Corpus 中透過 learning 萃取出精華。

而在 Corpus-based representation 中，主要分為兩大類：
```
1. Atomic-symbols : one-hot represnetation -> High dimension
2. Low dimension dense word vector
   ├── 2-(1) dimension reduction
   ├── 2-(2) directly learn low-dimension word vectors
```
------------------------------------------------------------------------------
(1) Atomic-symbols : one-hot represnetation

此方法就是有名的 one-hot encoding，將一個字詞以一個非常高維度的 vector 表示，而此 vector 之維度即為len(corpus)。
Example:
<br>假設我們有下面幾個句子，而我們把它丟入陣列作為元素:
```bash
corpus = [
    'he is a man',
    'she is a woman',
    'Taipei is a beautiful city',
    'Tokyo is a crowded city',
]
```
