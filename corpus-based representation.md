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

此方法就是有名的 one-hot encoding，將一個字詞以一個非常高維度的 vector 表示，而此 vector 之維度即為 corpus 的長度 -> len(corpus)。

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
接著，寫一個簡單的 function，我們能輕易的把字詞透過 split 斷開。在這有兩個小補充：
* 因為此為非常簡短的直述句，斷詞顯得很簡單。但一般在 Corpus 中，我們必須配合 regular expression 才能表現得更佳。
* 在中文，斷詞就顯得不是那麼簡單了。現在中文斷詞最好的 library 為 jieba，它是透過HMM技巧有效將中文字詞斷開。

```python
def tokenize_corpus(corpus):
    return [sentence.split() for sentence in corpus]
```
就會有以下結果：
```bash
>>> tokenized_corpus = tokenize_corpus(corpus)
>>> tokenized_corpus

[['he', 'is', 'a', 'man'],
 ['she', 'is', 'a', 'woman'],
 ['Taipei', 'is', 'a', 'beautiful', 'city'],
 ['Tokyo', 'is', 'a', 'crowded', 'city']]
```
分割完字詞後，接著欲將這些字詞中重複的單字濾出。
```python
def unique_vocabulary(tokenized_corpus):
    vocab = []
    for sentence in tokenized_corpus:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    return vocab
```
同樣地，我們來看後續結果：
```bash
>>> uni_vocab = unique_vocabulary(tokenized_corpus)
>>> uni_vocab

['he',
 'is',
 'a',
 'man',
 'she',
 'woman',
 'Taipei',
 'beautiful',
 'city',
 'Tokyo',
 'crowded']
 ```
