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
字詞處理上，除了簡單以陣列存放字詞，若將字詞設定ID且與字詞形成一對一對應表，再後續有許多使用方便之處(Ex:建構 one-hot encoding...)。
```python
def id_to_token(vocab):
    return {index:word for index, word in enumerate(vocab)}

def token_to_id(vocab):
    return {word:index for index, word in enumerate(vocab)}
```
```bash
>>> token_to_id = token_to_id(uni_vocab)
>>> token_to_id

{'he': 0,
 'is': 1,
 'a': 2,
 'man': 3,
 'she': 4,
 'woman': 5,
 'Taipei': 6,
 'beautiful': 7,
 'city': 8,
 'Tokyo': 9,
 'crowded': 10}
 ```
 好了，終於要針對我們 Corpus 中的 unique 單字建構 one-hot encoding。
 <br> One-hot encoding: 在此我是利用 dictionary 中的 value，將它對應回 vector 之 index，並設定為1，以代表表示此單字。

 ```python
 import numpy as np

def one_hot(word_dict, word):
    one_hot_vector = np.zeros(len(word_dict))

    if word in word_dict:
        value = word_dict[word]
        one_hot_vector[value] = 1

    return one_hot_vector
 ```

 ```bash
 >>> one_hot(a, 'he')

 array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  ```

以上為單字`he`的 one-hot encoding，其他字可以此類推。
<br>但使用 one-hot encoding 有個致命的缺點 : 難以計算字詞間的相似度(similarity)。

```bash
我們假設字詞間為 independent(vector為orthogonal)，故任何兩個字詞之 vector 所算出之 cosine similarity 必為0。
 ```
說白了 one-hot encoding 只讓我們能有效辨識此向量在表達和字詞，但沒辦法表達詞意。
<br>為了解決這個問題，考慮了 distribution hypothesis，他的基本想法是「字詞的詞意是由周圍字詞所形成」。

```bash
Goal:
我們由 distribution hypothesis 出發，透過計算周圍的字詞，建構 Co-occurence matrix。

Thought:
如果2個字詞的意義很相近，則他們將有接近的 Neighors。
 ```

 Definition of Neighbors:
 1. Full document -> 把整個文章當成 neighbor，此方法所得結果偏向於了解文章 Topic。
 2. Windows

由第一種以整個文章作為鄰居方式了解字詞間的關係有點過於粗糙，因而考量上下文(context)作為 neighbor。
<br>在此我假設 window_size = 1 建構一個 Co-occurence matrix。

```python
def co_matrix(uni_vocab, tokenized_corpus):
    vocab_size = len(uni_vocab)
    co_matrix = np.zeros((vocab_size, vocab_size))

    for sentence in tokenized_corpus:
        for index, word in enumerate(sentence[1:], start=1):
            prev_word, curr_word = sentence[index-1], sentence[index]
            prev_word_id, curr_word_id = token_to_id[prev_word], token_to_id[curr_word]

            co_matrix[prev_word_id, curr_word_id] += 1

    return co_matrix
```

```bash
>>> co_matrix(uni_vocab, tokenized_corpus)

array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
 ```
上方為根據 Corpus 所建構出之 Co-occurence matrix，而每一 row/column 所對應的字詞，即為在 token_to_id dictionary 之 value 所對應的 key。
<br>故第一列即為代表`he`的 vector，其餘以此類推。

建構完 Co-occurence matrix後，就可以計算各個 vector 間的相似度啦！
<br> 這邊要用的是前面所提到的 Cosine similarity。
<br> 定義就直接看 Wiki 即可 -> https://zh.wikipedia.org/wiki/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7

以下是 Cosine similarity，並計算'he'跟'is'兩個字詞相似度:
```python
def cos_similarity(vector1, vector2, eps=1e-8):
  dx = x / (np.sqrt(np.sum(x**2)) + eps)
  dy = y / (np.sqrt(np.sum(y**2)) + eps)

  return np.dot(dx, dy)
```

```bash
>>> he_vec = co_matrix[token_to_id['he']]
>>> is_vec = co_matrix[token_to_id['is']]

>>> cos_similarity(he_vec, is_vec)

0.0
```
相較 one-hot encoding，此方法考量了 word 間的 neighbors。
<br> 但他仍有許多問題：
1. Matrix size increase with vocabulary -> 隨著字詞增加，要不斷修改矩陣(矩陣不斷增大)。
2. High Dimension -> 一個字詞就要作為一個 vector，其維度與 one-hot encoding 一樣很高。
3. Sparsity -> 矩陣中有很多 element 為 0，容易受許多雜訊影響，故以此為架構建構出之結果較不 robust。
