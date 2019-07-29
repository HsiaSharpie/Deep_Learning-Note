前一篇大略提到 `Word Embedding`的概念，並以Keras、Pytorch的API實作它。
```bash
我們可把Embedding layer想像成一個 dictionary，它能夠將integer map到低維度的dense vector，
故Embedding matrix也被稱為Wordvector lookup table。
```
在前篇是討論第一種Word Embedding:
<br>Embedding layer是透過給定的維度生成random vector，並從我們輸入的integer對應到相應的dense vector，
而後續該如何更新Word representation，就是由data、target-test、model等因素去決定。

此篇主要是紀錄如何使用Pretrained Word Embedding。
```bash
Pretrained word embedding是由NLP開發團隊事先由Google News、Wiki等資源對許多字預先進行訓練而成，
而這些預訓練資料幾乎是公開免費的。
最著名：Word2Vec、Standford的GLoVe、Facebook的Fasttext...。
```

```bash
在這邊的練習我是使用GLoVe的'glove.6B.100d.txt'實作，但其實使用Pretrained資料跟Keras、Pytorch沒什麼太大關係，
使用Python內建資料型態及numpy預處理後，再把他們餵進去即可。
```

在 GLoVe 提供的資料，大概是長這樣：
<br>dog 0.30817 0.30938 0.52803 -0.92543 -0.73671 0.63475 .....，因為選取的資料是100d，故一個 word 是以100維的向量去表示。

Python Code:
```python
def embeddings_files(embeddings_file):
    word_vectors = []
    word_to_index = {}
    with open(embeddings_file) as file:
        for line in file.readlines():
            values = line.split(' ')
            values[-1] = values[-1].replace('\n', '')
            word = values[0]
            coeff = values[1:]

            word_to_index[word] = len(word_to_index)
            word_vectors.append(coeff)

    return word_to_index, word_vectors

word_to_index, word_vectors = embeddings_files('glove.6B.100d.txt')
```

由以上一個簡短的function，我們就分別取出word和其對應的embedding。
<br>接著，可以架構一個簡單的Class去取用裡面的資訊。
```python
class Pretrained_Embedding(object):
    def __init__(self, word_to_index, word_vectors):
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors

    def get_embeddings(self, word):
      return self.word_vectors[self.word_to_index[word]]

>>> embeddings = Pretrained_Embeddings(word_to_index, word_vectors)
>>> embeddings.get_embedding('dog')

['0.30817',
 '0.30938',
 '0.52803',
 '-0.92543',
 '-0.73671',
 '0.63475',
 '0.44197',
 '0.10262',
 '-0.09142',
 ...
 ...]
```

```bash
在 Word Embedding 中其實有許多非常有趣且特別的資訊:
word vector間之幾何關係，經常隱含著他們的語意關係(semantic relationship)。
```

```bash
(1)若將word vector投影到低維(ex:2維)，相對於大象，因為兩者貓跟老虎皆為貓科動物，投影後的點'可能'更近。
(2)我們也可透過向量的加減運算求出未知向量，最有名的莫過於：king(vector) + female(vector) -> queen(vector)。

而之所以說是'可能'，是因為隨著不同的'target task'所訓練出之 word embedding會有所不同，
想當然，也會有不同之語意關係(semantic relationship)。

後續，再記錄一些使用`Pretrained Word Embedding`的練習。
```
