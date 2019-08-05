在建構 Deep Learning 模型時，預處理(Preprocess the data)經常需要花上許多時間完成，而在自然語言處理中，文字的處理更是相當困難且複雜。
```bash
在看NLP相關Tutorial時，意外地發現一個蠻好用的library - 'torchtext'，
它能為使用Pytorch做NLP的使用者省去不少預處理的時間也提供一些dataset供我們實作。

下面會稍微簡單地介紹他的方便性跟使用方式，分別拆成三部分:
1. tokenize
2. vocabulary
3. Batch
```
------------------------------------------------------------------------------
第一部分:Tokenize

1. Field、LabelField:
<br>Torchtext的基本概念是建構在`Field`、`LabelField`這兩個class中。透過`Field`、`LabelField`的建構，我們能清楚定義出該如何處理文字，並透過實例化去完成。
```python
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)
```

```bash
上面是一個非常簡單使用'Field class'去建構實例的方式。
在Field中，包含許多arguments可進行設定，在此例僅對tokenize進行設定。
tokenize即為分詞(斷詞)，即為如何將將一堆string劃分成'tokens'的方式。
若沒有額外設定，torchtext之default是根據空格進行切分。
這邊的使用的'spacy'，其實也是在python中對'NLP'提供很多好用功能的library，在此能與'torchtext'完美搭配，
除了tokenize之外，spacy提供字詞中許多與語言學、ML相關的資訊。
```

2. 使用定義之Field對資料集之文字進行tokenize:

```python
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
```

```bash
這邊直接使用torchtext提供的IMDB dataset，我們可以直接從'torchtext'中導入，並直接用其'splits method'將他劃分成訓練及測試集。
雖然平常在實作時，我們還會需要個validationset，但只要在額外從testset切分出來即可！

在'splits method'中我們要丟入的arguments即為上述在'Field','LabelField'所初始化的實例。
如此我們就能夠簡單的透過自訂的tokenize方法，將大量的文字進行簡單的預處理。
```

3. 其餘幾個重要arguments:
```bash
* fix_length:
就如同argument的字面意義，我們打算將後續切割的各個batch設定為多長的固定長度(length)？
這個參數其實相當重要，我們在使用batch更新參數時希望各個batch的大小是相同的。
Ex:
TEXT = data.Field(tokenize = 'spacy', fix_length=20)
則此時，若seq_length小於20就補齊，反之則將超過20的部分刪除。


* pad_token:
此argument可搭配著前面的fix_length一起使用，在seq_length小於fix_length時，
我們需要把seq_length補成相同長度。
By default，它會將缺失的部分以'<pad>'補上，但經常我們也會用0補上(0-padding)。


* include_lengths:
此argument為一boolean。
若設定為True時，當我們使用batch.text印出結果，他會是個tuple。
第一個element: padding過後的句子。
第二個element: 原實際句子的長度。

* 其實還有超多很方便的arguments，而且參數的設定相當直覺:
ex:
sequential: 是否屬於序列文字。
lower: 是否轉換為皆是小寫。
unk_token: 未知字的表達，預設為'<unk>'。
stop_words: 是否除去常用字，ex:the, is, am....
tokenizer_language: 斷詞的語言，預設即為英文('en')。
```
------------------------------------------------------------------------------
