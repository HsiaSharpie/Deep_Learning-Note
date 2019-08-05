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
雖然平常在實作時，我們還會需要個validationset，但只要在額外從trainset或testset切分出來即可！

在'splits method'中我們要丟入的arguments即為上述在'Field','LabelField'所初始化的實例。
如此我們就能夠簡單的透過自訂的tokenize方法，將大量的文字進行簡單的預處理。

切分完後我們可使用examples attribute檢查裡面的值，他是以dictionary包裝而成。
```

```bash
假設我們想看train_data中的第一個sentences:

>>> train_data.examples[0]
{'text': ['elvira', 'mistress', 'of', 'the', 'dark', 'is', 'one', 'of', 'my', 'fav', 'movies', ',', 'it', 'has', 'every', 'thing', 'you', 'would', 'want', 'in', 'a', 'film', ',', 'like', 'great', 'one', 'liners', ',', 'sexy', 'star', 'and', 'a', 'Outrageous', 'story', '!', 'if', 'you', 'have', 'not', 'seen', 'it', ',', 'you', 'are', 'missing', 'out', 'on', 'one', 'of', 'the', 'greatest', 'films', 'made', '.', 'i', 'ca', "n't", 'wait', 'till', 'her', 'new', 'movie', 'comes', 'out', '!'], 'label': 'pos'}

以上就會產生一個dictionary，而裡面分別有text與label兩個key。
text對應之value: TEXT -> tokenize後的結果，以陣列形式呈現。
label對應之value: LABEL -> 'pos' or 'neg'。
```

3. 其餘幾個重要arguments:
```bash
* fix_length:
就如同argument的字面意義，我們打算將後續切割的各個batch設定為多長的固定長度(length)？
這個參數其實相當重要，我們在餵batch給model時希望各個batch的大小是相同的。
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
第二部分: Vocabulary

```bash
在Corpus-based representation的紀錄有提及，我們要想要從Corpus中進行learning！
最基本且常見的作法就是: One-hot encoding。

我們也有記錄到雖然One-hot encoding是個不錯的字詞轉向量方法，但他有些缺點:
1. 維度過高 -> dimension = len(vocab)。
2. 各個vector彼此間為相互獨立。

後續我們會以Embedding的方式，將字詞轉換至相對低維度空間稠密進行表示。
除此之外，我們可將tokenize後的vocab'取部分納入訓練'。
基本上也分為兩種:
1. 出現次數少於幾次之字詞，則刪除。
2. 取maximum vocabulary size。
```

```bash
這邊使用第二種，取maximum vocabulary size的方式:
max_size = 20000

TEXT.build_vocab(train_data, max_size=max_size)
LABEL.build_vocab(train_data)

使用build_vocab method建構完後，我們能用TEXT.vocab看有什麼資訊可以使用:
>>> print(vars(TEXT.vocab).keys())
dict_keys(['freqs', 'itos', 'unk_index', 'stoi', 'vectors'])

相同地他是以dictionary包裝而成:
1. freqs -> 以collections中之Counter計數字詞出現頻率
2. itos -> integer to string, type:list
3. stoi -> string to integer, type:defaultdict
```

```python
>>> print(TEXT.vocab.freqs.most_common(20))
[('the', 202789), (',', 192769), ('.', 165632), ('and', 109469), ('a', 109242), ('of', 100791), ('to', 93641), ('is', 76253), ('in', 61374), ('I', 54030), ('it', 53487), ('that', 49111), ('"', 44657), ("'s", 43331), ('this', 42385), ('-', 36979), ('/><br', 35822), ('was', 35035), ('as', 30388), ('with', 29940)]


>>> print(TEXT.vocab.itos[:10])
['<unk>', '<pad>', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is']
```
------------------------------------------------------------------------------
第三部分: Batch
```bash
接著，是將完成Preprocess後之train,validation,test三個set以batch_size進行劃分。
在Pytorch中通常是以DataLoader完成此任務，
但torchtext的BucketIterator又讓事情更簡單了。
```

```bash
他是一個非常強大的'Iterator'，在arguments給定後，他能夠在每個迭代回傳相同長度的batch，
並在每次回傳時極小化必須要padding的數目。

Note:
劃分出來之Iterator，text將轉換成'TEXT.vocab.stoi'中之index，且以tensor(LongTensor)表示。
而label也將轉換成'LABEL.vocab.stoi'之0, 1index，也是以tensor表示，但其為FloatTensor。

而之所以label是轉換為FloatTensor在於他後續在模型計算loss、更新參數時能夠進行計算。
```
```python
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    device = device)
```
------------------------------------------------------------------------------
Torchtext這個很優的library就先暫時紀錄至此，我會再紀錄一篇如何搭配上Pytorch架構的模型，
<br>若後續在Torchtext使用上有更新更不錯的用途我會再繼續紀錄的！
