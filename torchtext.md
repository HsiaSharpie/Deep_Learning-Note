在建構 Deep Learning 模型時，預處理(Preprocess the data)經常需要花上許多時間完成，而在自然語言處理中，文字的處理更是相當困難且複雜。
```bash
在看NLP相關Tutorial時，意外地發現一個蠻好用的library - 'torchtext'，
它能為使用Pytorch做NLP的使用者省去不少預處理的時間並提供一些dataset供我們實作。

下面會稍微簡單地介紹他的方便性跟使用方式，並比較如果是直接使用Python內建資料型態+numpy、pandas等與之處理起來的差別。
```
------------------------------------------------------------------------------

1. Field、LabelField
Torchtext的基本概念是建構在'Field'這個class中。透過'Field'的建構，我們能清楚定義出該如何處理文字，並透過實例化去完成。
```python
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)
```

上面是一個非常簡單使用`Field class`去建構實例的方式。在Field中，包含許多arguments可進行設定，在此例僅對tokenize進行設定。
<br>tokenize即為分詞(斷詞)，即為如何將將一堆string劃分成`tokens`的方式。若沒有額外設定，torchtext之default是根據空格進行切分。
<br>這邊的使用的`spacy`，其實也是在python中對'`NLP`提供很多好用功能的library。
<br>除了tokenize之外，spacy提供字詞中許多與語言學、ML相關的資訊。
