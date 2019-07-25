前一篇大略提到 `Word Embedding`的概念，並以Keras、Pytorch的API實作它。
```bash
我們可把Embedding layer想像成一個 dictionary，它能夠將一integer map到低維度的 dense vector，
故Embedding matrix也被稱為 Wordvector lookup table。
```
在前篇是討論第一種 Word Embedding:
<br>Embedding layer是透過給定的維度生成random vector，並從我們輸入的integer對應到相應的dense vector，
而後續該如何更新Word representation，就是由data、target-test、model等因素去決定。

此篇主要是紀錄如何在Keras、Pytorch中使用 Pretrained Word Embedding。
```bash
Pretrained Word Embedding是由NLP開發團隊事先由Google News、Wiki等資源對許多字預先進行訓練而成，
而這些預訓練資料幾乎是公開免費的。
最著名：Word2Vec、Standford的GLoVe、Facebook的Fasttext...。

在這邊的練習我是使用GLoVe實作，但其實使用Pretrained資料跟Keras、Pytorch的API無關，
都是使用Python內建資料型態及numpy就可完成。
```

In Pytorch:
