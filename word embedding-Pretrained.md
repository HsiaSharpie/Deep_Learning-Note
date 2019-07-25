前一篇大略提到 `Word Embedding`的概念，並以Keras、Pytorch的API實作它。
```bash
我們可把Embedding layer想像成一個 dictionary，它能夠將一integer map到低維度的 dense vector，
故Embedding matrix也被稱為 Wordvector lookup table。
```
在前篇是討論第一種 Word Embedding:
<br>Embedding layer是透過給定的維度生成random vector後，並從我們輸入的integer對應到相應的dense vector，
而後續該如何更新Word representation，就是由data、target-test、model去決定。
