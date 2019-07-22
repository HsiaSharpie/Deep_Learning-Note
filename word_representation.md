在日常生活中，中文、英文等我們平常使用的話語稱為「自然語言」(Natural Language)。自然語言處理(Natural Language Processing; NLP)主要在探討如何處理及運用自然語言。
```bash
自然語言有其柔軟度，對於相同意義的文字、句子，我們有各種不同的方式能夠表達它。且隨著時代不同，自然語言會不斷改變、演進！
```
照片、影像有固定的大小，我們可以用矩陣或向量有效地表示。但文字有長有短，故要用向量去表示它就有許多需要考量的點，且將文字轉換成向量後，我們期待此向量不但能告訴我們它代表什麼字，也告訴我們其意義。

```bash
Goal of Meaning Representation:
我們要用一個好的向量表示法表示每個字，它可以呈現字與字之間的關係。
```
而 Meaning Representation 主要可以分為：
1. Knowledge-based representation
2. Corpus-based representation

------------------------------------------------------------------------------
(1)Knowledge-based representation:
```bash
此方法必須大量依賴語言學家，由他們人工定義詞意，建構出字與字之間的關係。
```
類似於字典，語言學家建構出詞庫，並把同義詞或相似詞放入同一個群組，例如：car 與 automobile。在將字詞分群後，語言學家也會定義出字詞間的「上下層」關係，以建構字詞間的關聯性。
<br>motocar 位於 motor vehicle 的下層，則 motocar is a motor vehicle，此關係稱為 Hypernyms(is-a) relationship。

而現在最有名的詞庫是`WordNet`，它是由普林士頓大學從1985年就開始研發的詞庫，在 Python 中可以直接從 NLTK 取得 WordNet 的資源。

但其實使用 Knowledge-based representation 有許多限制：
* Newly-invented words -> 更新不頻繁，缺少許多新字
* Subjective -> 過於主觀
* Annotation effort -> 對字詞進行標注 knowledge resource 相當辛苦，且成本昂貴
* Difficult to compute word similarity -> 僅能從由上下層關係得知相對關係，但難以將之進行量化

與其純粹依賴於語言學家，後來衍伸出 Corpus-Based Representation，我們可直接由資料去 learn 出字詞間的關係。
