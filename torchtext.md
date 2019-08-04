在建構 Deep Learning 模型時，預處理(Preprocess the data)經常需要花上許多時間完成，而在自然語言處理中，文字的處理更是相當困難且複雜。
```bash
在看網路的NLP相關Tutorial時，意外地發現一個蠻好用的library - 'torchtext'，
它能為使用Pytorch做NLP的使用者省去不少預處理的時間。
下面會稍微簡單地介紹他的方便性跟使用方式，並比較如果是直接使用Python內建或numpy、pandas等處理起來的差別。
```
