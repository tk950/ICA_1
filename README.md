# 簡単な説明です
# 独立成分分析によるカクテルパーティー効果の実装
入力を`speechA1.wav`とする。これは女性と男性の声が混ざった音声データである。  
これを独立成分分析によって分解することで2人の声に分ける。`A2.wav`は実際に女性の声のデータを取り出したものである。  
pythonにはICAを行うライブラリが用意されているが、今回は自分で実装した。アルゴリズムについては、下記の文献を参考にした。  

### 参考文献
Hyvärinen, Aapo, Juha Karhunen, and Erkki Oja. Independent component analysis. Vol. 46. John Wiley
& Sons, 2004

