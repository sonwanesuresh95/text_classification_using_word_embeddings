# text_classification_using_word_embeddings
A Deep Learning Text Classification model using Word Embeddings.
## Info
This is a deep learning based spam-ham classifier which is trained using keras Embedding layer.<br>
Model architecture is as follows:

Model: "sequential_0"

|Layer (type)      |           Output Shape      |        Param #   |
| :---        |    :----:   |          ---: |
|embedding_0 (Embedding) |     (None, 100, 50)     |      431900    |
|flatten_0 (Flatten)    |      (None, 5000)       |       0         |
|dense_0 (Dense)        |     (None, 2)        |         10002     |

Total params: 441,902
Trainable params: 441,902
Non-trainable params: 0


## Usage
To install requirements, do<br>
<code>
  $pip install requirements.txt
</code><br><br>
To train your own text classification model, do<br>
<code>
  $python train.py
</code><br><br>
To start webapp, do<br>
<code>
  $python app.py
</code><br><br>

## In Action
![image](https://github.com/sonwanesuresh95/text_classification_using_word_embeddings/blob/master/Text%20Classification%20-%20Google%20Chrome%2026-09-2020%2001_31_21.png)<br>
![image](https://github.com/sonwanesuresh95/text_classification_using_word_embeddings/blob/master/Text%20Classification%20-%20Google%20Chrome%2026-09-2020%2001_42_26.png)
