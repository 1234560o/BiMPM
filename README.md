# BiMPM
Re-implementation of BIMPM(Bilateral Multi-Perspective Matching for Natural Language Sentences)



### Training

* On [Snli](https://nlp.stanford.edu/projects/snli/) data: `bash script/run_SNLI.py` 
* On Chinese sentence Similarity task like BQ_corpus: `bash run_BQ.py`. BQ_corpus is not public data set, you can apply from here: http://icrc.hitsz.edu.cn/info/1037/1162.html
* On other data: Modify the code from `data_loader.py` and other related code. 



### Evaluation

Accuracy on Snli data set（%）：

| data set | Accuracy（%） |
| :------: | :-----------: |
| dev set  |     85.74     |
| test set |     84.91     |

Accuracy on the article is **86.0**(test set).



### Prediction

Upload soon!

