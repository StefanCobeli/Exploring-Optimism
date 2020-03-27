# Exploring-Optimism
Implementation of the ideas proposed in
[Exploring Optimism and Pessimism in Twitter Using Deep Learning](https://www.aclweb.org/anthology/D18-1067/),
(C. Caragea et al. 2018)

## Description

Analysis of the Optimism/Pessimism present in a tweet using _Deep Learning_ methods.

The study focuses on the _Twitter Optimism_ (__OPT__) dataset proposed in
[Finding Optimists and Pessimists on Twitter](https://www.aclweb.org/anthology/P16-2052/)
(Ruan et al. 2016).
In this project we follow the methodology proposed in
[Exploring Optimism and Pessimism in Twitter Using Deep Learning](https://www.aclweb.org/anthology/D18-1067/),
(C. Caragea et al. 2018).

## Basic usage

Download pre-trained embeddings and data:

```
  mkdir optimism && cd optimism
  wget http://web.eecs.umich.edu/~mihalcea/downloads/optimism-twitter-data.zip data/
  unzip data/optimism-twitter-data.zip data/
  wget http://nlp.stanford.edu/data/glove.twitter.27B.zip embeddings/
  unzip glove.twitter.27B.zip embeddings/
```

Clone project & prepare environment:

```
  git clone https://github.com/StefanCobeli/Exploring-Optimism.git
  conda env create -f environment.yml
  conda activate nlp
```

Run minimal project (train a Deep Learning model on the __OPT__ dataset):

```
  python main.py -c config/initial_configs/OPT_small_RAM
```

## Performed work

 - Trained models on __OPT__ on Tweet Level on both 0 & 1/-1 settings (_Section 3.1_);
 - Used BiLSTM, GRUStack & CNN as encoders.
 - Used multiple types of static word embedding:
  - [GloVe](https://nlp.stanford.edu/projects/glove/)
 (Common Crawl 840B & Twitter 27B);
  - [FastText](https://fasttext.cc/docs/en/english-vectors.html).
 - Trained on __TSA__ and tested on __OPT__ (_Section 3.2_);
 - First try on finetunig __TSA__ trained model on __OPT__.  

## Observations

  - Results (accuracy on __OPT__) are similar when using any type of encoding & embedding;
  - Large training batch size (>2048) seem to stuck training performance in local minima;     
  - After 1 training epoch on __TSA__, accuracy on __OPT__ is ~ 0.74;
  - Freezing pre-trained embedding weights provide more stable accuracy on validation set (prevent overfitting on sentiment);
  - Need to investigate how to perform knowledge transfer between TSA and OPT.
  Training only the final MLPs of the architecture (freezing the others, embedding & encoding) seems to work better.   

<!-- ### Part 1:

Analyze the __OPT__ dataset using Deep Learning models.

### Part 2:

Analyze the difference between sentiment and optimism. -->

## Results

Training Deep Learning architecture on [Twitter Sentiment Analysis (TSA)](http://thinknook.com/Twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
dataset and validating results on [Twitter Optimism (__OPT__)](https://lit.eecs.umich.edu/downloads.html#Twitter%20Optimism%20Dataset) dataset.

![](/plots/TSA_train_OPT_test.png)
