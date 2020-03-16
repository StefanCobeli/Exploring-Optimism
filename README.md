# Exploring-Optimism
Implementation of the ideas proposed in the paper 
[Exploring Optimism and Pessimism in Twitter Using Deep Learning](https://www.aclweb.org/anthology/D18-1067/),
(C. Caragea et al. 2018)

## Description

Analysis of the Optimism/Pessimism present in a tweet using _Deep Learning_ methods.
We mainly use the __OPT__ dataset proposed in 
[Finding Optimists and Pessimists on Twitter](https://www.aclweb.org/anthology/P16-2052/)
(Ruan et al. 2016).
We follow closely the methodolgy proposed in 
[Exploring Optimism and Pessimism in Twitter Using Deep Learning](https://www.aclweb.org/anthology/D18-1067/),
(C. Caragea et al. 2018).

### Part 1:

Analyze the __OPT__ dataset using Deep Learning models.

### Part 2:

Analyze the difference between sentiment and optimism.

## Results

To be added.

## Usage

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
  python main.py -c config/OPT_small_RAM
```
