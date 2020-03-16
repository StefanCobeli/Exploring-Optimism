# Exploring-Optimism
Implementation of the ideas proposed in the paper 
[Exploring Optimism and Pessimism in Twitter Using Deep Learning](https://www.aclweb.org/anthology/D18-1067/),
(C. Caragea et al. 2018)


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
