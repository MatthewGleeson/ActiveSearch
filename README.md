# Active Search

This repo contains python implentations of [Efficient Nonmyopic Active Search, ICML 2017](http://proceedings.mlr.press/v70/jiang17d.html) and [Efficient Nonmyopic Batch Active Search, NeurIPS 2018](https://papers.nips.cc/paper/7387-efficient-nonmyopic-batch-active-search).

Active search is an active learning paradigm which seeks to sequentially inspect data so as to discover members of a rare, desired class. The labels are not known a priori but can be revealed by querying a costly labeling oracle. The goal is to design an policy to sequentially query(observe) points to find as many valuable points as possible under a labeling budget. 
[Efficient Nonmyopic Active Search](http://proceedings.mlr.press/v70/jiang17d.html) 

Below is an example of Efficient Nonmyopic Active Search(ENS) running on a toy 2-d problem with a budget of 300 queries:

![Example of ENS running on a toy 2-d problem](ens.gif)

For a Matlab implementation of ENS written by Roman Garnett and Shali Jiang, look [here](https://github.com/shalijiang/efficient_nonmyopic_active_search). A 3-minute video introducing Efficient Nonmyopic Batch Active search can be found [here](https://www.youtube.com/watch?v=9y1HNY95LzY&feature=youtu.be) 

# Getting Started

This code can run on Unix,MacOS, Windows. Simply clone the repository, install requirements, and run the demo like this:

``` bash
$ git clone https://github.com/MatthewGleeson/ActiveSearch.git
$ pip install -r requirements.txt
$ python3 scriptToRun.py
```

Change parameter settings in the script file to try different datasets and policies!

The code has been tested on MacOS High Sierra 10.13.6 with Python 3.6.0.

