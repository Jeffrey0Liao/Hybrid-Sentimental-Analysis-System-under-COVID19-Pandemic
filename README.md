# 2020-2021-Final-Year-Project-Joint-Sentimental-Analysis-Based-on-Tree-topology
This is the official repository for the project: `Joint Sentimental Analysis Based on Tree topology` as a commitment to Chen Liao's final dissertation in University of Nottingham Ningbo China (UNNC). 

## Background
This project leverages information from both visual and textual modalities to train a sentimental classifier with Tree-Long-Short-Term-Memory netwrok, VGG-19 feature extractor, and attention mechanism as backbones. It takes raw text and image pairs as inputs, and deliver sentiment categories with its confidence (in percentage). Raw text processing pipeline and raw image processing pipeline are also implemented to help data conversion. The output lies in one of the five categories ranging from 0 to 4:<br>

* `Positive` (labeled with 0)
* `Somehow positive` (labeled with 1)
* `Neutral` (labeled with 2)
* `Somehow Negative` (labeled with 3)
* `Negative` (labeled with 4)

The model workflow could be referred to as:<br><br>
![](https://github.com/Jeffrey0Liao/2020-2021-Final-Year-Project-Joint-Sentimental-Analysis-Based-on-Tree-topology/blob/main/resource/f8.png)
<br>

## Setup
This code has been tested on Windows10, Python 3.8.5, Pytorch 1.8.0+cuda 11.1, Numpy 1.17.3, Nvidia RTX2080super, Intel core i7 7700k
