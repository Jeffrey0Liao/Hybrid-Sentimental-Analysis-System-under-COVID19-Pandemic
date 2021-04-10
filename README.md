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

### Environmental setup
To set up the evnironment, run `pip install -r requirements.txt`. If you get a permission denied, append `--user` flag and try again.

### Backbones
Several third-party tools are applied to achieve funtionalities. However, some of them are not necessary.

* The text pipeline leverages `Stanford Parser` as a **prerequisite**, download it from [its offical page](https://nlp.stanford.edu/software/lex-parser.shtml#Download) and set up path variables.
* The image pipeline uses `VGG-19 feature extractor`, it will **automatically download** the pretrained weight for you if not detected. 

## Train the model

### Datasets
Datasets in this project is derived from [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) and [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k/activity). We manipulated original datasets for the joint model. There are three datasets you can use to train the model:

* Download [SST+Generated Dataset]()
* Download [SST+Scraped Dataset]()
* Download [TreeFlickr8k Dataset]()

Once you have download the dataset, put it into `data` folder.

To train the model yourself, you can run the `experiment.ipynb` script, hyperparameters are free for adjustment in the script. There are also a visualization system and a check point management system encrypted for evaluating the model. 

States of experiment will be saved once per epoch, the model state dict will be saved in the `experiment_results` folder along with data logs and plots. Some preliminary training results are shown as:

![](https://github.com/Jeffrey0Liao/2020-2021-Final-Year-Project-Joint-Sentimental-Analysis-Based-on-Tree-topology/blob/main/resource/data.png)

## Usage
* If you want to use pretrained model for sentimental classification, download [HS_model_dict.pth]() and put it into the `bin` folder.
* If you want to use pretrained word to vector model as embeddings, download [word2vec.wv]() and put it into the `bin`folder.
* If you want to use a broad scale dictionary to build general corpus, download [text8.txt]() and put it into the `doc` folder. (Note that we recommend you to use text8 corpus for a correct embedding. To some uncommon words that are not included in this corpus, we replaced them with special `<unk>` words for genreality.)

The project has predefined strcuture, make sure you have the same directory hierarchy as:



To run the script, enter the project directory and run `main.py`, with command line arguments: `-s <sentence> -i <image>`, whereas `<sentence>` is the path of your textual input and `<image>` is the path of your image folder. 

For multiple sentences, put them in a single `txt` file one in a line with the format: `image_name.jpg [Tab] Your input sentence here.` Your images should be named accordingly in your image folder.

An examplary commandline would be:
`python main.py -s C/users/noob/project/input.txt -i C/users/noob/project/images/`

When the program finishes, results will be stored in `result.txt` in `./result/`. The output should match inputs with lines, with a format of `Sentimental Prediction, Confidence: Percentile%`
