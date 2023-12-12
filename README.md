# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Run Training

`python3 train.py data/flowers --gpu --save_dir checkpoint.pth --learning_rate 0.001 --hidden_units 4096 --epochs 10`

## Run prediction 

`python3 predict.py data/flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5`

