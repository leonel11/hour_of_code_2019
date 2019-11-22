# CommentsСlassifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone https://github.com/smorzhov/hour_of_code_2019.git`

## Running

1. Download pretrained [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) model (2.03 GB). Unzip it into `src/data` directory.
2. If you plan to use nvidia-docker, you need to build nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_keras_tf_py3:gpu .
    ```
    Run container (run this command in the same directory where Dockerfile is)
    ```bash
    nvidia-docker run --user $(id -u):$(id -g) -dt --name sm_hoc -m 50GB -v $(pwd)/src:/$(basename $(pwd)) -w /$(basename $(pwd)) sm_keras_tf_py3:gpu /bin/bash
    ```
3. Cleaning dataset
    ```bash
    nvidia-docker exec --env CUDA_VISIBLE_DEVICES='0' sm_hoc python3 -u nlp.py prepare-data
    ```
4. Training

    By default, only the 0th GPU is visible for the docker container. You can change this by passing `--env` option to `exec`. For example:
    ```bash
    nvidia-docker exec --env CUDA_VISIBLE_DEVICES='0' sm_hoc python3 -u nlp.py train --data-path ./processed_data 
    ```
    This will start training on 0th, 1st and 2nd GPUs.

## Advices

You can add some custom stop words. They must be placed in `~src/data/stopwords.txt` file (one word per line).