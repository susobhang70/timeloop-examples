
# Tiny Models for Dementia Detection on the Edge
### Authors: Shvetank Prakash, Alex Liu, Sameer Khurana, Bruce Hecht
#### Report Due Date: May 03, 2022, 11:59PM EST


## Workloads


### Preparation

The exact conda environment we used for training dementia detection models are available at [workload_requirement.txt](./workload_requirement.txt). [Pytorch2timeloop converter](https://github.com/Accelergy-Project/pytorch2timeloop-converter) is also required for generating timeloop modules. 


### Training from Scratch and Knowledge Distillation

Code for training the dataset is availabe in [this notebook](model/fhs_training.ipynb). However, FHS dataset is a private dataset (due to patient privacy) and cannot be shared. Alternatively, we provided our pretrained model and codes for evaluation in next section.



### Model Quantization, Evaluation (classification), and Converting to Timeloop

Pretrained models are available [here](https://drive.google.com/drive/folders/19GSJVqJ1m25-_w-bObYgh5xI6_H4DBfb?usp=sharing), please change `ckpt_root` in `model/check_q_perf.py` to the download directory on your machine. To convert pretrained models into a set of timeloop layer configs, run:
```
cd model
python3 <model_name>.py
```
where the available options for `<model_name>` are `cnn_raw`/`cnn_effnet`/`cnn_small_v1`/`cnn_small_v2`/`cnn_small_v3`.


To run the full evaluation:
```
cd model
python3 -W ignore check_q_perf.py > result.txt
```
We also provided the log file for numbers reported in our report at [model/result.txt](model/result.txt).


### Additional Codebase for Keyword Spotting Model

This part is just providing information regarding the keyword spotting model mentioned as a reference in our report. Codebase is taken and modified from [Honk: CNNs for Keyword Spotting](https://github.com/castorini/honk.git). To get the models size and layer configs for timeloop, run
```
cd model/honk/
python3 -m utils.train --type eval
```



## Hardware Support
For detailed instructions, please read [./workspace/final-project/README.md](./workspace/final-project/README.md). 

### Using Docker

Please pull the docker first to update the container, and then start with `docker-compose up`. 
```
cd <your-git-repo-for-final-project>
docker-compose pull
docker-compose up
```
After finishing the project, please commit all changes and push back to this repository.

###  Related reading
 - [Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/)
 - [Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html)
 - [SparseLoop tutorial](https://accelergy.mit.edu/sparse_tutorial.html)
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
 - simple weight stationary architecture: you can refer to the related lecture notes
 - simple output stationary architecture: you can refer to the related lecture notes
