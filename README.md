# RomniStereo

The Pytorch code for our following paper

> **RomniStereo: Recurrent Omnidirectional Stereo Matching**, [RA-L 2024 (pdf)](https://arxiv.org/pdf/2401.04345.pdf)
>
> [Hualie Jiang](https://hualie.github.io/), Rui Xu, Minglang Tan and Wenjie Jiang \[[**Insta360**](https://www.insta360.com/)\]


<p align="center">
<img src='/assets/pipeline.jpg' width=1000>
</p>


## Preparation

#### Installation

Create the environment

```bash
conda create -n romnistereo python=3.8
```

Install pytorch

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch  # for cuda 10
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch  # for cuda 11
```

Then install other requirements

```bash
pip install -r requirements.txt
```

#### Download Datasets 

Please download the datasets from [dataset link](https://rvlab.snu.ac.kr/research/omnistereo) and use *download.sh* for processing. 


## Training 

#### Train on OmniThings

```
python train.py --name romnistereoC --dbname omnithings --base_channel C --mixed_precision --total_epochs 30 
# C can be 4, 8, 32, 64
```

#### Finetune on Omnihouse and Sunny

```
python train.py --name romnistereoC_ft --dbname omnihouse sunny --base_channel C --mixed_precision --total_epochs 16 --pretrain_ckpt checkpoints/romnistereoC/romnistereoC_e29.pth
# C can be 4, 8, 32, 64
```


## Evaluation  

The pretrained models of our paper is available on [Google Drive](https://drive.google.com/drive/folders/1KcC5QByDlSKxD174JtValobcrY-E6WiE?usp=sharing). 


```
python eval.py --dbname omnithings/omnihouse/sunny/cloudy/sunet --restore_ckpt models/romnistereoC[_ft].pth --save_result 
# C can be 4, 8, 32, 64
```


## Test on real samples  


```
python test.py --dbname itbt_sample/real_indoor_sample --restore_ckpt models/romnistereoC[_ft].pth --vis --save_result  
# C can be 4, 8, 32, 64
```



## Acknowledgements

The project borrows codes from [OmniMVS](https://github.com/hyu-cvlab/omnimvs-pytorch) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). Many thanks to their authors. 

## Citation

Please cite our paper if you find our work useful in your research.

```
@inproceedings{jiang2024romnistereo,
  title={RomniStereo: Recurrent Omnidirectional Stereo Matching},
  author={Jiang, Hualie and Xu, Rui and Tan, Minglang and Jiang, Wenjie},
  booktitle={IEEE Robotics and Automation Letters},
  year={2024}
}
```
