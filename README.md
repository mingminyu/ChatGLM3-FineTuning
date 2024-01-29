# ChatGLM3-FineTuning

基于 ChatGLM3-6B 的 Finetuning，主要参考官方仓库中的微调方法，支持 LoraTuning 和 P-Tuning。


## 1. 环境安装

如果需要加速，则可以使用 deepspeed 进行训练加速的话，除了安装 deepspeed 之外，还需要安装 mpi4py 这个库。建议使用 conda 进行安装，pip 安装比较麻烦。 

```shell
conda install mpi4py
```
