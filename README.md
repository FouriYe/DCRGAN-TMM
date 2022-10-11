# DCRGAN-TMM

This is a [PyTorch](https://pytorch.org/) implementation for the paper ["Disentangling Semantic-to-visual Confusion for Zero-shot Learning"](https://arxiv.org/abs/2106.08605) in IEEE Transactions on Multimedia 2021.

# Prepare dataset
Download the [data](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/) and uncompress it to the folder '../data'.

## Reproduce results
Run demo.sh for CUB, AWA1, APY, SUN dataset
```shell
bash demo.sh
```

## Citation

If you think this code is useful in your research or wish to refer to the baseline results published in our paper, please use the following BibTeX entry.

```
@ARTICLE{ye2021disentangling,
  author={Ye, Zihan and Hu, Fuyuan and Lyu, Fan and Li, Linyan and Huang, Kaizhu},
  journal={IEEE Transactions on Multimedia}, 
  title={Disentangling Semantic-to-Visual Confusion for Zero-Shot Learning}, 
  year={2022},
  volume={24},
  number={},
  pages={2828-2840},
  doi={10.1109/TMM.2021.3089017}}
```
