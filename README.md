# iLPC

This repo covers the implementation of the following paper: 

**"Iterative label cleaning for transductive and semi-supervised few-shot learning (ICCV2021)"** [Pre-print](https://arxiv.org/abs/2012.07962),
<p align='center'>
  <img src='idea.png' width="800px">
</p>

## Abstract

Few-shot learning amounts to learning representationsand acquiring knowledge such that novel tasks may be solvedwith both supervision and data being limited.   Improvedperformance is possible by transductive inference, where theentire test set is available concurrently, and semi-supervisedlearning, where more unlabeled data is available.  Theseproblems are closely related because there is little or noadaptation of the representation in novel tasks.Focusing on these two settings, we introduce a new al-gorithm  that  leverages  the  manifold  structure  of  the  la-beled  and  unlabeled  data  distribution  to  predict  pseudo-labels,  while  balancing  over  classes  and  using  the  lossvalue distribution of a limited-capacity classifier to selectthe cleanest labels, iterately improving the quality of pseudo-labels. Our solution sets new state of the art results on fourbenchmark datasets, namelyminiImageNet,tieredImageNet,CUB and CIFAR-FS, while being robust over feature spacepre-processing  and  the  quantity  of  available  data.


If you find this repo useful for your research, please consider citing the paper
```
@conference{iLPC,
   title = {Iterative label cleaning for transductive and semi-supervised few-shot learning},
   author = {M. Lazarou and Y. Avrithis and T. Stathaki},
   booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
   address = {Virtual},
   year = {2021}
}
```
## Datasets
For all of our experiments regarding WRN-28-10 backbone, we used the pre-trained features provided by [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot) and can be downloaded from here [checkpoints](https://drive.google.com/drive/folders/1KfPzwMvVzybvp13IQW5ipHvSxBncTA-C).

For resnet12 experiments we used the datasets as provided by [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot) for tieredImagenet, CIFAR-FS and CUB experiments. For miniImagenet where we used the .pkl files provided from here [miniImagenet_pkl](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view)

## Running

Exemplar commands for running the code can be found in `scripts/run.sh`.

For unuspervised learning methods `CMC` and `MoCo`, please refer to the [CMC](http://github.com/HobbitLong/CMC) repo.

## Contacts
For any questions, please contact:

Michalis Lazarou (ml6414@ic.ac.uk)  
Yannis Avrithis (yannis@avrithis.net)  
Tania Stathaki (t.stathaki@imperial.ac.uk)


## Acknowlegements
[PT-MAP](https://github.com/yhu01/PT-MAP)

[LR+ICI](https://github.com/Yikai-Wang/ICI-FSL)

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

[MCT](https://github.com/seongmin-kye/MCT)

[S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)




