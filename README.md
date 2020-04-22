# Learning in the Machine: To Share or not to Share?

This repository is associated with [our paper](https://www.sciencedirect.com/science/article/pii/S0893608020300939).

```
@article{ott2020learning,
  title={Learning in the machine: To share or not to share?},
  author={Ott, Jordan and Linstead, Erik and LaHaye, Nicholas and Baldi, Pierre},
  journal={Neural Networks},
  year={2020},
  publisher={Elsevier}
}
```

# Pipeline
1. [Hyperparameter grid search](https://github.com/jordanott/WeightSharing/tree/master/HyperOpt)
  * Number of layers
  * Learning rate
2. [Full experiments](https://github.com/jordanott/WeightSharing/blob/master/runner.py)
  * Types of augmentation
    * Edge noise
    * Translation
    * Rotation
    * Noise
    * Quadrant swap
3. [Results](https://github.com/jordanott/WeightSharing/blob/master/Results/Results.ipynb)
  * Metrics
    * Loss
    * Validation loss
    * Training accuracy
    * Validation accuracy
    * Edge noise augmented validation accuracy
    * Translation augmented validation accuracy
    * Noise augmented validation accuracy
    * Rotation augmented validation accuracy
  * Approximate weight sharing
    * Observing the distance between free convolutional filters within a layer
  * Variable Connection Patterns
    * Varying the probability
