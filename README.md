# Canonical Correlation Forests (CCFs)

[CCFs](https://arxiv.org/abs/1507.05444) are a decision tree ensemble method for classification and regression. CCFs naturally
accommodate multiple outputs, provide a similar computational complexity to random forests,
and inherit their impressive robustness to the choice of input parameters.

This implementation is completely done using Numpy and SciPy, which are open-sourced
numerical computing libraries.

## Prerequisites
1. Numpy == 1.17.3
2. SciPy == 1.3.1
3. Matplotlib == 3.1.2


### Code
For classification example run the following command:
```bash
python3 classification_example.py
```

For regression example run the following command:
```bash
python3 regression_example.py
```

#### Original Code

```
https://github.com/twgr/ccfs
```

#### Paper citation
```
@article{rainforth2015canonical,
  title={Canonical correlation forests},
  author={Rainforth, Tom and Wood, Frank},
  journal={arXiv preprint arXiv:1507.05444},
  year={2015}
}
```
