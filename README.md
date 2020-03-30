# Canonical Correlation Forests (CCFs)

[CCFs](https://arxiv.org/abs/1507.05444) are a decision tree ensemble method for classification and regression. CCFs naturally
accommodate multiple outputs, provide a similar computational complexity to random forests,
and inherit their impressive robustness to the choice of input parameters.

This implementation is completely done using Numpy and SciPy, which are open-sourced
numerical computing libraries.

| ![alt-text-1](./results/spiral_c.png "CCFs results on Spiral Dataset")  | ![alt-text-2](./results/camel_c.png "CCFs results on Camel Dataset") |
|:---:|:---:|
| CCFs results on Spiral Dataset | CCFs results on Camel Dataset |

## Prerequisites
1. Numpy == 1.17.3
2. SciPy == 1.3.1
3. Matplotlib == 3.1.2  # For Visualization

(This code base was developed on Python3.6)

## Code
```bash
pip install -r requirements.txt
```

For classification example run the following command:
```bash
python3 classification_example.py
```

For regression example run the following command:
```bash
python3 regression_example.py
```

### Contribution
Any improvements to the code base are welcomed.


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

![alt-text-1](./logo/plai.jpeg "PLAI-LAB")
