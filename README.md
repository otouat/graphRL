# graphRL
Comparative study between GraphRNN, GRAN and GraphOpt

This is using the official PyTorch implementation of [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760) as described in the following NeurIPS 2019 paper:

```
@inproceedings{liao2019gran,
  title={Efficient Graph Generation with Graph Recurrent Attention Networks}, 
  author={Liao, Renjie and Li, Yujia and Song, Yang and Wang, Shenlong and Nash, Charlie and Hamilton, William L. and Duvenaud, David and Urtasun, Raquel and Zemel, Richard}, 
  booktitle={NeurIPS},
  year={2019}
}
```

## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is one of {```gran_grid```, ```gran_community```, ```graphrnn_mlp_community```, ```graphrnn_rnn_community```}:

  ```python run_exp.py -c config/X.yaml```

The training file are stored in the ```exp/GRAN``` and ```exp/GraphRNN``` directory

### Test

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/X.yaml -t```