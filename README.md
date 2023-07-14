# Piecewise Deterministic Markov Processes for Bayesian Neural Networks

[arxiv](https://arxiv.org/abs/2302.08724)


## More instructions coming soon!

Sorry for not having more documentation here, its coming (am just swamped finishing my dissertations). If you have any questions in the interim, please email me at ej.goan@qut.edu.au :)

## Scripts to train models

The script [train_conv.py](bin/train_conv.py) handles training for the MAP initialisation, and fitting of the various MCMC methods.
Can the exact scripts used to train all these models in the [run](run) directory.

Just for an example,

``` shell
# fit cifar_10 map
mkdir -p ~/data/ethan/data/pdmp/uai2023_sgd/cifar_10_resnet_map
python train_conv.py resnet20 categorical 1.0 \
       --out_dir <MAP_OUT_DIR> \
       --batch_size 512 \
       --data cifar_10 \
       --map_iters 25000 \
       --opt sgd \
       --lr 0.0005 \
       --momentum 0.9 \
       --schedule cosine\
       --no_log 
```

And then to run Boomerang sampler

``` shell
mkdir -p ~/data/ethan/data/pdmp/uai2023_sgd/cifar_10_resnet_boomerang_interpolation
python train_conv.py resnet20 categorical 0 \
       --out_dir <BOOM_OUT_DIR> \
       --batch_size 512 \
       --data cifar_10 \
       --num_results 200 \
       --steps_between 10 \
       --boomerang \
       --ipp_sampler interpolation \
       --ref 0.1 \
       --std_ref 0.001 \
       --map_path <MAP_OUT_DIR>/map_weights.pkl \
       --no_log 
```

Can then get performance metrics from the [bin/entropy.py](bin/entropy.py) script (is called entropy but this will report other calibration metrics as well).

``` shell
python entropy.py resnet20 cifar_10 --out_dir <BOOM_OUT_DIR> --batch_size 1024 --reliability
```

Can get effective sample size measurements from the [bin/autocorrelation.py](bin/autocorrelation.py) script.

Again any dramas please reach out via email.
