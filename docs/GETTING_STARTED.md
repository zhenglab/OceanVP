# Getting Started

All the experiments are implemented on a single GPU. Below are the examples for training, testing and visualizing the water temperature models. The other models (water velocity or salinity, cotere or baseline) can be implemented via setting the corresponding config files. The detailed description for the arguments can be found in [parser.py](../lib/utils/parser.py).

#### Training

```shell
python tools/train.py -d ocean_t0_32_64 -c configs/ocean/t0_32_64/COTERE_Bottleneck.py --ex_name ocean_t0_cotere_bottleneck --temp_stride 2 --epoch 50 --fps
```

#### Testing

```shell
python tools/test.py -d ocean_t0_32_64 -c configs/ocean/t0_32_64/COTERE_Bottleneck.py --ex_name ocean_t0_cotere_bottleneck --temp_stride 2
```

#### Visualizing

```shell
python tools/vis.py -d ocean_t0_32_64 -c configs/ocean/t0_32_64/COTERE_Bottleneck.py --ex_name ocean_t0_cotere_bottleneck --temp_stride 2
```
