# Model Zoo

We provide the efficiency and performance of the REL-STPN benchmark methods (Baseline and CoTeRe). The efficiency metrics contain Params, FLOPs and FPS. The performance metrics contain RMSE and MAE. All the models are trained for 50 epochs, and they can be downloaded via the [Google Drive](https://drive.google.com/file/d/1NVh-nrjo2eU-rr-Vn9j5gRn-ucWU0OEg/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/13WNLXxVHjZHO_9_T9kfzBQ?pwd=ovp2
) links.

## The Efficiency of REL-STPN

| Method | Params | FLOPs | FPS |
| :----: | :----: | :---: | :-: |
| Baseline | 0.12M | 1.25G | 1575 |
| CoTeRe | 1.13M | 1.39G | 1027 |

## The Results of REL-STPN on Water Temperature (T0)

| Method | Step(ts) | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| Baseline | 1 | 0.3063 | 0.1654 | [configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 1 | 0.2624 | 0.1328 | [configs/ocean/t0_32_64/COTERE_Bottleneck.py](../configs/ocean/t0_32_64/COTERE_Bottleneck.py) |
| Baseline | 2 | 0.3777 | 0.2105 | [configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 2 | 0.3255 | 0.1726 | [configs/ocean/t0_32_64/COTERE_Bottleneck.py](../configs/ocean/t0_32_64/COTERE_Bottleneck.py) |
| Baseline | 4 | 0.4773 | 0.2818 | [configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/t0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 4 | 0.3960 | 0.2200 | [configs/ocean/t0_32_64/COTERE_Bottleneck.py](../configs/ocean/t0_32_64/COTERE_Bottleneck.py) |

## The Results of REL-STPN on Water Velocity (UV0)

| Method | Step(ts) | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| Baseline | 1 | 0.1053 | 0.0628 | [configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py) | [`link`](https://drive.google.com/file/d/1ieR_2NpWhYCtz_AUezC80MoLEAyyMAqa/view?usp=sharing) |
| CoTeRe | 1 | 0.0932 | 0.0539 | [configs/ocean/uv0_32_64/COTERE_Bottleneck.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck.py) |
| Baseline | 2 | 0.1226 | 0.0732 | [configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 2 | 0.1150 | 0.0680 | [configs/ocean/uv0_32_64/COTERE_Bottleneck.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck.py) |
| Baseline | 4 | 0.1436 | 0.0857 | [configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 4 | 0.1324 | 0.0788 | [configs/ocean/uv0_32_64/COTERE_Bottleneck.py](../configs/ocean/uv0_32_64/COTERE_Bottleneck.py) |

## The Results of REL-STPN on Water Salinity (S0)

| Method | Step(ts) | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| Baseline | 1 | 0.0862 | 0.0344 | [configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 1 | 0.0867 | 0.0346 | [configs/ocean/s0_32_64/COTERE_Bottleneck.py](../configs/ocean/s0_32_64/COTERE_Bottleneck.py) |
| Baseline | 2 | 0.1105 | 0.0475 | [configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 2 | 0.1053 | 0.0454 | [configs/ocean/s0_32_64/COTERE_Bottleneck.py](../configs/ocean/s0_32_64/COTERE_Bottleneck.py) |
| Baseline | 4 | 0.1389 | 0.0614 | [configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py](../configs/ocean/s0_32_64/COTERE_Bottleneck_NONE.py) |
| CoTeRe | 4 | 0.1244 | 0.0565 | [configs/ocean/s0_32_64/COTERE_Bottleneck.py](../configs/ocean/s0_32_64/COTERE_Bottleneck.py) |
