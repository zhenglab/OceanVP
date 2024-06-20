# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

dataset_parameters = {
    'ocean_t0_32_64': {  # 0m depth, water_temp
        'in_shape': [16, 1, 32, 64],
        'pre_seq_length': 16,
        'aft_seq_length': 16,
        'total_length': 32,
        'data_name': 'ocean_t0',
        'train_time': ['1994', '2013'], 'val_time': ['2014', '2014'], 'test_time': ['2015', '2015'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'ocean_s0_32_64': {  # 0m depth, salinity
        'in_shape': [16, 1, 32, 64],
        'pre_seq_length': 16,
        'aft_seq_length': 16,
        'total_length': 32,
        'data_name': 'ocean_s0',
        'train_time': ['1994', '2013'], 'val_time': ['2014', '2014'], 'test_time': ['2015', '2015'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'ocean_uv0_32_64': {  # 0m depth, water_u and water_v
        'in_shape': [16, 2, 32, 64],
        'pre_seq_length': 16,
        'aft_seq_length': 16,
        'total_length': 32,
        'data_name': 'ocean_uv0',
        'train_time': ['1994', '2013'], 'val_time': ['2014', '2014'], 'test_time': ['2015', '2015'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
}