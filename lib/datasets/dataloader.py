# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        temp_stride=kwargs.get('temp_stride', 1),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if 'ocean' in dataname:  # 'ocean_t0', 'ocean_s0', 'ocean_uv0', etc.
        from .dataloader_ocean import load_data
        data_split_pool = ['32_64', '64_128']
        data_split = '32_64'
        for k in data_split_pool:
            if dataname.find(k) != -1:
                data_split = k
        return load_data(batch_size, val_batch_size, data_root, num_workers,
                         distributed=dist, data_split=data_split, **kwargs)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
