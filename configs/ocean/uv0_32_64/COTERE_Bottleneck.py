method = 'COTERE'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'bottleneck'
hid_S = 32
hid_T = 64
N_T = 8
N_S = 2
# training
lr = 5e-3
batch_size = 16
drop_path = 0.2
sched = 'cosine'
warmup_epoch = 0
# COTERE
cotere_type = 'CTSR'
cotere_c_mlp_r = 0.25
cotere_t_mlp_r = 32
cotere_s_conv_size = 3
cotere_embed_pos = 'A'
block_type = 'i3d'
middle_ratio = 4
