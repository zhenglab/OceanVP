#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code adapted from:
# https://github.com/facebookresearch/SlowFast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

class GetWeightAndActivation:
    """
    A class used to get weights and activations from specified layers from a Pytorch model.
    """

    def __init__(self, model, layers):
        """
        Args:
            model (nn.Module): the model containing layers to obtain weights and activations from.
            layers (list of strings): a list of layer names to obtain weights and activations from.
                Names are hierarchical, separated by /. For example, If a layer follow a path
                "s1" ---> "pathway0_stem" ---> "conv", the layer path is "s1/pathway0_stem/conv".
        """
        self.model = model
        self.hooks = {}
        self.layers_names = layers
        # eval mode
        self.model.eval()
        self._register_hooks()

    def _get_layer(self, layer_name):
        """
        Return a layer (nn.Module Object) given a hierarchical layer name, separated by /.
        Args:
            layer_name (str): the name of the layer.
        """
        layer_ls = layer_name.split("/")
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        return prev_module

    def _register_single_hook(self, layer_name):
        """
        Register hook to a layer, given layer_name, to obtain activations.
        Args:
            layer_name (str): name of the layer.
        """

        def hook_fn(module, input, output):
            self.hooks[layer_name] = output.clone().detach()

        layer = get_layer(self.model, layer_name)
        layer.register_forward_hook(hook_fn)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.layers_names`.
        """
        for layer_name in self.layers_names:
            self._register_single_hook(layer_name)

    def get_activations(self, input):
        """
        Obtain all activations from layers that we register hooks for.
        Args:
            input (tensors, list of tensors): the model input.
            bboxes (Optional): Bouding boxes data that might be required
                by the model.
        Returns:
            activation_dict (Python dictionary): a dictionary of the pair
                {layer_name: list of activations}, where activations are outputs returned
                by the layer.
        """
        input_clone = input.clone()
        preds = self.model(input_clone)

        activation_dict = {}
        for layer_name, hook in self.hooks.items():
            # list of activations for each instance.
            activation_dict[layer_name] = hook

        return activation_dict, preds

    def get_weights(self):
        """
        Returns weights from registered layers.
        Returns:
            weights (Python dictionary): a dictionary of the pair
            {layer_name: weight}, where weight is the weight tensor.
        """
        weights = {}
        for layer in self.layers_names:
            cur_layer = get_layer(self.model, layer)
            if hasattr(cur_layer, "weight"):
                weights[layer] = cur_layer.weight.clone().detach()
            else:
                print(
                    "Layer {} does not have weight attribute.".format(layer)
                )
        return weights


def get_indexing(string):
    """
    Parse numpy-like fancy indexing from a string.
    Args:
        string (str): string represent the indices to take
            a subset of from array. Indices for each dimension
            are separated by `,`; indices for different dimensions
            are separated by `;`.
            e.g.: For a numpy array `arr` of shape (3,3,3), the string "1,2;1,2"
            means taking the sub-array `arr[[1,2], [1,2]]
    Returns:
        final_indexing (tuple): the parsed indexing.
    """
    index_ls = string.strip().split(";")
    final_indexing = []
    for index in index_ls:
        index_single_dim = index.split(",")
        index_single_dim = [int(i) for i in index_single_dim]
        final_indexing.append(index_single_dim)

    return tuple(final_indexing)

def process_layer_index_data(layer_ls, layer_name_prefix=""):
    """
    Extract layer names and numpy-like fancy indexing from a string.
    Args:
        layer_ls (list of strs): list of strings containing data about layer names
            and their indexing. For each string, layer name and indexing is separated by whitespaces.
            e.g.: [layer1 1,2;2, layer2, layer3 150;3,4]
        layer_name_prefix (Optional[str]): prefix to be added to each layer name.
    Returns:
        layer_name (list of strings): a list of layer names.
        indexing_dict (Python dict): a dictionary of the pair
            {one_layer_name: indexing_for_that_layer}
    """

    layer_name, indexing_dict = [], {}
    for layer in layer_ls:
        ls = layer.split()
        name = layer_name_prefix + ls[0]
        layer_name.append(name)
        if len(ls) == 2:
            indexing_dict[name] = get_indexing(ls[1])
        else:
            indexing_dict[name] = ()
    return layer_name, indexing_dict

def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split("/")
    prev_module = model
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module

def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor

def MSE(pred, true, spatial_norm=False):
    
    if not spatial_norm:
        return torch.mean((pred-true)**2, axis=[0, 1]).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return torch.mean((pred-true)**2 / norm, axis=[0, 1])

class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
        self, model, target_layers, data_mean, data_std, colormap="viridis"
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """

        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            self.activations[layer_name] = output.clone().detach()

        target_layer = get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        assert len(inputs) == len(
            self.target_layers
        ), "Must register the same number of target layers as the number of input pathways."
        input_clone = [inp.clone() for inp in inputs]
        preds = self.model(input_clone[0])

        score = MSE(preds, labels[0], spatial_norm=True)

        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()
        localization_maps = []
        for i, inp in enumerate(inputs):
            _, _, T, H, W = inp.permute(0,2,1,3,4).size()

            gradients = self.gradients[self.target_layers[i]]
            activations = self.activations[self.target_layers[i]]
            B, C, Tg, _, _ = gradients.size()

            weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

            weights = weights.view(B, C, Tg, 1, 1)
            localization_map = torch.sum(
                gradients * activations, dim=1, keepdim=True
            )
            # localization_map = torch.sum(
            #     F.relu(gradients) * activations, dim=1, keepdim=True
            # )
            localization_map = F.relu(localization_map)
            localization_map = F.interpolate(
                localization_map,
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            )
            localization_map_min, localization_map_max = (
                torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
                torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
            )
            localization_map_min = torch.reshape(
                localization_map_min, shape=(B, 1, 1, 1, 1)
            )
            localization_map_max = torch.reshape(
                localization_map_max, shape=(B, 1, 1, 1, 1)
            )
            # Normalize the localization map.
            localization_map = (localization_map - localization_map_min) / (
                localization_map_max - localization_map_min + 1e-12
            )
            localization_map = localization_map.data

            localization_maps.append(localization_map)

        return localization_maps, preds

    def __call__(self, inputs, labels=None, alpha=0.5):
        localization_maps, _ = self._calculate_localization_map(
            inputs, labels=labels
        )
        return localization_maps
