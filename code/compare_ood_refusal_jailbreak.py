import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils import PCA_DIM
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def smooth_fn(x0, temp=2):
    x0 = np.minimum(np.maximum(x0, 0.01), 0.99)
    x = np.power(x0, 1 / temp) / (np.power(x0, 1 / temp) + np.power(1 - x0, 1 / temp))
    return x


def calculate_boundary(xlim, ylim, weight, bias):
    if np.abs(weight[0]) > np.abs(weight[1]):
        xlim_by_ylim_0 = (-bias - weight[1] * ylim[0]) / weight[0]
        xlim_by_ylim_1 = (-bias - weight[1] * ylim[1]) / weight[0]
        return [(xlim_by_ylim_0, ylim[0]), (xlim_by_ylim_1, ylim[1])]
    else:
        ylim_by_xlim_0 = (-bias - weight[0] * xlim[0]) / weight[1]
        ylim_by_xlim_1 = (-bias - weight[0] * xlim[1]) / weight[1]
        return [(xlim[0], ylim_by_xlim_0), (xlim[1], ylim_by_xlim_1)]


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_paths", type=str, nargs='+', required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # prepare data
    fname = f'all_soft_refusal'
    dataset_1 = 'advbench'
    fname += f'_{dataset_1}'
    dataset_2 = 'malicious'
    fname += f'_{dataset_2}'
    dataset_harmless = 'testset'
    with open(f"./data/advbench.txt") as f:
        lines_1 = f.readlines()[:100]
    with open(f"data/MaliciousInstruct.txt") as f:
        lines_2 = f.readlines()
    with open(f"./data_harmless/testset.txt") as f:
        lines_harmless = f.readlines()
    os.makedirs(args.output_path, exist_ok=True)

    #colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        'held-out': 'tab:blue',
        'held-out + jailbreak': 'tab:cyan',
        'malicious': 'tab:red',
        'malicious + jailbreak': 'tab:pink',
        'advbench': 'tab:brown',
        'advbench + jailbreak': 'tab:orange',
    }

    all_queries_1 = [e.strip() for e in lines_1 if e.strip()]
    all_queries_2 = [e.strip() for e in lines_2 if e.strip()]
    n_queries = len(all_queries_1)

    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]
    n_queries_harmless = len(all_queries_harmless)

    ncols = 1
    if len(args.pretrained_model_paths) % ncols != 0:
        raise ValueError(f"len(args.pretrained_model_paths) % ncols != 0")
    nrows = len(args.pretrained_model_paths) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 3.8 * nrows))
    fig2 = plt.figure()

    for mdx, pretrained_model_path in enumerate(args.pretrained_model_paths):
        logging_cuda_memory_usage()
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(pretrained_model_path)

        # prepare model
        model_name = pretrained_model_path.split('/')[-1]
        config = AutoConfig.from_pretrained(pretrained_model_path)
        num_layers = config.num_hidden_layers


        # w/o
        logging.info(f"Running w/o")
        hidden_states = safe_open(f'hidden_states_harmless/{model_name}_{dataset_harmless}.safetensors',
                                  framework='pt', device=0)
        all_hidden_states_harmless = []
        for idx, query in enumerate(all_queries_1):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_harmless.append(tmp_hidden_states)

        hidden_states = safe_open(f'hidden_states/{model_name}_{dataset_1}.safetensors',
                                  framework='pt', device=0)
        all_hidden_states_1 = []
        for idx, query in enumerate(all_queries_1):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_1.append(tmp_hidden_states)

        hidden_states = safe_open(f'hidden_states/{model_name}_{dataset_2}.safetensors',
                                  framework='pt', device=0)
        all_hidden_states_2 = []
        for idx, query in enumerate(all_queries_2):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_2.append(tmp_hidden_states)

        all_hidden_states_1 = torch.stack(all_hidden_states_1)
        all_hidden_states_2 = torch.stack(all_hidden_states_2)
        all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)

        # jailbreak
        logging.info(f"Running jailbreak")
        hidden_states_with_jailbreak = safe_open(f'hidden_states_with_jailbreak_harmless/{model_name}_{dataset_harmless}.safetensors',
                                                        framework='pt', device=0)
        all_hidden_states_with_jailbreak_harmless = []
        for idx, query_harmless in enumerate(all_queries_harmless):
            tmp_hidden_states = hidden_states_with_jailbreak.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_jailbreak_harmless.append(tmp_hidden_states)

        hidden_states_with_jailbreak = safe_open(f'hidden_states_with_jailbreak/{model_name}_{dataset_1}.safetensors',
                                                framework='pt', device=0)
        all_hidden_states_with_jailbreak_1 = []
        for idx, query in enumerate(all_queries_1):
            tmp_hidden_states = hidden_states_with_jailbreak.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_jailbreak_1.append(tmp_hidden_states)

        hidden_states_with_jailbreak = safe_open(f'hidden_states_with_jailbreak/{model_name}_{dataset_2}.safetensors',
                                                framework='pt', device=0)
        all_hidden_states_with_jailbreak_2 = []
        for idx, query in enumerate(all_queries_2):
            tmp_hidden_states = hidden_states_with_jailbreak.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_jailbreak_2.append(tmp_hidden_states)

        all_hidden_states_with_jailbreak_1 = torch.stack(all_hidden_states_with_jailbreak_1)
        all_hidden_states_with_jailbreak_2 = torch.stack(all_hidden_states_with_jailbreak_2)
        all_hidden_states_with_jailbreak_harmless = torch.stack(all_hidden_states_with_jailbreak_harmless)


        scores_1 = get_following_indices(
            model_name, dataset_1, config=args.config, use_harmless=False, return_only_scores=True)
        scores_2 = get_following_indices(
            model_name, dataset_2, config=args.config, use_harmless=False, return_only_scores=True)
        scores_harmless = get_following_indices(
            model_name, dataset_harmless, config=args.config, use_harmless=True, return_only_scores=True)
        scores_with_jailbreak_1 = get_following_indices(
            model_name, dataset_1, config=args.config, use_jailbreak_prompt=True, use_harmless=False, return_only_scores=True)
        scores_with_jailbreak_2 = get_following_indices(
            model_name, dataset_2, config=args.config, use_jailbreak_prompt=True, use_harmless=False, return_only_scores=True)
        scores_with_jailbreak_harmless = get_following_indices(
            model_name, dataset_harmless, config=args.config, use_jailbreak_prompt=True, use_harmless=True, return_only_scores=True)

        scores_1 = torch.tensor(scores_1, device='cuda', dtype=torch.float)
        scores_2 = torch.tensor(scores_2, device='cuda', dtype=torch.float)
        scores_harmless = torch.tensor(scores_harmless, device='cuda', dtype=torch.float)
        scores_with_jailbreak_1 = torch.tensor(scores_with_jailbreak_1, device='cuda', dtype=torch.float)
        scores_with_jailbreak_2 = torch.tensor(scores_with_jailbreak_2, device='cuda', dtype=torch.float)
        scores_with_jailbreak_harmless = torch.tensor(scores_with_jailbreak_harmless, device='cuda', dtype=torch.float)

        with safe_open(f'./estimations/{model_name}_all/transform.safetensors', framework='pt') as f:
            mean = f.get_tensor('mean').float().to('cuda')
            V = f.get_tensor('V').float().to('cuda')

        hidden_states = torch.cat([
            all_hidden_states_harmless,
            all_hidden_states_with_jailbreak_harmless,
            all_hidden_states_1,
            all_hidden_states_with_jailbreak_1,
            all_hidden_states_2,
            all_hidden_states_with_jailbreak_2,
        ], dim=0).float()

        lower_dim = torch.matmul(hidden_states - mean, V)

        ax = fig.add_subplot(nrows, ncols, mdx + 1)
        ax.set_title(model_name)
        ax.set_aspect(1)

        values = torch.cat([
            scores_harmless,
            scores_with_jailbreak_harmless,
            scores_1,
            scores_with_jailbreak_1,
            scores_2,
            scores_with_jailbreak_2,
        ], dim=0)
        values = values.cpu().numpy()

        values = smooth_fn(values)

        num = lower_dim.shape[0]
        if model_name not in ['openchat-3.5', 'openchat-3.5-1210']:
            ax.scatter(lower_dim[:num//3, 0].cpu().numpy(), lower_dim[:num//3, 1].cpu().numpy(),
                    marker='o', alpha=0.3, c=values[:num//3], cmap='jet_r')
            ax.scatter(lower_dim[num//3:2*num//3, 0].cpu().numpy(), lower_dim[num//3:2*num//3, 1].cpu().numpy(),
                    marker='x', alpha=0.3, c=values[num//3:2*num//3], cmap='jet_r')
            ax.scatter(lower_dim[2*num//3:, 0].cpu().numpy(), lower_dim[2*num//3:, 1].cpu().numpy(),
                    marker='*', alpha=0.3, c=values[2*num//3:], cmap='jet_r')
        else:
            ax.scatter(lower_dim[:num//3, 0].cpu().numpy(), lower_dim[:num//3, 1].cpu().numpy(),
                    marker='o', alpha=0.3, c=1-values[:num//3], cmap='jet')
            ax.scatter(lower_dim[num//3:2*num//3, 0].cpu().numpy(), lower_dim[num//3:2*num//3, 1].cpu().numpy(),
                    marker='x', alpha=0.3, c=1-values[num//3:2*num//3], cmap='jet')
            ax.scatter(lower_dim[2*num//3:, 0].cpu().numpy(), lower_dim[2*num//3:, 1].cpu().numpy(),
                    marker='*', alpha=0.3, c=1-values[2*num//3:], cmap='jet')

        scatter = fig2.add_subplot(nrows, ncols, mdx + 1).scatter([], [], c=[], cmap='jet')
        cmap = scatter.get_cmap()
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if (xlim[1] - xlim[0]) * 0.8 > (ylim[1] - ylim[0]):
            delta = (xlim[1] - xlim[0]) * 0.8 - (ylim[1] - ylim[0])
            ylim = (ylim[0] - delta / 2, ylim[1] + delta / 2)
        else:
            delta = (ylim[1] - ylim[0]) / 0.8 - (xlim[1] - xlim[0])
            xlim = (xlim[0] - delta / 2, xlim[1] + delta / 2)

        with safe_open(f'estimations/{model_name}_all/harmfulness.safetensors', framework='pt') as f:
            weight = torch.mean(f.get_tensor('weight'), dim=0).squeeze(0).tolist()
            bias = torch.mean(f.get_tensor('bias'), dim=0).squeeze(0).tolist()
        boundary_points = calculate_boundary(xlim, ylim, weight, bias)
        logging.info(f"harmfulness boundary: {boundary_points}")

        weight1 = torch.tensor(weight, device='cuda', dtype=torch.float)
        weight1_cut = weight1[:2] / torch.norm(weight1[:2])
        weight1 = weight1 / torch.norm(weight1)

        with safe_open(f'estimations/{model_name}_all/refusal.safetensors', framework='pt') as f:
            weight = torch.mean(f.get_tensor('weight'), dim=0).squeeze(0).tolist()
            bias = torch.mean(f.get_tensor('bias'), dim=0).squeeze(0).tolist()
        boundary_points = calculate_boundary(xlim, ylim, weight, bias)
        logging.info(f"refusal boundary: {boundary_points}")
        ax.plot([boundary_points[0][0], boundary_points[1][0]],
                [boundary_points[0][1], boundary_points[1][1]],
                color='tab:gray', alpha=1, linewidth=3, linestyle='--')

        weight2 = torch.tensor(weight, device='cuda', dtype=torch.float)
        weight2_cut = weight2[:2] / torch.norm(weight2[:2])
        weight2 = weight2 / torch.norm(weight2)

        axins = inset_axes(ax, width="75%", height="3%", loc='upper center',
                           bbox_to_anchor=(0, -0.01, 1, 1),
                           bbox_transform=ax.transAxes)
        cb = plt.colorbar(sm, cax=axins, orientation='horizontal', pad=0.05)
        cb.set_alpha(0.5)

        refusal_direction = - weight2_cut.cpu().numpy()
        middle_point = (
        (boundary_points[0][0] + boundary_points[1][0]) / 2, (boundary_points[0][1] + boundary_points[1][1]) / 2)
        boundary_length = np.sqrt(
            (boundary_points[0][0] - boundary_points[1][0]) ** 2 + (boundary_points[0][1] - boundary_points[1][1]) ** 2)
        direction_length = boundary_length * 0.2
        vector = refusal_direction * direction_length
        head_width = (xlim[1] - xlim[0]) * 0.02
        start_point = (middle_point[0] - vector[0] / 2, middle_point[1] - vector[1] / 2)
        ax.arrow(start_point[0], start_point[1], vector[0], vector[1],
                 color='tab:gray', alpha=1, linewidth=3, head_width=head_width, head_length=head_width)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if model_name in ['Llama-2-7b-chat-hf', 'vicuna-7b-v1.5', 'CodeLlama-7b-Instruct-hf',
                          'Mistral-7B-Instruct-v0.2']:
            ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(f"{args.output_path}/{fname}_{args.config}.pdf")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
