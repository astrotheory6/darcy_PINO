import numpy as np

import torch

from typing import Dict, List

from modulus.domain.validator import Validator
from modulus.domain.constraint import Constraint
from modulus.utils.io.vtk import grid_to_vtk
from modulus.utils.io import GridValidatorPlotter, DeepONetValidatorPlotter
from modulus.graph import Graph
from modulus.key import Key
from modulus.node import Node
from modulus.constants import TF_SUMMARY
from modulus.distributed import DistributedManager
from modulus.dataset import Dataset, DictGridDataset
from modulus.domain.validator import GridValidator

class GridValidator_PINO(GridValidator):
    def __init__(
        self,
        nodes: List[Node],
        dataset: Dataset,
        batch_size: int = 100,
        plotter: GridValidatorPlotter = None,
        requires_grad: bool = False,
        num_workers: int = 0,
    ):

        # get dataset and dataloader
        self.dataset = dataset
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            distributed=False,
            infinite=False,
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set foward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = plotter
    
    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        relative_losses = GridValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)
        absolute_losses = GridValidator_PINO._l2_absolute_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:

            grid_to_vtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in relative_losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True)
                
        for j, loss in absolute_losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + j, loss, step, new_style=True)

            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + j, loss, step, new_style=True)

        return relative_losses

    @staticmethod
    def _l2_absolute_error(true_var, pred_var):  # TODO replace with metric classes
        new_var = {}
        for key in true_var.keys():
            new_var["l2_absolute_error_" + str(key)] = torch.sqrt(
                torch.mean(torch.square(true_var[key] - pred_var[key]))
            )
        return new_var


    # DELETE THIS LATER
    def _l2_relative_error(true_var, pred_var):
        new_var = {}
        for key in true_var.keys():
            new_var["l2_relative_error_" + str(key)] = torch.sqrt(
                torch.mean(torch.square(true_var[key] - pred_var[key]))
            )
        return new_var
        