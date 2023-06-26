from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.models.layers.spectral_layers import fourier_derivatives
from modulus.node import Node

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.utils.io.plotter import GridValidatorPlotter
from modulus.utils.io.vtk import grid_to_vtk

from utilities import download_FNO_dataset, load_FNO_dataset
from operations import dx, ddx

from Solver_PINO import Solver_PINO
from GridValidator_PINO import GridValidator_PINO

class Darcy(torch.nn.Module):

    def __init__(self, gradient_method: str = "fdm"):
        super().__init__()
        self.gradient_method = gradient_method

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        u = input_var["sol"]
        k = input_var["coeff"]

        dkdx = input_var["Kcoeff_y"]
        dkdy = input_var["Kcoeff_x"]

        dxf = 1.0 / u.shape[-2]
        dyf = 1.0 / u.shape[-1]

        if self.gradient_method == "exact":
            dudx_exact = input_var["sol__x"]
            dudy_exact = input_var["sol__y"]

            dduddx_exact = input_var["sol__x__x"]
            dduddy_exact = input_var["sol__y__y"] 

            darcy = (
                1
                + (dkdx * dudx_exact)  
                + (k * dduddx_exact)
                + (dkdy * dudy_exact)
                + (k * dduddy_exact)
            )

        elif self.gradient_method == "fdm":
            dudx_fdm = dx(u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
            dudy_fdm = dx(u, dx=dyf, channel=0, dim=1, order=1, padding="replication")

            dduddx_fdm = ddx(u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
            dduddy_fdm = ddx(u, dx=dyf, channel=0, dim=1, order=1, padding="replication")

            darcy = (
                1
                + (dkdx * dudx_fdm)  
                + (k * dduddx_fdm)
                + (dkdy * dudy_fdm)
                + (k * dduddy_fdm)
            )

        elif self.gradient_method == "fourier":
            dim_u_x = u.shape[2]
            dim_u_y = u.shape[3]

            u = F.pad(
                u, (0, dim_u_y-1, 0, dim_u_x - 1), mode="reflect"
            )

            f_du, f_ddu = fourier_derivatives(u, [2.0, 2.0])
            dudx_fourier = f_du[:, 0:1, :dim_u_x, :dim_u_y]
            dudy_fourier = f_du[:, 1:2, :dim_u_x, :dim_u_y]
            dduddx_fourier = f_ddu[:, 0:1, :dim_u_x, :dim_u_y]
            dduddy_fourier = f_ddu[:, 1:2, :dim_u_x, :dim_u_y]


            darcy = (
                1.0
                + (dkdx * dudx_fourier)
                + (k * dduddx_fourier)
                + (dkdy * dudy_fourier)
                + (k * dduddy_fourier)
            )
        else:
            raise ValueError(f"Derivative method {self.gradient_method} not supported")
        
        #pad outer boundary with zeros
        darcy = F.pad(darcy[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        output_var = {"darcy": dxf * darcy}
        return output_var


@modulus.main(config_path="config", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    #run_gm(cfg, "fourier")
    #run_gm(cfg, "fdm")
    #run_gm(cfg, "exact")

#def run_gm(cfg, gradient_method):
    # data
    input_keys = [
        Key("coeff", scale=(7.48360e00, 4.49996e00)),
        Key("Kcoeff_x"),
        Key("Kcoeff_y"),
    ]
    output_keys = [
        Key("sol", scale=(5.74634e-03, 3.88433e-03)),
    ]

    download_FNO_dataset("Darcy_241", outdir="datasets/")
    invar_train, outvar_train = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntest,
    )

    # add additional constraining values for darcy variable
    outvar_train["darcy"] = np.zeros_like(outvar_train["sol"])

    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)

    # model
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=[input_keys[0]],
        decoder_net=decoder_net,
    )
    #if cfg.custom.gradient_method == "exact":
    if cfg.custom.gradient_method == "exact":
        derivatives = [
            Key("sol", derivatives=[Key("x")]),
            Key("sol", derivatives=[Key("y")]),
            Key("sol", derivatives=[Key("x"), Key("x")]),
            Key("sol", derivatives=[Key("y"), Key("y")]),
        ]
        fno.add_pino_gradients(
            derivatives=derivatives,
            domain_length=[1.0, 1.0],
        )

    # node
    inputs = [
        "sol",
        "coeff",
        "Kcoeff_x",
        "Kcoeff_y",
    ]
    #if cfg.custom.gradient_method == "exact":
    if cfg.custom.gradient_method == "exact":
        inputs += [
            "sol__x",
            "sol__y",
        ]
    darcy_node = Node(
        inputs=inputs,
        outputs=["darcy"],
        #evaluate=Darcy(gradient_method=cfg.custom.gradient_method),
        evaluate=Darcy(gradient_method=cfg.custom.gradient_method),

        name="Darcy Node",
    )
    nodes = [fno.make_node('fno'), darcy_node]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")
    # [constraint]

    # add validator
    val = GridValidator_PINO(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver_PINO(cfg, domain)

    # start solver
    slv.solve()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device count:', torch.cuda.device_count())
    print('Using device:', device)
    print(torch.cuda.get_device_name(0))

    run()



