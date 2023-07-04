
import numpy as np
import os
import matplotlib.pyplot as plt

from typing import Dict

from modulus.utils.io.plotter import GridValidatorPlotter 

class GridValidatorPlotter_PINO(GridValidatorPlotter):
    def __init__(self, n_examples):
        super(GridValidatorPlotter_PINO, self).__init__(n_examples=n_examples)
        self.n_examples = n_examples

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        "Try to make plots and write them to tensorboard summary"

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:

                save_dir = results_dir + "/" + tag

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                f.savefig(
                    save_dir + "/" + name + "_" + tag + "_" + str(step) + ".png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")
