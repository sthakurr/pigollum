from typing import Any, Dict, Optional
from gollum.data.utils import torch_delete_rows
from gollum.utils.config import instantiate_class
from torch import Tensor
import torch
import warnings

class BotorchOptimizer:
    def __init__(
        self,
        design_space: Optional[Tensor] = None,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        batch_strategy: str = "kriging",
        batch_size: int = 1,
        tkwargs: Optional[Dict[str, Any]] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
    ):

        self.design_space = design_space
        self.surrogate_model_config = (
            surrogate_model_config or BotorchOptimizer.default_surrogate_model_config()
        )
        self.acq_function_config = (
            acq_function_config or BotorchOptimizer.default_acq_function_config()
        )
        self.acquisition_function = None
        self.batch_strategy = batch_strategy
        self.batch_size = batch_size

        self.tkwargs = tkwargs
        print("Using device:", self.tkwargs["device"])


    def lie_to_me(self, candidate, train_y, strategy="kriging"):
        supported_strategies = ["cl_min", "cl_mean", "cl_max", "kriging"]
        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of "
                + str(supported_strategies)
                + ", "
                + "got %s" % strategy
            )

        if strategy == "cl_min":
            y_lie = (
                torch.min(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-min lie
        elif strategy == "cl_mean":
            y_lie = (
                torch.mean(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-mean lie
        elif strategy == "cl_max":
            y_lie = (
                torch.max(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-max lie
        else:
            y_lie, _ = self.surrogate_model.predict(candidate)
        return y_lie

    def train_surrogate_model(self, train_x, train_y):
        with warnings.catch_warnings():
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config,
                train_x=train_x,
                train_y=train_y,
            )
            self.surrogate_model.fit()

    def suggest_next_experiments(
        self,
        train_x,
        train_y,
        design_space,
    ):
        self.train_surrogate_model(train_x, train_y)

        additional_acq_function_params = self.update_acquisition_function_params(
            train_y
        )
        self.acquisition_function = instantiate_class(
            self.acq_function_config, **additional_acq_function_params
        )

        if self.batch_size == 1:
            best_point, best_indices, _ = self.optimize_acquisition_function(design_space)
            return [best_point], [best_indices.item()]
        else:
            candidates, candidate_indices, _ = self.optimize_acquisition_function_batch(
                train_x,
                train_y,
                design_space,
            )
            return candidates, candidate_indices

    def optimize_acquisition_function(
        self,
        design_space,
        chunk_size: int = 512,
    ):
        # Evaluate acquisition in chunks to bound peak GPU memory regardless
        # of design space size (1C fix).
        acq_chunks = []
        with torch.no_grad():
            for start in range(0, design_space.size(0), chunk_size):
                chunk = design_space[start : start + chunk_size].unsqueeze(-2)
                acq_chunks.append(self.acquisition_function(chunk).squeeze(-1).cpu())
        acq_values = torch.cat(acq_chunks, dim=0)
        best_indices = acq_values.topk(1)[1]
        best_point = design_space[best_indices].squeeze(1)
        return best_point, best_indices, acq_values

    def optimize_acquisition_function_batch(self, train_x, train_y, design_space):

        if self.batch_strategy in ["kriging", "cl_min", "cl_mean", "cl_max"]:
            candidates = []
            candidate_indices = []
            candidate_acq_values = []
            # Track original positions as rows are removed from design_space
            original_positions = torch.arange(design_space.size(0))

            for _ in range(self.batch_size):
                best_point, best_idx, acq_values = (
                    self.optimize_acquisition_function(design_space)
                )
                y_lie = self.lie_to_me(
                    best_point, train_y, strategy=self.batch_strategy
                )
                train_x = torch.cat([train_x, best_point])
                train_y = torch.cat([train_y, y_lie])

                original_pos = original_positions[best_idx].item()
                design_space = torch_delete_rows(design_space, best_idx)
                original_positions = torch_delete_rows(original_positions, best_idx)

                candidates.append(best_point)
                candidate_indices.append(original_pos)
                candidate_acq_values.append(acq_values)

        return candidates, candidate_indices, candidate_acq_values

    @staticmethod
    def default_surrogate_model_config():
        # return default surrogate model config
        return {
            "class_path": "gollum.surrogate_models.gp.GP",
            "init_args": {
                "covar_module": {
                    "class_path": "gpytorch.kernels.ScaleKernel",
                    "init_args": {
                        "base_kernel": {
                            "class_path": "gpytorch.kernels.MaternKernel",
                            "init_args": {"nu": 2.5},
                        }
                    },
                },
                "likelihood": {
                    "class_path": "gpytorch.likelihoods.GaussianLikelihood",
                    "init_args": {"noise": 1e-4},
                },
                "normalize": False,
                "initial_noise_val": 1.0e-4,
                "noise_constraint": 1.0e-05,
                "initial_outputscale_val": 1.0,
                "initial_lengthscale_val": 1.0,
            },
        }

    @staticmethod
    def default_acq_function_config():
        # return default acquisition function config
        return {
            "class_path": "botorch.acquisition.UpperConfidenceBound",
            "init_args": {"beta": 2.0, "maximize": True},
        }

    def update_acquisition_function_params(self, train_y):
        params = {"model": self.surrogate_model}
        if "ExpectedImprovement" in self.acq_function_config["class_path"]:
            params["best_f"] = train_y.max().item()
           

        return params
