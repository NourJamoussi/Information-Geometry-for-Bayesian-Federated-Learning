import ivon 
from torch import Tensor
from typing import Tuple
import torch
import numpy as np

class IVON_SAMP(ivon.IVON):
    # A subclass of IVON that rewrites _sample_params 
    # To make the sampling with a given covariance instead of the ess and the hess parameters in the original verison used to compute the covariance 
    def __init__(self, *args, cov, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups: 
            group["cov"] = cov      
    def _sample_params(self) -> Tuple[Tensor, Tensor]:
            noise_samples = []
            param_avgs = []
            offset = 0
            for group in self.param_groups:
                gnumel = group["numel"]
                #noise_sample = (
                #    torch.randn(gnumel, device=self._device, dtype=self._dtype)
                #    / (
                #        group["ess"] * (group["hess"] + group["weight_decay"])
                #    ).sqrt()
                #)
                cov = group["cov"].astype(np.float32)
                noise_sample = (
                    torch.randn(gnumel, device=self._device, dtype=torch.float32)
                    * torch.from_numpy(np.sqrt(cov)).to(self._device)
                )

                noise_samples.append(noise_sample)

                goffset = 0
                for p in group["params"]:
                    if p is None:
                        continue

                    p_avg = p.data.flatten()
                    numel = p.numel()
                    p_noise = noise_sample[offset : offset + numel]

                    param_avgs.append(p_avg)
                    p.data = (p_avg + p_noise).view(p.shape)
                    goffset += numel
                    offset += numel
                assert goffset == group["numel"]  # sanity check
            assert offset == self._numel  # sanity check

            return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

