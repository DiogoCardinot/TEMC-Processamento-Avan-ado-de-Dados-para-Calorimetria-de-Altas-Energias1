"""
            MASTERS ANALYSIS

Bernardo S. Peralva    <bernardo@iprj.uerj.br>
Guilherme I. Goncalves <ggoncalves@iprj.uerj.br>

Copyright (C) 2019 Bernardo & Guilherme

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from scipy.optimize import linprog
from .base import FilterBase


class SparseCOF(FilterBase):
    """ SparseCOF filter """

    def __init__(self, k=.5, e=.12, cof_threshold=3.0):
        super(SparseCOF, self).__init__(7)
        self.coeff_k = k
        self.coeff_e = e
        self.coeff_t = cof_threshold
        self.in_time_pulse_index = 6

        # pileup pulse shape
        # pylint: disable=bad-whitespace,line-too-long
        self.mat_h = np.array([
            [0, 0.,     0.,     0.,     0.,     0.,     0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424],  # nopep8
            [0, 0.,     0.,     0.,     0.,     0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.    ],  # nopep8
            [0, 0.,     0.,     0.,     0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.,     0.    ],  # nopep8
            [0, 0.,     0.,     0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.,     0.,     0.    ],  # nopep8
            [0, 0.,     0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.,     0.,     0.,     0.    ],  # nopep8
            [0, 0.,     0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.,     0.,     0.,     0.,     0.    ],  # nopep8
            [0, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424, 0.,     0.,     0.,     0.,     0.,     0.    ]  # nopep8
        ])
        # pylint: enable=bad-whitespace,line-too-long

        self.n_cols = self.mat_h.shape[1]

    def apply(self, pulse):
        if pulse.size != self.filter_size:
            raise "Incompatible input size"

        sparse_solution = self.perform_sparse(pulse)
        if sparse_solution["success"]:
            vec_sparse_x = sparse_solution["x"][0:self.n_cols]
            vec_ene = self.perform_cof(pulse, vec_sparse_x)
            return max(vec_ene)
        return 0

    def perform_sparse(self, pulse):
        """
        identify bunchs of interest using SPARSE
        """
        mat_aa = np.dot(self.mat_h.T, self.mat_h)

        mat_a = np.block([
            [mat_aa, -mat_aa],
            [-mat_aa, mat_aa]
        ])

        mat_b = np.block([
            np.dot(self.mat_h.T, pulse) + self.coeff_e * np.ones(self.n_cols),
            np.dot(-self.mat_h.T, pulse) + self.coeff_e * np.ones(self.n_cols)
        ])

        vec_f = np.block([
            np.ones(self.n_cols),
            self.coeff_k * np.ones(self.n_cols)
        ])

        return linprog(vec_f, A_ub=mat_a, b_ub=mat_b)

    def perform_cof(self, pulse, vec_sparse_x):
        """
        estimate the energy of bunchs of interest using COF
        """
        vec_selected_samples = np.where(vec_sparse_x > self.coeff_t)[0]
        if vec_selected_samples.size == 0:
            vec_selected_samples = [self.in_time_pulse_index]
        elif self.in_time_pulse_index not in vec_selected_samples:
            vec_selected_samples = np.append(
                vec_selected_samples, self.in_time_pulse_index)

        vec_h = self.mat_h[:, vec_selected_samples]
        mat_aux_a = np.dot(vec_h.T, vec_h)
        mat_aux_a_inv = np.linalg.inv(mat_aux_a)
        mat_aux_b = np.dot(vec_h, mat_aux_a_inv)
        output = np.dot(pulse.T, mat_aux_b)

        return output
