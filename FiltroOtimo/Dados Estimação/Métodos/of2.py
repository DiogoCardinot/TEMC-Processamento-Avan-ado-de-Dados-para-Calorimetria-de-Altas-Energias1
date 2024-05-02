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

from textwrap import dedent
import numpy as np
from .base import FilterBase


class OF2(FilterBase):
    """ OF2 filter """

    def __init__(self, noise_dataset=None):
        super().__init__(7)
        self.noise_dataset = noise_dataset
        self.project_filter_weights()

    def apply(self, pulse):
        if self.weights.size != pulse.size:
            raise "Incompatible input size"

        energy = 0.0
        for i in range(self.filter_size):
            energy = energy + (self.weights[i] * pulse[i])
        return energy

    def project_filter_weights(self):
        """
        calculates the OF2 weights using unitary covariance matrix
        """
        # pulse shape parameters
        vec_g = np.array(
            [0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424])
        vec_dg = np.array(
            [0.00004019, 0.00333578, 0.03108120, 0.00000000, -0.02434490, -0.00800683, -0.00243344])

        # unitary covariance matrix
        mat_c = self.__get_noise_covariance_matrix()

        # defines solution vector
        vec_b = np.zeros(self.filter_size + 3)
        vec_b[self.filter_size] = 1.0

        # defines matrix A, where Aw=b
        mat_a = np.zeros((self.filter_size + 3, self.filter_size + 3))

        for i in range(self.filter_size):
            # copies C to A
            for j in range(self.filter_size):
                mat_a[i][j] = mat_c[i][j]

            # copies g to A
            mat_a[self.filter_size][i] = vec_g[i]
            mat_a[i][self.filter_size] = -vec_g[i]

            # copies dg to A
            mat_a[self.filter_size + 1][i] = vec_dg[i]
            mat_a[i][self.filter_size + 1] = -vec_dg[i]

            # sets the unitary column
            mat_a[self.filter_size + 2][i] = 1.0
            mat_a[i][self.filter_size + 2] = -1.0

        vec_w = np.linalg.solve(mat_a, vec_b)

        self.weights = np.zeros(self.filter_size)
        for i in range(self.filter_size):
            self.weights[i] = vec_w[i]

    def __get_noise_covariance_matrix(self):
        if self.noise_dataset is not None:
            return np.cov(self.noise_dataset, rowvar=False)
        return np.identity(self.filter_size)

    def __str__(self):
        return dedent(f"""\
          OF2 Filter:
            size = {self.filter_size}
            weights = {", ".join("%.5f" % w for w in self.weights)}\
        """)
