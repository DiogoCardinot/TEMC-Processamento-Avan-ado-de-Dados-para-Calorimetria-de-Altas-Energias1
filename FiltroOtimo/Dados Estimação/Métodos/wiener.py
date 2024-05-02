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


class Wiener(FilterBase):
    """ Wiener filter """

    def __init__(self, train_dataset, pulse_generator, using_bias=True):
        super().__init__(train_dataset.shape[1])
        self.dataset = train_dataset
        self.pulse_generator = pulse_generator
        self.using_bias = using_bias
        self.project_filter_weights()

    def apply(self, pulse):
        if self.weights.size < pulse.size:
            raise "Incompatible input size"

        energy = 0.0
        for i in range(self.filter_size):
            energy = energy + (self.weights[i] * pulse[i])

        if self.using_bias:
            energy = energy + self.weights[-1]

        return energy

    def project_filter_weights(self):
        """
        calculates the Wiener weights
        """
        n_samples = self.dataset.shape[0]
        n_cols = self.filter_size + 1 if self.using_bias else self.filter_size

        vec_d = np.zeros(n_samples)
        mat_x = np.zeros((n_samples, n_cols))

        # generate the matrix X and the vector d
        # by summing known pulses to each noise row
        for i, noise in enumerate(self.dataset):

            # generate known pulse
            analog_pulse = self.pulse_generator.generate_pulse()
            digital_samples = analog_pulse.get_digital_samples()
            amplitude = analog_pulse.amplitude

            # sum the noise to the knows pulse
            for j in range(self.filter_size):
                mat_x[i][j] = noise[j] + digital_samples[j]

            # additional element
            if self.using_bias:
                mat_x[i][self.filter_size] = 1

            # desired amplitude
            vec_d[i] = amplitude

        # cross-correlation of X
        mat_r = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                summ = 0
                for k in range(n_samples):
                    summ = summ + (mat_x[k][i] * mat_x[k][j])
                mat_r[i][j] = summ / n_samples

        # correlation between X and D
        vec_p = np.zeros(n_cols)
        for i in range(n_cols):
            summ = 0
            for k in range(n_samples):
                summ = summ + (mat_x[k][i] * vec_d[k])
            vec_p[i] = summ / n_samples

        self.weights = np.linalg.solve(mat_r, vec_p)

    def __str__(self):
        return dedent(f"""\
          Wiener Filter:
            size = {self.filter_size}
            weights = {", ".join("%.5f" % w for w in self.weights)}\
        """)
