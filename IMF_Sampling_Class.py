#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad


# In[2]:


from scipy.interpolate import interp1d

class PDFSampling(object):
    """Class for approximations with a given pdf sample."""

    def __init__(self, bin_edges, pdf_array):
        """

        :param bin_edges: bin edges of PDF values
        :param pdf_array: pdf array of given bins (len(bin_edges)-1)
        """
        assert len(bin_edges) == len(pdf_array) + 1
        self._cdf_array, self._cdf_func, self._cdf_inv_func = approx_cdf_1d(
            bin_edges, pdf_array
        )

    def draw(self, n=1):
        """

        :return:
        """
        p = np.random.uniform(0, 1, n)
        return self._cdf_inv_func(p)

    @property
    def draw_one(self):
        """

        :return:
        """
        return self.draw(n=1)


def approx_cdf_1d(bin_edges, pdf_array):
    """

    :param bin_edges: bin edges of PDF values
    :param pdf_array: pdf array of given bins (len(bin_edges)-1)
    :return: cdf, interp1d function of cdf, inverse interpolation function
    """
    assert len(bin_edges) == len(pdf_array) + 1
    norm_pdf = pdf_array / np.sum(pdf_array)
    cdf_array = np.zeros_like(bin_edges)
    cdf_array[0] = 0
    for i in range(0, len(norm_pdf)):
        cdf_array[i + 1] = cdf_array[i] + norm_pdf[i]
    cdf_func = interp1d(bin_edges, cdf_array)
    cdf_inv_func = interp1d(cdf_array, bin_edges)
    return cdf_array, cdf_func, cdf_inv_func


# In[3]:


class IMF:
    
    def __init__(self, func, m_min, m_max):
        self.func = func
        self.m_min = m_min
        self.m_max = m_max
        self.normalized = self.normalize()
        self.expected_mass = self.calculate_expected_mass()

    def normalize(self):
        
        unnormalized_integral = quad(self.func, self.m_min, self.m_max)[0]
        normalization_constant = 1 / unnormalized_integral

        def normalized_function(m):
            return normalization_constant * self.func(m)

        return normalized_function

    def calculate_expected_mass(self):
     
        first_moment = lambda m: m * self.normalized(m)
        return quad(first_moment, self.m_min, self.m_max)[0]
    
    def draw(self, M_tot, oversampling_factor = 2, allowable_percent_error = 10**7):
        
        mass_space = np.linspace(self.m_min, self.m_max, 100000)
        mass_bin_edges = np.linspace(self.m_min, self.m_max, 100001)
        mass_pdf = self.normalized(mass_space)
        mass_sampler = PDFSampling(mass_bin_edges, mass_pdf)
        N_initial = M_tot / self.expected_mass # the bigger M_tot, the closer N_initial * m_exp to M_tot
        safe_N = int(oversampling_factor * N_initial)
        percent_error = np.inf 
        while percent_error > allowable_percent_error: # allow at least one iteration
            mass_draw = mass_sampler.draw(n = safe_N)
            accumulated_mass = np.cumsum(mass_draw)
            first_excess = np.argmax(accumulated_mass > M_tot)
            truncated_mass_draw = mass_draw[:first_excess]
            percent_error = (abs(np.sum(truncated_mass_draw) - M_tot) / M_tot) * 100
        return truncated_mass_draw


# In[4]:


def unnormalized_salpeter(m):
    return m ** -2.35

def unnormalized_kroupa(m):
    # we combine the third and fourth branches of the IMF to one range from 0.5 M_sol to 150 M_sol
    m_min = 0.01
    m_1 = 0.08
    m_2 = 0.5
    m_max = 150
    k_0 = 1
    k_1 = k_0 * m_1 ** (-0.3 + 1.3)
    k_2 = k_1 * m_2 ** (-1.3 + 2.3)
    kroupa_conditions = [(m_min <= m) & (m < m_1), (m_1 <= m) & (m < m_2), (m_2 <= m) &(m <= m_max)]
    kroupa_functions = [lambda m: k_0 * m ** -0.3, lambda m: k_1 * m ** -1.3, lambda m: k_2 * m ** -2.3]
    return np.piecewise(m, kroupa_conditions, kroupa_functions)

def unnormalized_chabrier(m):
    m_min = 0.01
    m_1 = 1
    m_max = 150
    k_1 = 0.158
    k_2 = 0.0443
    chabrier_conditions = [m < m_1, m > m_1]
    chabrier_functions = [lambda m: k_1 * (1 /m) * np.exp((1 / 2) * ((np.log10(m) - np.log10(0.079)) / 0.69) ** 2),
                          lambda m: k_2 * m ** (-2.3)]
    return np.piecewise(m, chabrier_conditions, chabrier_functions)

m_min_salpeter = 0.1
m_max_salpeter = 100
m_min_kroupa = 0.01
m_max_kroupa = 150
m_min_chabrier = 0.01
m_max_chabrier = 150

salpeter = IMF(unnormalized_salpeter, m_min_salpeter, m_max_salpeter)
kroupa = IMF(unnormalized_kroupa, m_min_kroupa, m_max_kroupa)
chabrier = IMF(unnormalized_chabrier, m_min_chabrier, m_max_chabrier)


# In[5]:


# define a lens-source configuration, here z_L = 0.5, z_S = 1.5
# consider an area of 1 mas^2 on the sky

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5)
sigma_crit = lens_cosmo.sigma_crit
proj_mass = lens_cosmo.kappa2proj_mass(kappa = 0.5) # M_sol / Mpc ^ 2
mpc_in_arcsec = lens_cosmo.phys2arcsec_lens(phys = 1) # arcsec value corresponding to 1 Mpc at lens redshift
proj_mass_arcsec = proj_mass / (mpc_in_arcsec) ** 2 # M_sol / arcsec ^ 2
M_tot = proj_mass_arcsec * 0.001 # surface density * area = total mass

