from itertools import chain
import sympy as sp
from lbmpy.equilibrium import GenericDiscreteEquilibrium, ContinuousHydrodynamicMaxwellian
from lbmpy.maxwellian_equilibrium import get_weights
from pystencils.sympyextensions import simplify_by_equality
from pystencils import Assignment, AssignmentCollection

class DiscreteThermalMaxwellianCHT(GenericDiscreteEquilibrium):

    def __init__(self, stencil, compressible=True,
                 order=2,
                 rho_Cp_T=sp.Symbol("rho_Cp_T"),
                 u=sp.symbols("u_:3"),
                 c_s_sq=sp.Symbol("c_s") ** 2,
                 substitutions=None,
                 temperature = sp.Symbol("T"),
                 Cp_ref = None):
        dim = stencil.D

        if order is None:
            order = 2

        self._order = order
        self._compressible = compressible
        self._rho_Cp_T = rho_Cp_T
        self._u = u[:dim]
        self._c_s_sq = c_s_sq
        self._substitutions = substitutions
        self._temperature = temperature
        self.Cp_ref = Cp_ref


        pdfs = discrete_thermal_equilibrium_cht(stencil, rho_Cp_T=self._rho_Cp_T, u=u,
                                                order=order, c_s_sq=c_s_sq,
                                                compressible=compressible,substitutions = self._substitutions,temperature= self._temperature, Cp_ref = self.Cp_ref)

        zeroth_order_moment = rho_Cp_T
        super().__init__(stencil, pdfs, zeroth_order_moment, u)

    #print("self.temp fieldi s " , self._temperature_field)

    @property
    def background_distribution(self):
        """Returns the discrete Maxwellian background distribution, which amounts exactly to the
    #    lattice weights."""
        return DiscreteThermalMaxwellianCHT(self._stencil, compressible=True,
                                            order=self._order, rho_Cp_T=sp.Integer(1),
                                            u=(0,) * self.dim, c_s_sq=self._c_s_sq,temperature=self._temperature,Cp_ref = self.Cp_ref)


def discrete_thermal_equilibrium_cht(stencil, rho_Cp_T=sp.Symbol("rho_Cp_T"), u=sp.symbols("u_:3"), order=2,
                                     c_s_sq=sp.Symbol("c_s") ** 2, compressible=True,substitutions=None,temperature=sp.Symbol("T"),Cp_ref=None):
    """
    Returns the common discrete LBM equilibrium as a list of sympy expressions

    Args:
        stencil: tuple of directions
        rho_Cp_T: sympy symbol for the internal energy
        u: symbols for macroscopic velocity, only the first 'dim' entries are used
        order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
        c_s_sq: square of speed of sound
        compressible: compressibility
    """
    weights = get_weights(stencil, c_s_sq)
    assert stencil.Q == len(weights)
    u = u[:stencil.D]
    res = [thermal_equilibrium_cht(e_q, u, rho_Cp_T, w_q, order, c_s_sq, compressible,substitutions,temperature, Cp_ref) for w_q, e_q in zip(weights, stencil)]
    return tuple(res)

def thermal_equilibrium_cht(v=sp.symbols("v_:3"), u=sp.symbols("u_:3"), rho_Cp_T=sp.Symbol("rho_Cp_T"), weight=sp.Symbol("w"),
                            order=2, c_s_sq=sp.Symbol("c_s") ** 2, compressible=True,substitutions=None,temperature=sp.Symbol("T"), Cp_ref=None):
    """
    Returns the common discrete LBM equilibrium depending on the mesoscopic velocity and the directional lattice weight

    Args:
        v: symbols for mesoscopic velocity
        u: symbols for macroscopic velocity
        rho_Cp_T: sympy symbol for the internal energy
        weight: symbol for stencil weights
        order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
        c_s_sq: square of speed of sound
        compressible: compressibility
    """

    ## implementation based on the total enthalpy LB method
    Cp = sp.Symbol("Cp")
    u_times_u = 0
    for u_alpha in u:
        u_times_u += u_alpha * u_alpha

    e_times_u = 0
    for c_q_alpha, u_alpha in zip(v, u):
        e_times_u += c_q_alpha * u_alpha

    if v == (0,0) or v == (0,0,0):
        result = rho_Cp_T - Cp_ref* 1 #+ weight*Cp*temperature* (Cp_ref/Cp - u_times_u/2*c_s_sq )

    else:
        fq = Cp_ref/Cp
        fq += e_times_u/c_s_sq + u_times_u/2*c_s_sq**2 - u_times_u/2*c_s_sq
        result = weight*Cp*temperature

    return result
