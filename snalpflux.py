from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps
import sys, os
import numpy as np
from .calc_alp_signal import ALPSNSignal
import json
from gammaALPs import Source, ALP, ModuleList


class SNALPflux(object):
    """
    Class for calculating ALP and gamma-ray flux from a supernova
    using the 'nester' sampler
    """
    def __init__(self, walkerfile,Mprog = 10.,
            cosmo = FlatLambdaCDM(H0=0.7 * 100. * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)):
        """
        Init the class

        Parameters
        ----------
        walkerfile: str
            path to MOSFIT results file which contains the walkers

        {options}

        Mprog: float
            mass of progenitor in solar masses, currently only 10. and 18. solar masses 
            are implemented
        
        cosmo: `astropy.cosmology.FlatLambdaCDM`
            the chosen cosmology. Default: 737
        """
        with open(walkerfile, 'r') as f:
            data = json.loads(f.read())
        if 'name' not in data:
            data = data[list(data.keys())[0]]

        # SN coordinates
        self._c = SkyCoord(ra = data['ra'][0]['value'], dec = data['dec'][0]['value'], unit = (u.hourangle, u.deg))
        self._z =  np.mean([float(data['redshift'][i]['value']) for i in range(len(data['redshift']))])
        self._src = Source(z = self._z, ra = float(self._c.ra.value), dec = float(self._c.dec.value))
        self._cosmo = cosmo

        # class to compute ALP flux 
        self._Mprog = Mprog
        self._alp = ALPSNSignal(Mprog=Mprog)

        # constant to calculate flux:
        dl = self._cosmo.luminosity_distance(self._z)
        self.fluxconstant = 1. / 4. / np.pi / dl.to('cm').value ** 2.

    @property
    def src(self):
        return self._s

    @property
    def Mprog(self):
        return self._Mprog

    @Mprog.setter
    def Mprog(self, Mprog):
        self._Mprog = Mprog
        self._alp = ALPSNSignal(Mprog=Mprog)

    @property
    def cosmo(self):
        return self._cosmo

    @property
    def lumidist(self):
        return self._cosmo.luminosity_distance(self._z)
 

    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = cosmo
        dl = self._cosmo.luminosity_distance(self._z)
        self.fluxconstant = 1. / 4. / np.pi / dl.to('cm').value ** 2.

    def ALPflux(self, EMeV, t_sec, g11):
        """
        Compute the ALP flux at an energy and time t after explosion

        Parameters
        ----------
        EMeV: array-like 
            n-dim array with energies in MeV

        t_sec: array-like 
            m-dim array with time in sec after core collapse

        g11: float
            ALP coupling in 10^-11 GeV-1 

        Returns
        -------
        nxm dim array with ALP flux in units of MeV-1 s-1
        """
        na_dedt = self._alp(EMeV=EMeV, ts = t_sec, g10 = g11 * 10.) # alp spectrum per energy and time
        return na_dedt * 1.e52

    def AvgALPflux(self, EMeV, t_sec, g11):
        """
        Compute the ALP flux averaged over some time span

        Parameters
        ----------
        EMeV: array-like 
            n-dim array with energies in MeV

        t_sec: float  
            time range in seconds over which 
            the spectrum should be averaged.

        g11: float
            ALP coupling in 10^-11 GeV-1 

        Returns
        -------
        Array with averaged ALP flux in units of MeV-1 s-1
        """
        t_sec_array = np.arange(0.,t_sec,0.1)
        na_dedt = self.ALPflux(EMeV, t_sec_array, g11)
        na_dedt_avg = simps(na_dedt, t_sec_array, axis = 1) / t_sec
        return na_dedt_avg

    def Pag(self, EMeV, g11, m_neV, bfield = 'jansson12'):
        """
        Compute the photon-ALP conversion probability

        Parameters
        ----------
        EMeV: array-like 
            n-dim array with energies in MeV

        g11: float
            ALP coupling in 10^-11 GeV-1 

        m_neV: float
            ALP mass in neV

        {options}

        bfield: str
            identifier for Milky Way magnetic field. default: 'jansson12'

        Returns
        -------
        Array with averaged ALP flux in units of MeV-1 s-1
        """

        # calculate conversion probability
        m = ModuleList(ALP(m=m_neV,g=g11), self._src,
            pin = np.diag([0.,0.,1.]), EGeV = EMeV / 1000.)
        m.add_propagation("GMF",0, model = bfield)
        px,py,pa = m.run()
        return px + py

    def integrateGRayFlux(self, emin, emax, t_sec, g11, m_neV, bfield = 'jansson12', esteps = 100, eflux = False):
        """
        Calculate the integrated gamma-ray flux 
        for an ALP flux averaged over some time 

        Parameters
        ----------
        emin: float
            minimum energy in MeV

        emax: float
            maximum energy in MeV

        t_sec: float  
            time range in seconds over which 
            the spectrum should be averaged.

        g11: float
            ALP coupling in 10^-11 GeV-1 

        m_neV: float
            ALP mass in neV

        {options}

        bfield: str
            identifier for Milky Way magnetic field. default: 'jansson12'

        esteps: int
            number of integration steps in energy. default: 100

        eflux: bool 
            if True, calculate energy flux. Default: False

        Returns
        -------
        flux (or energy flux) in units of gamma-rays (MeV) / s / cm2
        """
        EMeV_array = np.logspace(np.log10(emin), np.log10(emax), esteps)

        dna_dedt = self.AvgALPflux(EMeV_array, t_sec, g11) # alps / MeV / s 
        pag = self.Pag(EMeV_array, g11, m_neV, bfield = bfield) # conversion prob
        dng_dedt = dna_dedt * pag # gamma rays / MeV / s 
        flux = dng_dedt * self.fluxconstant # gamma rays / MeV / s / cm^2

        if eflux:
            return simps(flux * EMeV_array * EMeV_array, np.log(EMeV_array))
        else:
            return simps(flux * EMeV_array, np.log(EMeV_array))
