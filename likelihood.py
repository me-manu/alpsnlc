import json
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.stats import gaussian_kde, binom
from scipy.interpolate import UnivariateSpline as USpline, interp1d
from scipy.integrate import simps
from os import path
from .plotutils import get_corner_input
from .snalpflux import SNALPflux
from time import time


class TexpPost(object):
    """
    Class for marginzalized posterior of explosion time from MOSFIT fit 
    using the 'nester' sampler

    TODO: implement smearing with delay
    """
    def __init__(self, walkerfile, min_delay = 0., max_delay = 0.):
        """
        Init the class

        Parameters
        ----------
        walkerfile: str
            path to MOSFIT results file which contains the walkers

        {options}

        min_delay: float
            minimum delay between core collapse and
            onset of optical emission in days, default: 0.

        max_delay: float
            maximum delay between core collapse and
            onset of optical emission in days, default: 0.
        """
        with open(walkerfile, 'r') as f:
            data = json.loads(f.read())
        if 'name' not in data:
            data = data[list(data.keys())[0]]

        model = data['models'][0]
        corner_input, var_names = get_corner_input(model)

        self._texpref = \
             data['models'][0]['realizations'][0]['parameters']['reference_texplosion']['value']

        m = ['exp' in v for v in var_names]
        # get the data of the explosion times
        self._texp = np.array(corner_input)[:,m].flatten()
        # Compute a gaussian kernel density estimate
        self._kde = gaussian_kde(self._texp, bw_method = 'scott')

        self._max_delay = max_delay
        self._min_delay = min_delay
        if self._max_delay > 0.:
            self._cdfinv = None
            self.__convolve_pdf()
        else:
            self._cdf = (np.cumsum(self._texp) - np.cumsum(self._texp)[0]) / \
                        (np.cumsum(self._texp)[-1] - np.cumsum(self._texp)[0]) 

            self._cdfinv = USpline(self._cdf, np.sort(self._texp), k = 1, s = 0, ext = 'const')

    @property
    def texp(self):
        return self._texp

    @property
    def texpref(self):
        return self._texpref

    @property
    def cdf(self):
        return self._cdf

    @property
    def kde(self):
        return self._kde

    @property
    def min_delay(self):
        return self._min_delay

    @property
    def max_delay(self):
        return self._max_delay

    @min_delay.setter
    def min_delay(self, min_delay):
        self._min_delay = min_delay
        self.__convolve_pdf()

    @max_delay.setter
    def max_delay(self, max_delay):
        self._max_delay = max_delay
        if max_delay:
            self.__convolve_pdf()

    def __convolve_pdf(self):
        """Convolve the posterior distribution with a time delay"""
        if self._max_delay <= self._min_delay or self._max_delay == 0.:
            raise ValueError("Invalid delay choices")
        delay_unc = self._max_delay - self._min_delay
        
        x = np.arange(self._texp.min() - self._max_delay,
                    -1. * (self._texp.min() - self._max_delay),
                    np.min([0.01,delay_unc / 2.]))

        boxcar = ((0. < (x + self._max_delay)) & \
                    ((x + self._max_delay) < delay_unc)).astype(np.float)
        boxcar /= boxcar.sum()

        pdf = self._kde.pdf(x)
        pdf_conv = np.convolve(pdf,boxcar,mode = 'same')
        self._pdf_conv = USpline(x,pdf_conv, k = 1, s = 0, ext = 'zeros')

        # integrate the pdf to get the cdf 
        self._cdf = np.zeros_like(x)
        for i,xi in enumerate(x[1:]):
            xx = np.linspace(x[0],xi,i + 10)
            self._cdf[i+1] = simps(self._pdf_conv(xx),xx)

        m = (x <= 0.) & (self._cdf < 1. - 1e-3)
        # remove values to close together, numerical imprecision
        n = np.diff(self._cdf[m]) > 1e-7
        if not np.all(np.diff(self._cdf[m][:-1][n]) > 0.):
            raise ValueError("CDF values must strictly increase!")

        self._cdfinv = USpline(self._cdf[m][:-1][n], x[m][:-1][n],
            k = 1, s = 0, ext = 'const')

        # test it:
        if np.any(np.isnan(self._cdfinv(self._cdf[m]))):
            raise ValueError("CDF Inverse returns nan!")

    def cdfinv(self, q):
        return self._cdfinv(q)

    def integrate_pdf(self, tmjd_min, tmjd_max, q = 0.9999, tshift = None, weights = None):
        """
        Integrate the pdf 

        Parameters
        ----------
        tmjd_min: array-like
            n-dim, lower integration bounds

        tmjd_max: array-like
            n-dim, upper integration bounds

        {options}

        q: float
            outside this quantile set integral to zero (default: 0.9999)

        tshift: array-like or None
            m-dim, shift for posterior pdf in days

        weights: array-like or None:
            n-dim, weights for integral

        Returns
        -------
        If tshift is none, n-dim array with intgrated posterior density, 
        else if tshift is m-dim array, will return nxm-dim array
        with integrated posterior density for each shift value
        """
        tmin = self._cdfinv((1. - q) / 2.)
        tmax = self._cdfinv((1. + q) / 2.)
        
        if tshift is None and weights is None and self._max_delay == 0.:
            m = (tmjd_min >= tmin + self._texpref) & (tmjd_max <= tmax + self._texpref)
            result = np.zeros_like(tmjd_min)
            for i,t in enumerate(tmjd_min):
                if not m[i]: continue
                result[i] = self._kde.integrate_box_1d(t - self._texpref,
                                            tmjd_max[i] - self._texpref)
            return result

        else:
            if tshift is None:
                tshift = [0.]

            if weights is None:
                weights = np.ones_like(tmjd_min)

            t1 = time()

            # for integration
            x = np.linspace(0.,1.,15)

            # build 3d arrays
            xx,tt0,tsts = np.meshgrid(x, tmjd_min, tshift, indexing = 'ij')
            tt1 = np.meshgrid(x, tmjd_max, tshift,  indexing = 'ij')[1]
            ww = np.meshgrid(x, weights, tshift,  indexing = 'ij')[1]

            # build array for integration from tmin to tmax 
            yy = (tt1 - tt0) * xx + tt0 # build a 2d integration array

            # make a mask for only those time values that are within quantile of 
            # shifted distribution
            m = np.equal(np.greater_equal(tt0 , tmin + self._texpref + tsts),
                        np.less_equal(tt1 , tmax + self._texpref +tsts))

            # combine mask with weights
            m = np.logical_and(m, np.greater(ww, 0.))

            result = np.zeros_like(yy)
            if self._max_delay == 0.:
                result[m] = self._kde.pdf((yy - tsts)[m].flatten() - self._texpref)
            else:
                result[m] = self._pdf_conv((yy - tsts)[m].flatten() - self._texpref)

            return simps(result * ww, yy, axis = 0)




    def __call__(self, x):
        """
        Return the marginalized posterior value at some time x
        from Gaussian kernel density estimate

        Parameter
        ---------
        x: array-like
            array with explosion times

        Returns
        -------
        Array with marginalized posterior
        """
        if self._max_delay == 0.:
            return self._kde.pdf(x)
        else:
            return self._pdf_conv(x)

class GammaRayLogLike(object):
    """
    Class for gamma-ray log likelihood 
    """
    def __init__(self, lcfile, walkerfile, m_neV, Mprog,
        bfield = 'jansson12',
        cosmo = FlatLambdaCDM(H0=0.7 * 100. * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3),
        min_delay = 0., max_delay = 0.,
        spline = dict(k = 2, s = 1e-3, ext = 'extrapolate')):
        """
        Initialize the class

        Parameters
        ----------
        lcfile: str
            path for combined lc file in fits or npy format

        walkerfile: str
            path to MOSFIT results file which contains the walkers

        m_neV: float
            ALP mass in neV

        Mprog: float
            progenitor mass in solar masses
            (currently only 10. and 18. implemented)

        {options}

        bfield: str
            Milky Way Bfield identifier, default: jansson12

        cosmo: `~astropy.cosmology.FlatLambdaCDM`
            used cosmology

        spline: dict
            dictionary with keywords for spline interpolation 
            of likelihood functions

        min_delay: float
            minimum delay between core collapse and
            onset of optical emission in days, default: 0.

        max_delay: float
            maximum delay between core collapse and
            onset of optical emission in days, default: 0.

        """
        if 'fits' in lcfile:
            self._t = Table.read(lcfile)
        elif 'npy' in lcfile:
            self._t = np.load(fname).flat[0]

        self._emin = np.unique(self._t['emin_sed'].data)
        self._emax = np.unique(self._t['emax_sed'].data)
        self._tmin = self._t['tmin'].data
        self._tmax = self._t['tmax'].data
        self._tcen = 0.5 * (self._tmin + self._tmax)
        self._dt = self._tmax - self._tmin

        self._walkerfile = walkerfile
        self._Mprog = Mprog
        self._m_neV = m_neV 
        self._bfield = bfield
        self._cosmo = cosmo
        self._snalpflux = SNALPflux(walkerfile, Mprog = Mprog, cosmo = cosmo)
        self._set_refflux()

        # get the posterior for the explosion time
        self._tpost = TexpPost(walkerfile,
                        min_delay = min_delay,
                        max_delay = max_delay)

        # arrays to store cached g11 and flux for likelihood calculation
        self.__g11cache = None
        self.__fluxcache = None

        # spline interpolations
        self._dlog_spline = []

        # loop over time bins
        for i,dlog in enumerate(self._t['dloglike_scan']):
            # loop energy bins
            self._dlog_spline.append([])
            for j in range(self._emax.size):
                self._dlog_spline[i].append(USpline(self._t['norm_scan'][i,j] * \
                                self._t['ref_flux'][i,j], 
                                dlog[j], **spline))
    @property
    def t(self):
        return self._t

    @property
    def tcen(self):
        return self._tcen

    @property
    def tmin(self):
        return self._tmin

    @property
    def tmax(self):
        return self._tmax

    @property
    def dt(self):
        return self._dt

    @property
    def m_neV(self):
        return self._m_neV

    @property
    def snalpflux(self):
        return self._snalpflux

    @m_neV.setter
    def m_neV(self, m_neV):
        self._m_neV = m_neV
        self._set_refflux()

    @property
    def Mprog(self):
        return self._Mprog

    @Mprog.setter
    def Mprog(self, Mprog):
        self._Mprog = Mprog 
        self._snalpflux = SNALPflux(self._walkerfile,
            Mprog = self._Mprog,
            cosmo = self._cosmo)
        self._set_refflux()

    @property
    def bfield(self):
        return self._bfield

    @bfield.setter
    def bfield(self,bfield):
        self._bfield = bfield
        self._set_refflux()

    @property
    def tpost(self):
        return self._tpost

    @property
    def walkerfile(self):
        return self._walkerfile

    @walkerfile.setter
    def walkerfile(self,walkerfile):
        self._walkerfile = walkerfile
        self._tpost = TexpPost(self._walkerfile)
        self._snalpflux = SNALPflux(self._walkerfile,
            Mprog = self._Mprog,
            cosmo = self._cosmo)
        self._set_refflux()

    def _set_refflux(self, t_sec = 20., g11 = 1.):
        self.__t_sec = t_sec
        self.__g11 = g11
        self.__ref_flux = np.zeros_like(self._emax)
        for i,e in enumerate(self._emax):
            self.__ref_flux[i] = self._snalpflux.integrateGRayFlux(self._emin[i], e,
                            t_sec, g11,
                            self._m_neV, bfield = self._bfield,
                            esteps = 100, eflux = False)

    def flux2coupling(self,flux):
        """Convert a flux array into photon-ALP coupling for each time bin"""
        ff, tt = np.meshgrid(flux, self._dt * 24. * 3600., self._emax,
                            indexing = 'ij')[:-1]
        
        coupling = np.power(ff / self.__ref_flux * \
                        tt / self.__t_sec, 0.25) * self.__g11

        return coupling

    def coupling2flux(self,g11):
        """Convert a photon-ALP coupling array into flux for each time bin"""
        # has dim: g11 x time bins x energy bins
        gg, tt = np.meshgrid(g11, self._dt * 24. * 3600., self._emax,
                                indexing = 'ij')[:-1]
        
        self.__fluxcache = np.power(gg / self.__g11, 4.) * \
                        self.__t_sec / tt * self.__ref_flux

        return self.__fluxcache

    def __call__(self, g11, tmjd, apply_optical = True, tshift = 0.):
        """
        Compute the likelihood as a function of photon-ALP coupling
        and explosion time 

        Parameters
        ----------
        g11: array-like
            an array with the photon-ALP couplings in 10^-11 GeV^-1

        tmjd: float
            explosion time in MJD

        {options}

        apply_optical: bool
            if True, multiply likelhood with optical posterior of explosion time,
            default: True

        tshift: float 
            time in MJD by which the optical posterior is shifted (e.g., to create mock data set).
            Default: 0.

        Returns
        -------
        tuple with arrays of log likelihood and flux
        """
        # determine time bin:
        m = (tmjd >= self._tmin) & (tmjd <= self._tmax)
        if not m.sum():
            raise ValueError('specified time outside of range or outside GTI')
        idm = np.where(m)[0][0]
        if self.__g11cache is not None and np.array_equal(g11, self.__g11cache):
            flux = self.__fluxcache[:,idm,:]
        else:
            flux = self.coupling2flux(g11)[:,idm,:]

        # sum over energy bins
        logl = np.zeros_like(g11)
        for i in range(self._emax.size):
            logl += self._dlog_spline[idm][i](flux[:,i])

        if apply_optical and self._walkerfile is not None:
            ptpost = self._tpost(tmjd - self._tpost.texpref - tshift)
            ptpost[ptpost <= 1e-20] = np.ones(np.sum(ptpost <= 1e-20)) * 1e-20
            logl += np.log(ptpost)
        return logl, flux

class CalcLimits(object):
    """Class to calculate limits, null distribution, etc."""
    def __init__(self, gloglike):
        """
        Initialize the class

        Parameters
        ----------
        gloglike: `snlc.likelihood.GammaRayLogLike`
            the likelihood object
        """
        self._glnl = gloglike

    @property
    def glnl(self):
        return self._glnl

    @staticmethod
    def calcmask(glnl, q):
        """Calculate a mask to blind the data"""
        # mask is true for time interval in which real explosion took place
        tlo = glnl.tpost.cdfinv((1. - q)/2.)
        tup = glnl.tpost.cdfinv((1. + q)/2.)
        if np.any(np.isnan([tlo,tup])):
            raise ValueError("Inverse CDF returned nan!")

        mdata = (glnl.tmin >= tlo + glnl.tpost.texpref) & \
                    (glnl.tmax <= tup + glnl.tpost.texpref)
        return mdata

    def tsim_array(self, q, tstep = 0.01):
        """Build an array of times outside interesting time interval"""
        mdata = CalcLimits.calcmask(self._glnl, q)

        if not mdata.sum():
            raise ValueError("No data points are within time" \
                "interval of quantile {0:.5f}".format(q))

        if self._glnl.tmin[mdata][0] > self._glnl.tmin[0]:
            tlo = np.arange(self._glnl.tmin[0], self._glnl.tmin[mdata][0], tstep)
        else:
            tlo = []
        if self._glnl.tmax[mdata][-1] < self._glnl.tmax[-1]:
            thi = np.arange(self._glnl.tmax[mdata][-1], self._glnl.tmax[-1], tstep)
        else:
            thi = []
        return np.concatenate([tlo,thi])

    @staticmethod
    def select_best_t(totlnl, ptpost = None, **kwargs):
        """
        Profile the combined likelihood over explosion time or select explosion time 
        that maximizes the posterior 
        """
        if ptpost is None:
            # profile
            return totlnl.max(axis = kwargs.get('axis', -1))
        else:
            # select the times that maximize the posterior
            idm = np.argmax(ptpost, kwargs.get('axis', 0))
            if len(totlnl.shape) == 3:
                return np.array([totlnl[i,:,idm[i]] for i in range(idm.size)])
            elif len(totlnl.shape) == 2:
                return totlnl[:,idm]
            else:
                raise ValueError("Dimensions of totlnl not understood")

    @staticmethod
    def findlimits(lnl_maxt, g11, thr = 2.71, axis = -1):
        """
        Find the coupling values that correspond to a certain 
        threshold
        """
        # compute loglikelihood ratio 
        if lnl_maxt.shape > 1:
            llr = -2. * (lnl_maxt.T - lnl_maxt.max(axis = axis)).T
        else:
            llr = -2. * (lnl_maxt - lnl_maxt.max(axis = axis))

        interp = interp1d(g11, llr, axis = axis)

        # make a finer binning of the coupling 
        gint = np.logspace(np.log10(g11[0]), np.log10(g11[-1]), g11.size * 20)
        llr_interp = interp(gint)

        # find the llr values closest to the threshold
        idx = np.argmin(np.abs(llr_interp - thr), axis = axis)
        
        # return the coupling values
        return gint[idx]

    @staticmethod
    def calc_bands(glim, q = [0.05,0.16, 0.5, 0.84, 0.95]):
        """
        From an array of simulated limit values of the coupling, 
        compute the quantiles q for a limit band
        """
        cdf = np.cumsum(glim)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        cdfinv = interp1d(cdf, np.sort(glim))
        return cdfinv(q),q

    def build_null_dist_exp_bands(self, g11 = np.logspace(-2.,1.,3*20 + 1),
                    tshift = None, samples = 10, q = 0.99, best_time = 'profile', thr = 2.71):
        """build a null distribution"""
        mdata = CalcLimits.calcmask(self._glnl, q)

        if tshift is None:
            tsim = self.tsim_array(q) - self._glnl.tpost.texpref
            # draw samples from tsim for shift in tpost
            tshift = np.random.choice(tsim, size = samples, replace = False) 

        cc, tt = np.meshgrid(self._glnl.tcen[~mdata], tshift, indexing = 'ij')

        loglike = np.zeros((self._glnl.tcen[~mdata].size, g11.size))

        # calculate the gray likelihood
        # has shape of tcen x g11
        for it, tcen in enumerate(self._glnl.tcen[~mdata]):
            loglike[it], flux = self._glnl(g11, tcen,
                                        apply_optical = False)

        # calculate the posterior for different shifts
        # via integration over posterior density between orbit min and max times
        # has shape tcen x tshift
        ptpost = self._glnl.tpost.integrate_pdf(tmjd_min = self._glnl.tmin[~mdata],
                                                tmjd_max = self._glnl.tmax[~mdata],
                                                tshift = tshift)
        #ptpost = self._glnl.tpost((cc - self._glnl.tpost.texpref - tt).flatten())
        ptpost[ptpost <= 1e-20] = np.ones(np.sum(ptpost <= 1e-20)) * 1e-20
        #ptpost = ptpost.reshape(tt.shape)

        # mutliply posterior and gray likelihood
        # is now in array shape of dim tshift x g11 x tcen
        res = np.array([loglike.T for i in range(ptpost.shape[-1])])
        res += np.log(ptpost.T[:,np.newaxis,:])

        #likelihood for g = 0
        lnl0 = np.array([self._glnl._t['dloglike_scan'][:,:,0].sum(axis = 1)[~mdata] for i in range(ptpost.shape[-1])])
        lnl0 += np.log(ptpost.T)
        if best_time == 'profile':
            res_maxt = CalcLimits.select_best_t(res, axis = -1)
            lnl0 = lnl0.max(axis = -1)
        elif best_time == 'max_t':
            res_maxt = CalcLimits.select_best_t(res, ptpost = ptpost, axis = 0)
            idm = np.argmax(ptpost, axis = 0)
            lnl0 = np.array([lnl0[i,idm[i]] for i in range(idm.size)])

        # calculate null distribution
        # add the g11 = 0 case:
        res_maxt_w0 = np.vstack([[lnl0], res_maxt.T]).T
        # and calculate the ts distribution
        ts = -2. * (res_maxt_w0[:,0] - res_maxt_w0.max(axis = -1))
        #self._t['dloglike_scan'].data[:,:,0].sum(axis = 1)

        # calculate the limit bands
        # first find coupling values close to assumed threshold
        glim = CalcLimits.findlimits(res_maxt, g11, thr = thr)
        # from limit values, calculate bands
        gband,q = CalcLimits.calc_bands(glim)
        return res_maxt, ts, gband, q

    @staticmethod
    def calc_obs_limits(glnl, g11 = np.logspace(-2.,1.,3*20 + 1),
                    q = 0.99, best_time = 'profile', thr = 2.71):
        """Calculate limits from data"""
        mdata = CalcLimits.calcmask(glnl, q)

        loglike = np.zeros((glnl.tcen[mdata].size, g11.size))

        # calculate the gray likelihood
        # has shape of tcen x g11
        for it, tcen in enumerate(glnl.tcen[mdata]):
            loglike[it], flux = glnl(g11, tcen,
                                        apply_optical = False)

        # calculate the posterior
        # via integration over posterior density between orbit min and max times
        # has shape tcen
        ptpost = glnl.tpost.integrate_pdf(tmjd_min = glnl.tmin[mdata],
                                                tmjd_max = glnl.tmax[mdata],
                                                )
        ptpost[ptpost <= 1e-20] = np.ones(np.sum(ptpost <= 1e-20)) * 1e-20
        ptpost = ptpost.flatten()

        # mutliply posterior and gray likelihood
        # is now in array shape of dim tcen, g11
        res = (loglike.T + np.log(ptpost)).T

        lnl0 = glnl._t['dloglike_scan'][:,:,0].sum(axis = 1)[mdata] + np.log(ptpost)
        if best_time == 'profile':
            res_maxt = CalcLimits.select_best_t(res, axis = 0)
            lnl0 = lnl0.max()
        elif best_time == 'max_t':
            res_maxt = CalcLimits.select_best_t(res, ptpost = ptpost, axis = 0)
            idm = np.argmax(ptpost)
            lnl0 = lnl0[idm]

        # calculate null distribution
        # after appending g11 = 0 case
        res_maxt_w0 = np.concatenate([[lnl0], res_maxt])
        ts = -2. * (res_maxt[0] - res_maxt.max())

        # calculate the limit bands
        # first find coupling values close to assumed threshold
        glim = CalcLimits.findlimits(res_maxt, g11, thr = thr)
        return res_maxt, ts, glim

    @staticmethod
    def combine_limts(logldata, loglsim, pobs, g11, samples = 0, thr = 2.71):
        """
        Combine limit curves

        Parameters
        ----------
        logldata: array-like
            log likelihood curves for data, shape: sources x m_neV x g11 
        loglsim: array-like
            log likelihood curves for simulations, shape: sources x m_neV x samples x g11 
        pobs: array-like
            arrays for each source with observation probability
        """
        # combine the likelihoods
        if samples == 0:
            tot_logl_sim = loglsim.sum(axis = 0)
            tot_logl_data = logldata.sum(axis = 0)

            # loop over masses:
            glim = np.zeros(tot_logl_data.shape[0])
            gband = np.zeros((tot_logl_data.shape[0],5))

            for imass in range(tot_logl_data.shape[0]):
                glim[imass] = CalcLimits.findlimits(tot_logl_data[imass], g11, thr = thr)
                glim_sim =  CalcLimits.findlimits(tot_logl_sim[imass], g11, thr = thr)
                gband[imass],q = CalcLimits.calc_bands(glim_sim)
            return glim, gband, q
        else:
            # has size samples x sources
            include = np.array([binom.rvs(n = 1, p = p, size = samples) for p in pobs]).T.astype(np.bool)

            glim = np.zeros((samples,logldata.shape[1]))
            gband = np.zeros((samples,loglsim.shape[1],5))

            for isample, m in enumerate(include):
                if not m.sum(): continue
                tot_logl_sim = loglsim[m].sum(axis = 0)
                tot_logl_data = logldata[m].sum(axis = 0)

                for imass in range(tot_logl_data.shape[0]):
                    glim[isample, imass] = CalcLimits.findlimits(tot_logl_data[imass], g11, thr = thr)
                    glim_sim =  CalcLimits.findlimits(tot_logl_sim[imass], g11, thr = thr)
                    gband[isample, imass],q = CalcLimits.calc_bands(glim_sim)
            # calcluate the total simulated bands and observed band
            tot_band = np.zeros(gband.shape[1:]) # total band for simulation
            band_data = np.zeros(gband.shape[1:]) # total band for observations
            for imass in range(gband.shape[1]):

                cdf_data = np.cumsum(glim[:,imass][glim[:,imass] > 0.])
                cdf_data = (cdf_data - cdf_data.min()) / (cdf_data.max() - cdf_data.min())

                interp_data = interp1d(cdf_data, np.sort(glim[:,imass][glim[:,imass] > 0.]))

                for iq, qi in enumerate(q):
                    cdf = np.cumsum(gband[:,imass,iq][gband[:,imass,iq] > 0.])
                    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
           
                    interp = interp1d(cdf, np.sort(gband[:,imass,iq][gband[:,imass,iq] > 0.]))
                    idx = np.argmin(np.abs(qi - cdf))
                    tot_band[imass, iq] = interp(qi)

                    band_data[imass,iq] = interp_data(qi)
            return band_data, tot_band, q, include
