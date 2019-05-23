"""
Class to fit an SN light curve
"""

__version__ = 0.1
__author__ = "Manuel Meyer"

# ----- Imports ---------- #
import numpy as np
import logging
import iminuit as minuit
import time
import functools
import copy
import emcee
import corner
import yaml
import snlc
import argparse
import copy
from snlc.getlc import SNLC,mag2flux
from scipy import stats
from collections import OrderedDict
from scipy import optimize as op
from scipy.special import erfi
from numpy import sqrt,pi
from os import path
from haloanalysis.batchfarm import utils,lsf
from astropy.table import Table
from glob import glob
# ------------------------ #

# global reference magnitude and flux
ref = {'mref' : 0., 'Iref' : 2e7 } # some arbitrary mag to flux reference

# nickel decay time
tauNi = 7.605e5 / 24. / 3600. # Nickel-56 decay time in days

# Arnett's Lambda function
def Lambda(x,y):
    """
    Arnett's Lambda function 

    Parameters
    ----------
    x: `~numpy.ndarray`
        n-dim array with x = (t-t0) / tau_m, where tau_m is 
        the scale parameter of the SN light curve (a fit parameter)
    y: float
        y = tau_m / 2 / tau_Ni

    Returns
    -------
    n-dim `~numpy.ndarray` with Arnett's lambda function
    """
    if np.isscalar(x):
        x = np.array([x])
    res = np.zeros(x.size)
    
    m = (y / x > 0.0005) & ((x - y) > -26.) & ((x - y) < 26.) 
    res[m] = np.exp(-2. * x[m]*y) - np.exp(-x[m]*x[m]) + y * sqrt(pi) * np.exp(-(x[m]*x[m] + y*y)) * \
        (erfi(x[m]-y) - erfi(-y))

    # use approximation for small numbers to avoid inf / nan:
    mn = y / x <= 0.0005
    res[mn] = np.exp(-2. * x[mn] * y) - np.exp(-x[mn]*x[mn])
    mn2 = x / y < 0.0005
    res[mn2] = 0.5 * (1. - np.exp(-2. * x[mn2] * y) * ( 1. + 2. * x[mn2] * y))
    
    #print res, 'here', y / x, np.exp(-(x[m]*x[m] + y*y)),erfi(x[m]-y)
    return np.squeeze(res)

# --- Model functions --------- #
def breakoutphase(t, **par):
    """
    Breakout phase model from Cowen et al. (2010):
    flux = a1 / (exp(a2 * sqrt(dt)) - 1) * dt ** 1.6
    where dt = t - t0

    Parameters
    ----------
    t: array, float
        time in MJD

    kwargs
    ------
    t0: float, 
            explosion time in MJD
    a1: float,
        flux normalization
    a2: exponential normalization

    Returns
    -------
    Flux of SN during breakout phase
    """
    if np.isscalar(t):
        t = np.array([t])
    dt = t - par['t0']
    m = dt > 0.
    result = np.zeros(dt.size)
    if np.any(m):
        result[m] = par['a1'] / (np.exp(par['a2'] * np.sqrt(dt[m])) - 1.) * np.power(dt[m], 1.6)
    return np.squeeze(result)

def breakoutexp(t, **par):
    """
    Expontial fit to breakout phase with empirical model from Ofek et al. (2014)
    doi: 10.1088/0004-637X/788/2/154
    flux = a1 * (1 - exp(dt / a2))
    where dt = t - t0

    Parameters
    ----------
    t: array, float
        time in MJD

    kwargs
    ------
    t0: float, 
            explosion time in MJD
    a1: float,
        flux normalization
    a2: characteristic rise time

    Returns
    -------
    Flux of SN during breakout phase
    """
    if np.isscalar(t):
        t = np.array([t])
    dt = t - par['t0']
    m = dt > 0.
    result = np.zeros(dt.size)
    if np.any(m):
        result[m] = par['a1'] * (1. - np.exp(-dt[m] / par['a2'])) 
    return np.squeeze(result)

def expansionphase(t, **par):
    """
    Expansion phase model from Cowen et al. (2010):
    flux = a3 * dt **a4, where a4 was originally taken to be equal to 2
    and dt = t - t0

    Parameters
    ----------
    t: array, float
        time in MJD

    kwargs
    ------
    t0: float, 
            explosion time in MJD
    a3: float,
        flux normalization
    a4: float,
        exponent of dt

    Returns
    -------
    Flux of SN during breakout phase
    """
    if np.isscalar(t):
        t = np.array([t])
    dt = t - par['t0']
    m = dt > 0.
    result = np.zeros(dt.size)
    if np.any(m):
        result[m] = par['a3'] * np.power(dt[m],par['a4'])
    return np.squeeze(result)

def expansionphase_arnett(t, **par):
    """
    Expansion phase model from Arnett (1982)
    flux = a3 * Lambda(x,y)
    where x = dt / a4 and y = a4 / tau_ni
    where dt = t - t0, and tau_ni the decay time of Nickel-56

    Parameters
    ----------
    t: array, float
        time in MJD

    kwargs
    ------
    t0: float, 
            explosion time in MJD
    a3: float,
        flux normalization
    a4: float,
        SN time scale in days

    Returns
    -------
    Flux of SN during breakout phase
    """
    if np.isscalar(t):
        t = np.array([t])
    dt = t - par['t0']
    x = dt / par['a4']
    y = par['a4'] / 2. / tauNi
    m = dt > 0.
    result = np.zeros(dt.size)

    if np.any(m):
        result[m] = par['a3'] * Lambda(x[m],y)
    return np.squeeze(result)

# --------------------------------------- #
# --- functions for initial guesses ----- #
def init_guess_breakout(tdisc, time, mag, a2 = 5., da = 1., maxt = 5.):
    """
    initial guess for breakout phase

    Parameters
    ----------
    tdisc: float
        discovery date
    time: `~numpy.ndarray`
        measured times
    mag: `~numpy.ndarray`
        measured magnitudes (no upper limits)

    kwargs
    ------
    a2: float, 
            initial guess for a2 parameter
    da: float,
        increment and decrement for a2 to find root
    maxt: float,
        maximum time of breakout phase

    Returns
    -------
    dict with initial guesses
    """
    par = {}

    par['t0'] = tdisc

    # convert mag to flux
    y = mag2flux(mag,**ref)

    # only use first days
    mask = time - par['t0'] < maxt

     
    itmax = np.argmax(y[mask])

    # get measurement closest to maximum
    ymax = y[mask][itmax]
    dt = time[mask][itmax] - par['t0']

    # determine a2 such that maximum flux coincides with max of function
    # the derivative = 0:
    df_dt = lambda a2,dt: (1. - a2 / 3.2 * np.sqrt(dt) ) * np.exp(a2 * np.sqrt(dt)) - 1.
    logging.info('root finding interval: [{0},{1}] gives [{2},{3}]'.format(a2 - 1, a2 + 1,
                        df_dt(a2 - da,dt), df_dt(a2 + da,dt)))
    try:

        res =  op.brentq(df_dt, a2 - da , a2 + da, args = (dt))
        if res > 1e-4:
            par['a2'] = res
            logging.info('Root found at {0[a2]:.2f}.'.format(par))
            else:
            par['a2'] = a2
    except ValueError:
        logging.warning('Could not run brentq, setting a2 to {0:.2f}'.format(a2))
        par['a2'] = a2

    
    # get initial guesses
    if dt > 10.:
        logging.warning('Estimated maximum more than 10 days after discovery!')

    par['a1'] = ymax * np.power(dt,-1.6) * (np.exp(par['a2'] * np.sqrt(dt)) - 1.)
    return par

def init_guess_breakout_exp(tdisc, time, mag, a2 = 1., maxt = 5.):
    """
    initial guess for breakout phase for exponential fit

    Parameters
    ----------
    tdisc: float
        discovery date
    time: `~numpy.ndarray`
        measured times
    mag: `~numpy.ndarray`
        measured magnitudes (no upper limits)

    kwargs
    ------
    a2: float, 
            initial guess for a2 parameter
    maxt: float,
        maximum time of breakout phase

    Returns
    -------
    dict with initial guesses
    """
    par = {}

    par['t0'] = tdisc

    # convert mag to flux
    y = mag2flux(mag,**ref)

    # exlcude upper limits and only use first days
    mask = time - par['t0'] < maxt

     
    itmax = np.argmax(y[mask])

    # get measurement closest to maximum
    ymax = y[mask][itmax]
    dt = time[mask][itmax] - par['t0']

    par['a2'] = a2
    
    # get initial guesses
    if dt > 10.:
        logging.warning('Estimated maximum more than 10 days after discovery!')

    par['a1'] = ymax / (1. - np.exp(-dt / par['a2']))
    return par

def init_guess_expansion(tdisc, time, mag, delay = 10., subtract = lambda t: 0.):
    """
    initial guess for expansion phase

    Parameters
    ----------
    tdisc: float
        discovery date
    time: `~numpy.ndarray`
        measured times
    mag: `~numpy.ndarray`
        measured magnitudes (no upper limits)

    kwargs
    ------
    delay: float, 
            delay time in days when expansion should dominate.
        Flux estimated from flux corresponding to closest time
    substract: function pointer,
        additional flux that might contribute at time of delay

    Returns
    -------
    dict with initial guesses
    """
    par = {}

    # take discovery date as initial guess for explosion time
    par['t0'] = tdisc

    # get measurement closest to maximum
    it = np.argmin(np.abs(time - (par['t0'] + delay)))

    # convert mag to flux
    y = mag2flux(mag,**ref)

    # get initial guesses
    par['a4'] = 2.
    par['a3'] = (y[it] - subtract(time[it])) / delay ** par['a4']
    if par['a3'] < 0.: par['a3'] = 1e-5
    return par

def init_guess_arnett(tdisc, time, mag, delay = 10., y = 2., subtract = lambda t: 0.):
    """
    initial guess for expansion phase model with arnett

    Parameters
    ----------
    tdisc: float
        discovery date
    time: `~numpy.ndarray`
        measured times
    mag: `~numpy.ndarray`
        measured magnitudes (no upper limits)
    y: float,
       estimate for tauM (a4), where y = tauM / 2 / tauNi

    kwargs
    ------
    delay: float, 
            delay time in days when expansion should dominate.
        Flux estimated from flux corresponding to closest time
    substract: fuction pointer,
        additional flux that might contribute at time of delay

    Returns
    -------
    dict with initial guesses
    """
    par = {}

    # take discovery date as initial guess for explosion time
    par['t0'] = tdisc

    # get measurement closest to maximum
    m = np.isfinite(mag)
    it = np.argmin(np.abs(time[m] - (par['t0'] + delay)))

    # convert mag to flux
    f = mag2flux(mag[m],**ref)

    # get initial guesses
    par['a4'] = 2. * y * tauNi
    x = delay / par['a4']
    par['a3'] = (f[it] - subtract(time[it])) / Lambda(x,y)

    if par['a3'] < 0.: par['a3'] = 1e-5
    return par

# minuit defaults ------------- #
minuit_def = {
        'verbosity': 0,        
        'int_steps': 0.01,
        'strategy': 2,
        'model': '',
        #'tol': 1.,
        #'tol': 1e-10,
        'tol': 1e-2,
        'up': 0.5,
        'max_tol_increase': 3000.,
        'ncall': 2000,
        'scan_bound': 5.,
        'pedantic': True,
        'precision': None,
        'pinit': {'t0' : 0.,
                'a1': 1.,
                'a2': 1.,
                'a3': 1.,
                'a4': 0.3},
        'fix' : {'t0' : False,
                'a1': False,
                'a2': False,
                'a3': False,
                'a4': False },
        'limits' : {'t0' : [0., 1e10],
                'a1':  [-5., 10.],
                'a2':  [0., 20.],
                'a3':  [-5., 10.],
                'a4':  [0.01, 2.]},
        'tmin' : -1e10,
        'tmax' : 1e10,
        'model': 'break+expansion',
        't0_steps': 40,
        'chi2': False
        }

def setDefault(func = None, passed_kwargs = {}):
    """
    Read in default keywords of the simulation and pass to function
    """
    if func is None:
        return functools.partial(setDefault, passed_kwargs = passed_kwargs)
    @functools.wraps(func)
    def init(*args, **kwargs):
        for k in passed_kwargs.keys():
            kwargs.setdefault(k,passed_kwargs[k])
        return func(*args, **kwargs)
    return init
# ----------------------------- #

class FitSNLC(object):
    """
    Class to fit an SN light curve in multiple bands simultaneously
    """
    @setDefault(passed_kwargs = minuit_def)
    def __init__(self,time,mag,dmag,tdisc,**kwargs):
        """
        Initialize the class

        Parameters
        ----------
        time: `~numpy.ndarray`
            measured times
        mag: `~numpy.ndarray`
            measured magnitudes (no upper limits)
        dmag: `~numpy.ndarray`
            uncertainties measured magnitudes (no upper limits)
        tdisc: float
            discovery date

        kwargs
        ------
        tmin: float,
            min time for fit
        tmax: float,
            max time for fit
        model: string
            identifier for the model. Possibilities are:
            - "break+expand" breakout and expansion phase, expansion phase modeled as 
                    simple quadratic expansion as in Cowen et al. (2010), modified 
                so that it can be non-quadratic
            - "break+arnett" breakout and expansion phase, expansion phase modeled as 
                    with function from Arnett (1982)        
            - "breakexp+expand" exponential breakout and expansion phase, breakout modeled 
                    with simple exponential function of Ofek et al. (2014)
            - "breakexp+arnett" breakout and expansion phase,  breakout modeled 
                    with simple exponential function of Ofek et al. (2014) and
                expansion phase modeled as with function from Arnett (1982)        
        """
        kwargs.setdefault('name','sn')
        self.name = kwargs['name']


        m = (time >= kwargs['tmin']) & (time <= kwargs['tmax'])
        self._t = time[m]
        self._y = mag2flux(mag, **ref)[m]
        self._dy = np.abs(np.array([self._y - mag2flux(mag + dmag,**ref)[m],
                    mag2flux(mag-dmag,**ref)[m] - self._y]).mean(axis = 0))

        self.t_first_data_point = tdisc
        self.__set_llhs()
        self.modelname = kwargs['model']

        self.parnames = ['t0','a1','a2']

        # Set the breakout model
        if kwargs['model'].find('break') == 0 and kwargs['model'].find('breakexp') < 0:
            self.breakout = lambda t,**par : breakoutphase(t,**par)
        elif kwargs['model'].find('breakexp') == 0:
            self.breakout = lambda t,**par : breakoutexp(t,**par)
        else:
            raise ValueError('Model {0[model]:s} unknown!'.format(kwargs))

        # Set the expansion model
        if kwargs['model'].find('expand') > 0:
            self.expansion = lambda t,**par : expansionphase(t,**par)
            self.parnames += ['a3','a4']
        elif kwargs['model'].find('arnett') > 0:
            self.expansion = lambda t,**par : expansionphase_arnett(t,**par)
            self.parnames += ['a3','a4']
        else:
            self.expansion = lambda t,**par : np.squeeze(np.zeros(t.size))

        self.flux_model = lambda t,**par : self.breakout(t,**par) + self.expansion(t,**par)

        self.corr  = []
        self.chi2 = kwargs['chi2']
        return

    @property
    def llhs(self):
        return self._llhs

    @property
    def y(self):
        return self._y

    @property
    def dy(self):
        return self._dy

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self,t):
        self._t = t
        return

    @y.setter
    def y(self,y):
        self._y = y
        self.__set_llhs()
        return

    @dy.setter
    def dy(self,dy):
        self._dy = dy
        self.__set_llhs()
        return

    def __set_llhs(self):
        """ get a likelihood curve for each data point """
        self.maxL = 0.
        self._llhs = []
        for j,f in enumerate(self._y):
            self._llhs.append( stats.norm(loc = f, scale = self._dy[j]  ) )
            self.maxL -= self._llhs[-1].pdf(self._y[j])
        return

### define funtion to do the actual fitting
    def __calcLikelihood(self,*args):
        """
        likelihood function passed to iMinuit
        """
        params = {'t0': args[0]}
        for i,x in enumerate(args[1:]):
            if i == 0 or i == 2:
                params.update({'a{0:n}'.format(i+1) : np.power(10.,x) })
            else:
                params.update({'a{0:n}'.format(i+1) : x })
        return self.returnLikelihood(params)

    def __wrapLikelihood(self,x):
        """
        likelihood function passed to scipy.optimize 
        """
        params = {}
        if not self.fitarg['fix_t0']:
            params['t0'] = x[0]
        else:
            params['t0'] = self.fitarg['t0']

        for i in range(1,len(self.parnames)):
            if not self.fitarg['fix_a{0:n}'.format(i)]:
                if i == 1 or i == 3:
                    params['a{0:n}'.format(i)] = np.power(10.,x[i])
                else:
                    params['a{0:n}'.format(i)] = x[i]
            else:
                if i == 1 or i == 3:
                    params['a{0:n}'.format(i)] = np.power(10.,self.fitarg['a{0:n}'.format(i)])
                else:
                    params['a{0:n}'.format(i)] = self.fitarg['a{0:n}'.format(i)]
        return self.returnLikelihood(params)

    def returnLikelihood(self,params):
        """
        Calculate likelihood function for all flux measurements and bands

        Parameters
        ----------
        params:        dict,
                dictionary with parameters of fitting parameters

        Returns 
        -------
        float with likelihood 
        """

        logL = 0.
        for j,l in enumerate(self._llhs):
            t = self._t[j] # the time for that band and flux
            flux = self.flux_model(t,**params)

            if self.chi2:
                logL += (self._y[j] - flux) ** 2. / \
                            self._dy[j] ** 2.
            else:
                logL += l.pdf(flux)

        #print params, logL
        if self.chi2:
            return logL
        else:
            return -1. * logL

    @setDefault(passed_kwargs = minuit_def)
    def fill_fitarg(self, **kwargs):
        """
        Helper function to fill the dictionary for minuit fitting
        """
        # set the fit arguments
        fitarg = {}
        fitarg.update(kwargs['pinit'])
        for k in kwargs['limits'].keys():
            fitarg['limit_{0:s}'.format(k)] = kwargs['limits'][k]
            fitarg['fix_{0:s}'.format(k)] = kwargs['fix'][k]
            if k == 't0':
                fitarg['error_{0:s}'.format(k)] = kwargs['int_steps']
            else: 
                fitarg['error_{0:s}'.format(k)] = kwargs['pinit'][k] *\
                                                kwargs['int_steps']
                                                
        fitarg = OrderedDict(sorted(fitarg.items()))
        return fitarg


    @setDefault(passed_kwargs = minuit_def)
    def run_migrad(self, fitarg, **kwargs):
        """
        Helper function to initialize migrad and run the fit.
        Initial parameters are estimated with scipy fit.
        """
        self.fitarg = fitarg
        logging.info(fitarg)

        # ---- perfrom the scipy initial fit
        values = []
        bounds = []
        for n in self.parnames:
            values.append(fitarg[n])
            bounds.append(fitarg['limit_{0:s}'.format(n)])

        self.res = op.minimize(self.__wrapLikelihood, 
                    values, 
                    bounds = bounds,
                    method='TNC',
                    #method='Powell',
                    options={'maxiter': kwargs['ncall']} #'xtol': 1e-20, 'eps' : 1e-20, 'disp': True}
                    #tol=None, callback=None, 
                    #options={'disp': False, 'minfev': 0, 'scale': None, 
                                #'rescale': -1, 'offset': None, 'gtol': -1, 
                                #'eps': 1e-08, 'eta': -1, 'maxiter': kwargs['ncall'], 
                                #'maxCGit': -1, 'mesg_num': None, 
                                #'ftol': -1, 'xtol': -1, 'stepmx': 0, 'accuracy': 0}
                    )
        logging.info(self.res)

        if np.all(np.isfinite(self.res.x)):
            for i,n in enumerate(self.parnames):
                fitarg[n] = self.res.x[i] + np.random.rand(1) * 1e-3

        logging.info(fitarg)

        # --- define the minuit function with arbitrary param names
        string_args = ", ".join(self.parnames)
        global f # needs to be global for eval to find it
        f = lambda *args: self.__calcLikelihood(*args)
        
        cmd_string = "lambda {0}: f({0})".format(string_args)
        logging.info(cmd_string)

        # work around so that the parameters get names for minuit
        minimize_f = eval(cmd_string, globals(), locals())

        # ----- Run migrad
        self.m = minuit.Minuit(minimize_f, 
            print_level =kwargs['verbosity'],
            errordef = 1. if self.chi2 else 0.5, 
            **fitarg)

        self.m.tol = kwargs['tol']
        self.m.strategy = kwargs['strategy']

        logging.debug("tol {0:.2f}, strategy: {1:n}".format(
               self.m.tol,self.m.strategy))

        self.m.migrad(ncall = kwargs['ncall']) #, precision = kwargs['precision'])
        logging.debug("Migrad minimization finished")
        return 

    @setDefault(passed_kwargs = minuit_def)
    def __call__(self,profile = True, **kwargs):
        """
        Run the fit
        """
        fitarg = self.fill_fitarg(**kwargs)

        t1 = time.time()

        self.run_migrad(fitarg, **kwargs)

        try:
            self.m.hesse()
            logging.debug("Hesse matrix calculation finished")
        except RuntimeError as e:
            logging.warning(
                "*** Hesse matrix calculation failed: {0}".format(e)
            )

        fmin = self.m.get_fmin()

        if not self.m.migrad_ok() and fmin['is_above_max_edm']:
            logging.warning(
                'Migrad did not converged, is above max edm. Increasing tol.'
                )
            tol = self.m.tol
            self.m.tol *= self.m.edm /(self.m.tol * 0.001 * self.m.errordef )

            logging.info('New tolerance : {0}'.format(self.m.tol))
            if self.m.tol >= kwargs['max_tol_increase']:
                logging.warning(
                    'New tolerance to large for required precision'
                )
            else:
                self.m.migrad(
                    ncall = kwargs['ncall'], 
                    precision = kwargs['precision']
                    )
                logging.info(
                    'Migrad status after second try: {0}'.format(
                        self.m.migrad_ok()
                        )
                    )
                self.m.tol = tol

        fmin = self.m.get_fmin()

        if not fmin.hesse_failed:
            try:
                self.corr.append(self.m.np_matrix(correlation=True))
            except:
                self.corr.append(-1)

        if self.m.migrad_ok():
            if profile:
                # get the likelihood profile for the 
                # explosion time t0
                logging.info('stepping over t0 within' \
                            + '{0[scan_bound]:.2f}sigma with {0[t0_steps]:n} steps'.format(
                                    kwargs))
                self.t0array, self.delta_logL, mr = self.m.mnprofile(vname='t0',
                        bound =kwargs['scan_bound'], 
                        #bound = ( self.m.values['t0'] - self.m.errors['t0'] * kwargs['scan_bound'], 
                    #            self.t_first_data_point), 
                        bins = kwargs['t0_steps'], 
                        subtract_min = True)

                self.t0array = np.array(self.t0array)[np.invert(np.isnan(self.delta_logL))]
                self.delta_logL = np.array(self.delta_logL)[np.invert(np.isnan(self.delta_logL))]


                # getting the minos errors for t0:
                logging.info('Getting minos error for t0')
                self.m_intervals = []
                for i in range(1,4):
                    r = self.m.minos('t0',i)
                    self.m_intervals.append([r['t0']['lower'],r['t0']['upper']])
                self.m_intervals = np.array(self.m_intervals)
        else:
            logging.warning(
                '*** migrad minimum not ok! Printing output of get_fmin'
                )
            logging.warning('{0:s}:\t{1}'.format('*** has_accurate_covar',
                fmin.has_accurate_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_covariance',
                fmin.has_covariance))
            logging.warning('{0:s}:\t{1}'.format('*** has_made_posdef_covar',
                fmin.has_made_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_posdef_covar',
                fmin.has_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_reached_call_limit',
                fmin.has_reached_call_limit))
            logging.warning('{0:s}:\t{1}'.format('*** has_valid_parameters',
                fmin.has_valid_parameters))
            logging.warning('{0:s}:\t{1}'.format('*** hesse_failed',
                fmin.hesse_failed))
            logging.warning('{0:s}:\t{1}'.format('*** is_above_max_edm',
                fmin.is_above_max_edm))
            logging.warning('{0:s}:\t{1}'.format('*** is_valid',
                fmin.is_valid))

        logging.info('fit took: {0}s'.format(time.time() - t1))
        
        self.mvalues = copy.deepcopy(self.m.values)
        self.fvalues = {}
        for i,n in enumerate(self.parnames):
            if n == 'a1' or n == 'a3':
                self.fvalues[n] = np.power(10.,self.res.x[i])
                self.mvalues[n] = np.power(10.,self.mvalues[n])
            else:
                self.fvalues[n] = self.res.x[i]

        return 

    def _lnprior(self, x):
        for i,p in enumerate(self.parnames):
            if not i:
                prior = self.fitarg['limit_{0:s}'.format(p)][0] \
                        < x[i] < self.fitarg['limit_{0:s}'.format(p)][1] 
            else:
                prior = prior & \
                    (self.fitarg['limit_{0:s}'.format(p)][0] \
                        < x[i] < self.fitarg['limit_{0:s}'.format(p)][1])

        if prior:
            return 0.
        else:
            return -np.inf
            return

    def _lnprob(self, x):
        lp = self._lnprior(x)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - self.__wrapLikelihood(x)

    @setDefault(passed_kwargs = minuit_def)
    def mcmc(self, steps = 1000, nwalkers = 100, offset = 1e-4, start_sample = 100,
                control_plots = True, threads = 1, plotdir = './',**kwargs):
        """
        Perform an MCMC sampling of the posterior
        """
        self.fitarg = self.fill_fitarg(**kwargs)

        values = []
        for n in self.parnames:
            values.append(self.fitarg[n])

        ndim, nwalkers = len(values), nwalkers
        pos = [np.array(values) + offset * np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob,
                            threads = threads)
        
        logging.info('Starting MCMC')
        t1 = time.time()
        sampler.run_mcmc(pos, steps)
        logging.info('Done. MCMC took {0:.2f} s'.format(time.time() - t1))

        self.samples = sampler.chain[:, start_sample:, :].reshape((-1, ndim))
        #self.samples[:, 1] = np.power(10.,self.samples[:, 1])
        #if len(values) > 3:
            #self.samples[:, 3] = np.power(10.,self.samples[:, 3])

        # get the best fit values and 3,2,1 sigma error bands
        perc = 0.5 * np.array([2.7e-3, 4.55e-2, 0.3173])
        perc = np.concatenate((perc,[0.5], 1. - perc[::-1]))

        # 3 sigma bound
        self.bf_mcmc = np.array(map(lambda v: (v[3], v[-1]-v[3], v[3]-v[0]),
            zip(*np.percentile(self.samples, 100. * perc,
            axis=0))))

        # generating control plots
        if control_plots:
            import matplotlib.pyplot as plt

            logging.info('Generating control plots...')
            plt.figure(figsize = (8,3*5))
            for i in range(sampler.chain.shape[-1]):
                ax = plt.subplot(sampler.chain.shape[-1],1,i+1)
                for j in range(sampler.chain.shape[0]):
                    plt.plot(range(sampler.chain.shape[1]), sampler.chain[j,:,i],
                        ls = '-', lw = 0.5, color = 'k', alpha = 0.5)
                plt.ylabel(self.parnames[i])
            plt.xlabel('sample')
            plt.savefig(path.join(plotdir,'sampler_{0:s}_{1:s}.png'.format(self.name, self.modelname)),
                format = 'png', dpi = 200)
            plt.close()

            corner.corner(self.samples, labels=self.parnames,
              truths=values, 
              quantiles=perc[2:4],
              bins = 20
             )
            plt.savefig(path.join(plotdir,'corner_{0:s}_{1:s}.png'.format(self.name, self.modelname)),
                format = 'png', dpi = 200)
            logging.info('Done')
            plt.close()

        return

@lsf.setLsf
def submit_to_lsf(config, **kwargs):
    """Submit all source jobs"""
    sns = Table.read(config['sntable'])
    njobs = len(sns)

    script = path.join(path.dirname(snlc.__file__), 'scripts/run_fit.py')

    utils.mkdir(config['plotdir'])
    utils.mkdir(config['outdir'])

    # check for missing files
    if not config['snset'] == 'ASASSN':
        fs = glob(path.join(config['outdir'],'fit_result_*{0[snset]:s}*_breakexp.fits'.format(config)
            ))
    else:
        fs = glob(path.join(config['outdir'],'fit_result_*_breakexp.fits'
            ))
        fi = copy.deepcopy(fs)
        for f in fi:
            if f.find('PTF') >= 0 or f.find('1987') >= 0:
                fs.remove(f)
    names_present = [path.basename(f).split('_')[2] for f in fs]
    missing = []
    for i,n in enumerate(sns['name']):
        if not n in names_present:
            missing.append(i+1)
    logging.info('SNe present: {0}'.format(names_present))
    logging.info('There are {0:n} files missing.'.format(len(missing)))

    if len(missing) and len(missing) < njobs:
        njobs = missing
    # check for missing files

    if len(missing):
        kwargs['logdir'] = utils.mkdir(path.join(
                            config['outdir'],'{0[snset]:s}/log/'.format(config)))
        kwargs['tmpdir'] = utils.mkdir(path.join(
                            config['outdir'],'{0[snset]:s}/tmp/'.format(config)))


        kwargs['jname'] = config['snset']
        config['configname'] = 'fit'
        lsf.submit_lsf(script,
            config,'',
            njobs, 
            **kwargs)
    else:
        logging.info('All files are present')
    return

@lsf.setLsf
def main(**kwargs):
    usage = "usage: %(prog)s"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('--conf', required = True)
    parser.add_argument('--dry', default = 0, type = int)
    parser.add_argument('--time', default = '09:59',help='Max time for lsf cluster job')
    parser.add_argument('--concurrent', default = 0,help='number of max simultaneous jobs', type=int)
    parser.add_argument('--sleep', default = 10,help='seconds to sleep between job submissions', type=int)
    parser.add_argument('--overwrite', default = 0,help='overwrite existing single files', type=int)
    args = parser.parse_args()
    kwargs['dry'] = args.dry
    kwargs['time'] = args.time
    kwargs['concurrent'] = args.concurrent
    kwargs['sleep'] = args.sleep
    
    utils.init_logging('DEBUG', color = True)
    config = yaml.load(open(args.conf))
    submit_to_lsf(config, **kwargs)

    return

if __name__ == '__main__':
    main()
