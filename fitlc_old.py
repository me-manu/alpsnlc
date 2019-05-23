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
from snlc.getlc import SNLC,mag2flux
from scipy import stats
from collections import OrderedDict
from scipy import optimize as op
from scipy.special import erfi
from numpy import sqrt,pi
# ------------------------ #

# global reference magnitude and flux
ref = {'mref' : 0., 'Iref' : 2e7 } # some arbitrary mag to flux reference

# nickel decay time
tauNi = 7.605e5 / 24. / 3600. # Nickel-56 decay time in days

# Arnett's Lambda function
Lambda = lambda x,y: np.exp(-2. * x*y) - np.exp(-x*x) + y * sqrt(pi) * np.exp(-(x*x + y*y)) * \
	(erfi(x-y) - erfi(-y))

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
    dt = t - par['t0']
    return par['a1'] / (np.exp(par['a2'] * np.sqrt(dt)) - 1.) * np.power(dt, 1.6)

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
    dt = t - par['t0']
    return par['a1'] * (1. - np.exp(-dt / par['a2'])) 

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
    dt = t - par['t0']
    return par['a3'] * np.power(dt,par['a4'])

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
    dt = t - par['t0']
    x = dt / par['a4']
    y = par['a4'] / 2. / tauNi
    return par['a3'] * Lambda(x,y)

# --------------------------------------- #
# --- functions for initial guesses ----- #
def init_guess_breakout(sn, tel, band, a2 = 5., da = 1., maxt = 5.):
    """
    initial guess for breakout phase

    Parameters
    ----------
    sn: snlc::getlc::SNLC object,
	SN light curve object
    tel: string, 
        telescope under consideration
    band: string, 
        band under consideration

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

    # discovery date
    tdisc =  sn.get_dates_mjd(datetype = 'discover')[0] 
    # first measurement
    t1 = sn.dat[tel][band].t[0]

#    par['t0'] = 0.5 * (tdisc + t1)
    par['t0'] = t1 - 1e-2

    # convert mag to flux
    y = mag2flux(sn.dat[tel][band].m,**ref)

    # exlcude upper limits and only use first days
    mask = (sn.dat[tel][band].dm > 0.) & \
	    (sn.dat[tel][band].t - par['t0'] < maxt)

     
    itmax = np.argmax(y[mask])

    # get measurement closest to maximum
    ymax = y[mask][itmax]
    dt = sn.dat[tel][band].t[mask][itmax] - par['t0']

    # determine a2 such that maximum flux coincides with max of function
    # the derivative = 0:
    df_dt = lambda a2,dt: (1. - a2 / 3.2 * np.sqrt(dt) ) * np.exp(a2 * np.sqrt(dt)) - 1.
    logging.info('root finding interval: [{0},{1}] gives [{2},{3}]'.format(a2 - 1, a2 + 1,
			df_dt(a2 - da,dt), df_dt(a2 + da,dt)))
    try:
	par['a2'] = op.brentq(df_dt, a2 - da , a2 + da, args = (dt))
	logging.info('Root found at {0[a2]:.2f}.'.format(par))
    except ValueError:
	logging.warning('Could not run brentq, setting a2 to {0:.2f}'.format(a2))
	par['a2'] = a2

    
    # get initial guesses
    if dt > 10.:
	logging.warning('Estimated maximum more than 10 days after discovery!')

    par['a1'] = ymax * np.power(dt,-1.6) * (np.exp(par['a2'] * np.sqrt(dt)) - 1.)
    return par

def init_guess_breakout_exp(sn, tel, band, a2 = 1., maxt = 5.):
    """
    initial guess for breakout phase for exponential fit

    Parameters
    ----------
    sn: snlc::getlc::SNLC object,
	SN light curve object
    tel: string, 
        telescope under consideration
    band: string, 
        band under consideration

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

    # discovery date
    tdisc =  sn.get_dates_mjd(datetype = 'discover')[0] 
    # first measurement
    t1 = sn.dat[tel][band].t[0]

#    par['t0'] = 0.5 * (tdisc + t1)
    par['t0'] = t1 - 1e-2

    # convert mag to flux
    y = mag2flux(sn.dat[tel][band].m,**ref)

    # exlcude upper limits and only use first days
    mask = (sn.dat[tel][band].dm > 0.) & \
	    (sn.dat[tel][band].t - par['t0'] < maxt)

     
    itmax = np.argmax(y[mask])

    # get measurement closest to maximum
    ymax = y[mask][itmax]
    dt = sn.dat[tel][band].t[mask][itmax] - par['t0']

    par['a2'] = a2
    
    # get initial guesses
    if dt > 10.:
	logging.warning('Estimated maximum more than 10 days after discovery!')

    par['a1'] = ymax / (1. - np.exp(-dt / par['a2']))
    return par

def init_guess_expansion(sn, tel, band, t0, delay = 10., subtract = lambda t: 0.):
    """
    initial guess for expansion phase

    Parameters
    ----------
    sn: snlc::getlc::SNLC object,
	SN light curve object
    tel: string, 
        telescope under consideration
    band: string, 
        band under consideration
    t0: float, 
	estimate for explosion time

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
    par['t0'] = t0

    # get measurement closest to maximum
    it = np.argmin(np.abs(sn.dat[tel][band].t - (par['t0'] + delay)))

    # convert mag to flux
    y = mag2flux(sn.dat[tel][band].m,**ref)

    # get initial guesses
    par['a4'] = 2.
    par['a3'] = (y[it] - subtract(sn.dat[tel][band].t[it])) / delay ** par['a4']
    if par['a3'] < 0.: par['a3'] = 1e-5
    return par

def init_guess_arnett(sn, tel, band, t0, delay = 10., y = 2., subtract = lambda t: 0.):
    """
    initial guess for expansion phase model with arnett

    Parameters
    ----------
    sn: snlc::getlc::SNLC object,
	SN light curve object
    tel: string, 
        telescope under consideration
    band: string, 
        band under consideration
    t0: float, 
	estimate for explosion time
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
    par['t0'] = t0

    # get measurement closest to maximum
    it = np.argmin(np.abs(sn.dat[tel][band].t - (par['t0'] + delay)))

    # convert mag to flux
    f = mag2flux(sn.dat[tel][band].m,**ref)

    # get initial guesses
    par['a4'] = 2. * y * tauNi
    x = delay / par['a4']
    par['a3'] = (f[it] - subtract(sn.dat[tel][band].t[it])) / Lambda(x,y)
    print subtract(sn.dat[tel][band].t[it]), sn.dat[tel][band].t[it]
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
	'scan_bound': 9.,
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
	'tmin' : 0.,
	'tmax' : 1e10,
	'model': 'break+exp',
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
    def __init__(self,sn,tel,bands,**kwargs):
	"""
	Initialize the class

	Parameters
	----------
	sn: snlc::getlc::SNLC object,
	    SN light curve object
	tel: string,
	    name of instrument for which light curve is fitted
	bands: list,
	    list with strings with the bands included in the fit

	kwargs
	------
	tmin: float,
	    min MJD for fit
	tmax: float,
	    max MJD for fit
	model: string
	    identifier for the model. Possibilities are:
	    - "break+exp" breakout and expansion phase, expansion phase modeled as 
	    	simple quadratic expansion as in Cowen et al. (2010)
	    - "break+arnett" breakout and expansion phase, expansion phase modeled as 
	    	with function from Arnett (1982)	
	"""

	self.llhs = {}
	self.y = {}
	self.dy = {}
	self.t = {}
	self.mask = {}
	self.maxL = {}
	self.bands = bands
	self.t_first_data_point = 1e10
	# get the data and build Gaussian likelihood curves from them
	for i,b in enumerate(bands):
	    if not i:
		self.llhs[b] = []
		self.t[b] = []
		self.y[b] = []
		self.dy[b] = []
		self.mask[b] = []

	    # mask min, max times and ULs
	    self.mask[b] = (sn.dat[tel][b].dm > 0) & (sn.dat[tel][b].t > kwargs['tmin']) & \
		    (sn.dat[tel][b].t < kwargs['tmax'])
	    if not np.sum(self.mask[b]):
		logging.warning('no data points remaining for {0:s} band {1:s}'.format(tel,b))

	    # convert magnitude to flux
	    # and symmetrize errors
	    self.y[b] = mag2flux(sn.dat[tel][b].m,**ref)
	    self.dy[b] = np.abs(np.array([self.y[b] - mag2flux(sn.dat[tel][b].m + sn.dat[tel][b].dm,**ref),
		    mag2flux(sn.dat[tel][b].m - sn.dat[tel][b].dm,**ref) - self.y[b]]).mean(axis = 0))

	    self.t[b] = sn.dat[tel][b].t
	    # find first data point 
	    if np.min(self.t[b][self.mask[b]]) < self.t_first_data_point:
		self.t_first_data_point = np.min(self.t[b][self.mask[b]])

	    # get a likelihood curve for each data point
	    self.maxL[b] = 0.
	    for j,f in enumerate(self.y[b][self.mask[b]]):
		self.llhs[b].append( stats.norm(loc = f, scale = (self.dy[b][self.mask[b]])[j] ) )
		self.maxL[b] -= self.llhs[b][-1].pdf((self.y[b][self.mask[b]])[j])

	if kwargs['model'] == 'break+exp':
	    self.flux_model = lambda t,**par : breakoutphase(t,**par) + expansionphase(t,**par)
	    self.parnames = ['t0','a1','a2','a3']
	elif kwargs['model'] == 'break+arnett':
	    self.flux_model = lambda t,**par : breakoutphase(t,**par) + expansionphase_arnett(t,**par)
	    self.parnames = ['t0','a1','a2','a3','a4']
	elif kwargs['model'] == 'breakexp+arnett':
	    self.flux_model = lambda t,**par : breakoutexp(t,**par) + expansionphase_arnett(t,**par)
	    self.parnames = ['t0','a1','a2','a3','a4']
	else:
	    raise ValueError('Model {0[model]:s} unknown!'.format(kwargs))

	self.corr  = []
	self.chi2 = kwargs['chi2']
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
	params:	dict,
		dictionary with parameters of fitting parameters

	Returns 
	-------
	float with likelihood 
	"""

	logL = 0.
	for i,b in enumerate(self.bands):
	    for j,l in enumerate(self.llhs[b]):
		t = (self.t[b][self.mask[b]])[j] # the time for that band and flux
		flux = self.flux_model(t,**params)
		if self.chi2:
		    logL += ((self.y[b][self.mask[b]])[j] - flux) ** 2. / \
				(self.dy[b][self.mask[b]])[j] ** 2.
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

	for i,n in enumerate(self.parnames):
	    fitarg[n] = self.res.x[i]

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

	self.m.migrad(ncall = kwargs['ncall'], precision = kwargs['precision'])
	logging.debug("Migrad minimization finished")
	return 

    @setDefault(passed_kwargs = minuit_def)
    def __call__(self,**kwargs):
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
	    # get the likelihood profile for the 
	    # explosion time t0
	    logging.info('stepping over t0 within {0[scan_bound]:.2f} sigma with {0[t0_steps]:n} steps'.format(
	    			kwargs))
	    self.t0array, self.delta_logL, mr = self.m.mnprofile(vname='t0',
		    bound =kwargs['scan_bound'], 
		    #bound = ( self.m.values['t0'] - self.m.errors['t0'] * kwargs['scan_bound'], 
		#	    self.t_first_data_point), 
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
