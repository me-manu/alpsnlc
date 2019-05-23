"""
Class to read a SN light curve from json file 
obtained from sne.space
"""

__version__ = 0.1
__author__ = "Manuel Meyer"

# ----- Imports ---------- #
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from myplot.funcs import *
# ------------------------ #

mag2flux = lambda mag,**ref : ref['Iref'] * np.power(10.,(ref['mref'] - mag) / 2.5)

class SNDataPoint(object):
    """
    Class for SN light curve data point
    """
    def __init__(self):
	"""
	Init the light curve data point class

	Parameters
	----------
	datapoint: `dict`
	dict with photometry data of one telescope from json file
	"""
	self.t = []
	self.dt = []
	self.m = []
	self.dm = []
	return

    def makeNpArray(self):
	"""
	Convert data to numpy arrays
	"""
	self.t = np.array(self.t)
	self.dt = np.array(self.dt)
	self.m = np.array(self.m)
	self.dm = np.array(self.dm)
	return

class SNLC(object):
    def __init__(self, jsonfile):
	"""
	Initiate light curve class 

	Parameters
	----------
	jsonfile: `string`
	full path to SN info file obtained from http://sne.space
	"""
	sn = json.load(open(jsonfile)) # first key is always the SN name
	self.__dict__.update(sn[sn.keys()[0]])

	# get the data
	self.dat = {}
	for i in self.photometry:
	    try:
		i['telescope']
	    except KeyError:
		i['telescope'] = 'source{0:s}'.format(i['source'])
	    try:
		self.dat[i['telescope']]
	    except KeyError: 
		self.dat[i['telescope']] = {}
    
	    try:
		i['band']
	    except KeyError: 
		logging.warning("{1:s}: No band for data point given, only information is {0}".format(i,self.name))
		i['band'] = '?'
	

	    try:
		self.dat[i['telescope']][i['band']]
	    except KeyError:
		self.dat[i['telescope']][i['band']] = SNDataPoint()

	    try:
		self.dat[i['telescope']][i['band']].t.append(float(i['time']))
	    except KeyError:
		logging.warning("{1:s}: No time for data point given, only information is {0}".format(i,self.name))
		continue

	    try:
		self.dat[i['telescope']][i['band']].dt.append(float(i['e_time']))
	    except KeyError:
		logging.debug("{1:s}: No time error for data point given, {0} setting it to -2".format(i,self.name))
		self.dat[i['telescope']][i['band']].dt.append(-2)

	    self.dat[i['telescope']][i['band']].m.append(float(i['magnitude']))
	    try:
		if not i['upperlimit']:
		    self.dat[i['telescope']][i['band']].dm.append(float(i['e_magnitude']))
		else:
		    self.dat[i['telescope']][i['band']].dm.append(-1)
	    except KeyError as e:
		try: 
		    self.dat[i['telescope']][i['band']].dm.append(float(i['e_magnitude']))
		except KeyError:
		    logging.warning("{1:s}: No mag error for data point {0} given, setting it to -2".format(i,self.name))
		    self.dat[i['telescope']][i['band']].dm.append(-2)
	for tel in self.dat.keys():
	    for band in self.dat[tel].keys():
		self.dat[tel][band].makeNpArray()
	return 

    def convert_date2utc(self,date):
    	"""
	Convert max and discovery date to UTC compatible format
	"""
	try:
	    t = Time(date)
	except ValueError:
	    d = date.split('/')
	    hour = (float(d[-1]) * u.day)
	    d[-1] = int(float(d[-1]))
	    hour = (hour - d[-1] * u.day).to(u.hour)
	    minutes = (hour - int(hour.value) * u.hour).to(u.minute)
	    seconds = (minutes - int(minutes.value) * u.minute).to(u.s)
	    hour = int(hour.value)
	    minutes = int(minutes.value)
	    seconds = seconds.value
	    d[-1] = '{0:n}T{1:n}:{2:n}:{3:.2f}'.format(d[-1],hour,minutes,seconds)
	    date = '-'.join(d)
	    t = Time(date)
	return t

    def get_ndata(self,tmin, tmax):
	"""
	Determine maximum number of data points in some time interval
	and return that number together with telescope and band where 
	that number was found
	"""
	ndata = 0
	band_max = ''
	tel_max = ''
	for tel,dv in self.dat.items():
	    for band in dv.keys():
		mask = dv[band].dm > 0
		if np.sum(mask):
		    n = np.sum((dv[band].t[mask] >= tmin) & (dv[band].t[mask] <= tmax))
		    if n > ndata:
			ndata = n
			band_max = band
			tel_max = tel
	return ndata, band_max, tel_max


    def get_dates_mjd(self,datetype = 'max'):
	"""
	Return all maxdates in MJD. Since maxdates only come as dates, assume 12pm as time
	for conversion
	"""
	times = []
	if datetype == 'max':
	    dates = self.maxdate
	elif datetype == 'discover':
	    dates = self.discoverdate
	else:
	    raise ValueError("datetype must either be 'max' or 'discover'")

	for md in dates:
	    times.append(self.convert_date2utc(md['value']))
	t = Time(times)
	return t.mjd


    def plot_data(self, tmin = 0, tmax = 0, plotmag = True, onetel = 'none', bands = []):
	"""
	Plot the data in all bands and from all sources
	"""
	# determine time for plot
	if not tmin:
	    tmin = 1e7
	    for tel,dv in self.dat.items():
		for band in dv.keys():
		    if np.min(dv[band].t) < tmin:
			tmin = np.min(dv[band].t)
	if not tmax:
	    tmax = 0
	    for tel,dv in self.dat.items():
		for band in dv.keys():
		    if np.max(dv[band].t) > tmax:
			tmax = np.max(dv[band].t)

	# determine number of ticks
	    majorTicks = int((tmax - tmin) / 10.)
	for i in range(4):
	    if int(tmax - tmin) <= 20 * 10**i:
		majorTicks = 10 **i
		break
	if majorTicks % 2:
	    majorTicks += 1
	minorTicks = majorTicks / 4.

	# plot the data
	xlabel = 'MJD'
	if plotmag:
	    ylabel = 'App. Mag.'
	else:
	    #ref = {'mref' : 0., 'Iref' : 1e6 }
	    ref = {'mref' : 0., 'Iref' : 2e7 }
	    ylabel = 'Flux (a.u.)'
	iplot = 1

	# get the max date
	tdmax = self.get_dates_mjd(datetype = 'max')[0]
	tddis = self.get_dates_mjd(datetype = 'discover')[0]

	if onetel.lower() == 'none':
	    fig = plt.figure(figsize = (12 , 2 * len(self.dat) + 1))
	else:
	    fig = plt.figure(figsize = (12 , 3 ))

	for tel,dv in self.dat.items():
	    if onetel.lower() == 'none':
		ax = plt.subplot(len(self.dat),1,iplot)
	    else:
		if not tel == onetel: continue
		ax = plt.subplot(111)

	   # if not plotmag:
	#	ax.set_yscale('log')

	    if iplot == 1:
		plt.title('{0:s}, Type {1[value]:s}'.format(self.name,self.claimedtype[0]), fontsize = 'x-large')
		plt.ylabel(ylabel)

	    plt.axvline(tdmax, ls = '-.', color = '0.', lw = 1.)
	    plt.axvline(tddis, ls = '-', color = 'r', lw = 1.)

	    for band in dv.keys():
		if len(bands):
		    try:
			bands.index(band)
		    except ValueError:
			continue

		if plotmag:
		    y = dv[band].m
		    dy = dv[band].m
		else:
		    y = mag2flux(dv[band].m,**ref)
		    dy = np.array([y - mag2flux(dv[band].m + dv[band].dm,**ref),
			    mag2flux(dv[band].m - dv[band].dm,**ref) - y])

		mask = dv[band].dm > 0
		if np.sum(mask):
		    plt.errorbar(dv[band].t[mask], y[mask],
			yerr = [dy[0][mask],dy[1][mask]],
			label = '{0:s}'.format(band),
			marker = 'o',
			ls = '--',
			lw = 0.5
			)
		mask = dv[band].dm == -1
		if np.sum(mask):
		    plt.errorbar(dv[band].t[mask], y[mask],
			label = 'UL {0:s}'.format(band),
			marker = 'v',
			ls = 'None',
			lw = 0.5
			)
		mask = dv[band].dm == -2
		if np.sum(mask):
		    plt.errorbar(dv[band].t[mask], y[mask],
			label = 'No dm {0:s}'.format(band),
			marker = 'o',
			mfc = 'None',
			mec = '0.',
			ls = 'None',
			lw = 0.5
			)
	    plt.legend(loc = 1, ncol = 2, fontsize = 'medium',
		  title = '{0:s}'.format(tel))
	    v = np.array(plt.axis())
	    v[0] = int(tmin)
	    v[1] = np.round(tmax,0)

	    plt.fill_betweenx([v[2],v[3]], (tddis - 0.5) * np.ones(2), x2 = tddis + 0.5,
				 color = 'r', lw = 1., zorder = -1 , alpha = 0.5)

	    if plotmag:
		plt.axis([v[0],v[1],v[3],v[2]])

		yMajorTicks = int(v[3] - v[2]) / 4.
		if not yMajorTicks: yMajorTicks = 1.
		if yMajorTicks % 2:
		    yMajorTicks += 1

		set_ticks(ax,yMajorTicks,axis = 'y', which = 'major')
		set_ticks(ax,yMajorTicks / 4.,axis = 'y', which = 'minor')
	    else:
		plt.axis([v[0],v[1],v[2],v[3]])
    
	    set_ticks(ax,majorTicks,axis = 'x', which = 'major')
	    set_ticks(ax,minorTicks,axis = 'x', which = 'minor')

	    #set_ticsNxMplot(iplot - 1, ax, 1, len(self.dat), 
		#xlabel, ylabel, size = 'xx-large', disable_y = -1)
    
	    plt.grid(True)
	    iplot += 1
    
	plt.xlabel(xlabel)
	plt.subplots_adjust(hspace = 0, wspace = 0)
	return
