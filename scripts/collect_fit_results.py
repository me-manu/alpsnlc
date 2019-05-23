# ------ imports ---------------------------- #
import yaml
import argparse
import matplotlib
matplotlib.use('Agg')
from haloanalysis.batchfarm import utils,lsf
from glob import glob
from os import path
import os
from astropy.table import Table
import copy
from collections import OrderedDict
from snlc import fitlc
from eblstud.tools.iminuit_fit import pvalue
import logging
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------- #

# ========================================================================== #
# === the script =========================================================== #
# ========================================================================== #

if __name__ == "__main__":
    usage = "usage: %(prog)s"
    description = "Collect the fit results"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('fitsdir', default = None,
			help='Directory containing the output fits files from the fit')
    parser.add_argument('fermipytemplate', default = None,
			help='yaml file containing a template for the fermi analysis')
    parser.add_argument('--plotdir', default = './', help='Directory for output plots')
    parser.add_argument('--yamldir', default = './', help='Directory for output yaml config files')
    parser.add_argument('--overwrite', default = 0, type = int , help='Overwrite joined fits table')
    args = parser.parse_args()

    # initialize 
    utils.init_logging('INFO')

    joined_t = path.join(args.fitsdir, 'joined_fit_table.fits')
    # join the fits files
    if not path.isfile(joined_t) or args.overwrite:
	# get the fit results
	fitsfiles = sorted(glob(path.join(args.fitsdir, 'fit*.fits')))
	names,models = [],[]
	for i,f in enumerate(fitsfiles):
	    m = path.basename(f).split('_')[3].split('.fits')[0]
	    n = path.basename(f).split('_')[2].split('.fits')[0]
	    t = Table.read(f)
	    t['model'] = [m]
	    t['name'] = np.array([n], dtype = 'S10')
	    if t['parnames'].size < 5: # extend arrays
		dts = 5 - t['parnames'].size
		c = {}
		c['parnames'] = [np.concatenate((t['parnames'][0], ['None','None']))]
		c['best_fit_minuit'] = [np.concatenate((t['best_fit_minuit'][0], np.nan*np.ones(dts)))]
		c['best_fit_mcmc'] = [np.vstack((t['best_fit_mcmc'][0], np.ones((dts,3)) * np.nan))]
		for k,v in c.items():
		    t[k] = v

	    if not i:
		tjoin = t
	    else:
		tjoin.add_row(t[0])
	tjoin.write(joined_t, overwrite = args.overwrite)

    # read the joined fits table
    else:
	tjoin = Table.read(joined_t)

    models = np.unique(tjoin['model'])
    names = np.unique(tjoin['name'])

    for n in names:
	# extract position and time span to write to a 
	# fermipy config file
	config = yaml.load(open(args.fermipytemplate))
	msk = (tjoin['name'] == n) 
	config['selection']['ra'] = float(np.round(tjoin['ra'][msk][0],6))
	config['selection']['dec'] = float(np.round(tjoin['dec'][msk][0],6))

	# time conversion
	if n == 'PTF10vgv':
	    t2mjd = lambda t: t - 2400000.5 + 2455453.6446
	elif tjoin['last_ul'][msk][0] > 2400000.5: # we are dealing with JD
	    t2mjd= lambda t: t - 2400000.5
	else:
	    t2mjd= lambda t: t

	met2mjd = lambda met: 54682.65 + (met - 239557414.0) / (86400.)
	mjd2met = lambda mjd: (mjd - 54682.65) * 86400. + 239557414.0

	config['selection']['tmin'] = float(np.round(mjd2met(t2mjd( tjoin['last_ul'][msk][0] )) \
					- 1e4, 6)) # account for possible delay of optical outburst
	config['selection']['tmax'] = float(np.round(mjd2met(t2mjd( tjoin['ddisc'][msk][0] )), 6)) 

	if n.find('PTF') < 0:
	    config['data']['evfile'] = config['data']['evfile'].replace('*',n, 1)
	    config['data']['scfile'] = config['data']['scfile'].replace('*',n, 1)
	    config['data']['evfile'] = config['data']['evfile'].replace('*','ASASSN-' + n, 1)
	    config['data']['scfile'] = config['data']['scfile'].replace('*','ASASSN-' + n, 1)
	else:
	    config['data']['evfile'] = config['data']['evfile'].replace('*',n)
	    config['data']['scfile'] = config['data']['scfile'].replace('*',n)
	config['fileio']['outdir'] = config['fileio']['outdir'].replace('*',n)
	config['fileio']['logfile'] = config['fileio']['logfile'].replace('*',n)
	config['fit_pars']['z'] = float(tjoin['redshift'][msk][0])

	yamlfile = path.join(args.yamldir,'{0:s}_fpy.yaml'.format(n))
	yaml.dump(config, stream = open(yamlfile, 'w'), default_flow_style=False)

	ddisc_mjd = t2mjd(tjoin['ddisc'][msk][0])
	# make posterior plots
	xmin, xmax = 1e9, -1e9
	for i,m in enumerate(models):
	    msk = (tjoin['name'] == n) & (tjoin['model'] == m)


	    t0 = np.percentile(tjoin['posterior_t0'][msk][0], 50.)
	    t1 = np.percentile(tjoin['posterior_t0'][msk][0], 2.7e-3 * 100.)
	    t2 = np.percentile(tjoin['posterior_t0'][msk][0], 100. * (1. - 2.7e-3))

	    plt.axvline(t2mjd(t0) - ddisc_mjd, ls = '-', color = plt.cm.Vega20c(0.2 * i), label = m + ', 50% quantile', lw = 1)
	    plt.axvline(t2mjd(t1) - ddisc_mjd, ls = '--', color = plt.cm.Vega20c(0.2 * i), lw = 1, label = m + ', $3\sigma$ quantile')
	    plt.axvline(t2mjd(t2) - ddisc_mjd, ls = '--', color = plt.cm.Vega20c(0.2 * i), lw = 1)

	    if t2mjd(t1) - ddisc_mjd < xmin:
		xmin = t2mjd(t1) - ddisc_mjd
	    if t2mjd(t2) - ddisc_mjd > xmax:
		xmax = t2mjd(t2) - ddisc_mjd

	for i,m in enumerate(models):
	    msk = (tjoin['name'] == n) & (tjoin['model'] == m)

	    bins = np.linspace(xmin - 0.05, xmax + 0.05, 201)
			    
	    plt.hist(t2mjd(tjoin['posterior_t0'][msk][0]) - ddisc_mjd,
			alpha = 0.5, 
	    		label = m, normed = True, bins = bins)


	plt.gca().set_xlim(xmin - 0.01,xmax + 0.01)
	plt.legend(loc = 0, fontsize = 'x-small', ncol = 1)
	plt.title(n + ', Discovery date: {0:.3f} MJD'.format(ddisc_mjd))
	plt.savefig(path.join(args.plotdir,'posterior_t0_{0:s}.png'.format(n)),
	    format = 'png', dpi = 200)
	plt.close()




