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
# ------------------------------------------- #

# ========================================================================== #
# === the script =========================================================== #
# ========================================================================== #
def get_pinit_from_bestfit(bestfit):
    pinit = copy.deepcopy(bestfit)
    pinit['a1'] = np.log10(pinit['a1'])
    if 'a3' in pinit.keys():
        pinit['a3'] = np.log10(pinit['a3'])
    return pinit

if __name__ == "__main__":
    usage = "usage: %(prog)s"
    description = "Fit optical supernova light curves."
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('--mcmc', default = 1, type=int)
    parser.add_argument('--profile', default = 1, type=int)
    parser.add_argument('-i', required=False, default = 0, 
                        help='Set local or scratch calculation', type=int)
    parser.add_argument('-c', '--conf', required = True)
    parser.add_argument('--lastul', required = False, type = float, default = 7.)
    args = parser.parse_args()

    # initialize 
    utils.init_logging('DEBUG')
    config = yaml.load(open(args.conf))
    tmpdir, job_id = lsf.init_lsf()
    if not job_id:
        job_id = args.i
        tmpdir = os.environ["PWD"]
    logging.info('tmpdir: {0:s}, job_id: {1:n}'.format(tmpdir,job_id))
    os.chdir(tmpdir)    # go to tmp directory
    logging.info('Entering directory {0:s}'.format(tmpdir))
    logging.info('PWD is {0:s}'.format(os.environ["PWD"]))

    sns = Table.read(config['sntable'])
    sn = sns[job_id - 1]

    # find closest upper limit before explosion
    sn['detec'] = sn['detec'] & np.isfinite(sn['time'])
    try:
        iul = np.argmin( np.abs( sn['time'][np.invert(sn['detec']) & \
                        (sn['time'] < sn['ddisc'])] \
                        - sn['ddisc']))
        last_ul = sn['time'][np.invert(sn['detec']) & (sn['time'] < sn['ddisc'])][iul]
    except: # no UL, set to 7 days before detection for fit 
        last_ul = np.min(sn['time'][sn['detec']]) - args.lastul
        logging.info("No upper limit before discovery found, setting to {0:.2f} days".format(args.lastul))

    # remove some unwanted data points
    if sn['name'].find('dkz') >= 0:
        remove = np.min(-sn['mag'][sn['detec']][:10])
        idx = np.where(-sn['mag'] == remove)[0][0]
        sn['detec'][idx] = False

    if sn['name'].find('aoi') >= 0 and config['dt'] < 10.:
        config['dt'] = 10.
        logging.info("set fitting period to 10 days")

    # loop over functions to fit the light curve
    for ifc, func in enumerate(config['funcs']):

        logging.info("Running fit for {0:s} and function {1:s}".format(sn['name'], func))
        # config['dt'] determines the maximum number of days 
        # after explosion for which data points are used in the fit
        snfit = fitlc.FitSNLC(time = sn['time'][sn['detec']],
                      mag = sn['mag'][sn['detec']],
                      dmag = sn['mag_unc'][sn['detec']],
                      tdisc = sn['ddisc'],
                      tmin = sn['ddisc'], 
                      tmax = sn['ddisc'] + config['dt'],
                      name = sn['name'],
                      model = func
                     )
        # add systematic uncertainty
        dy_orig = copy.deepcopy(snfit.dy)
        logging.info("Setting sys error to {0[fsys_init]:.4f}".format(config))
        snfit.dy = np.sqrt(dy_orig**2. + config['fsys_init']**2. * snfit.y**2.)


        # ---- get initial guesses ---------------------------------- #
        maxt = 2.
        da = 0.5 
        a2 = 3.
        y = 2.

        if sn['name'] == '16in' or sn['name'] == '15bd' \
            or sn['name'] == '16at' or sn['name'] == '16jt':
            m = sn['detec'] & (sn['time'] >= sn['ddisc'])
        else:
            m = sn['detec']

        delay = 10.
        if sn['name'].find('dkz') >= 0:
            a2 = 1.5
            dtexp = 0.9
            delay = 5.
        if sn['name'].find('dqy') >= 0:
            a2 = 2
            dtexp = 0.1
        elif sn['name'].find('dkk') >= 0  :
            dtexp = 1.
            a2 = 1.9
            delay = 3.
        elif sn['name'].find('vgv') >= 0  :
            dtexp = 0.1
            a2 = 1.9
            delay = 7.
        elif sn['name'].find('iqb') >= 0  :
            a2 = 1.5
            dtexp = 0.5
        elif sn['name'].find('bvn') >= 0  :
            a2 = 1.
            dtexp = 0.3
            delay = 5.
        elif sn['name'].find('16at') >= 0:
            a2 = 2
            dtexp = 1.
            delay = 7.
        elif sn['name'].find('16in') >= 0:
            a2 = 5
            dtexp = 0.5
            delay = 7.
        elif sn['name'].find('aoi') >= 0  :
            a2 = 1.
            dtexp = 0.2
            maxt = 3.
            delay = 5.
        else:
            dtexp = 0.1


        if func.find('break+') == 0:
            par = fitlc.init_guess_breakout(sn['ddisc'] - dtexp, 
                                sn['time'][m], 
                                sn['mag'][m],
                                a2 = a2, da = da, maxt = maxt
                               )
        else:
            par = fitlc.init_guess_breakout_exp(sn['ddisc'] - dtexp, 
                                sn['time'][m], 
                                sn['mag'][m],
                                a2 = a2, maxt = maxt
                               )
        if func.find('expand') > 0:
            par.update(fitlc.init_guess_expansion(
                                sn['ddisc'] - dtexp,  
                                sn['time'][m], 
                                sn['mag'][m],
                                subtract = lambda t: snfit.breakout(t, **par),
                                delay = delay
                                ))
        elif func.find('arnett') > 0:
            par.update(fitlc.init_guess_arnett(
                                sn['ddisc'] - dtexp,  
                                sn['time'][m], 
                                sn['mag'][m],
                                subtract = lambda t: snfit.breakout(t, **par),
                                delay = delay,
                                y = y
                                ))
        # ----------------------------------------------------------- #
        # perform the fit: preparation
        pinit = pinit = get_pinit_from_bestfit(par)
        limits = OrderedDict({
            't0' : [last_ul, sn['ddisc'] - 0.0001],
            'a1' : [pinit['a1'] - 1., pinit['a1'] + 1.],
            'a2' : [pinit['a2'] / 10.,pinit['a2'] * 30.] if func.find('breakexp') == 0 else \
            [1.,100.]})

        fix = OrderedDict({
            't0' : False,
            'a1' : False,
            'a2' : False,
        })
        if 'a3' in par.keys():
            limits.update({'a3' : [pinit['a3'] - 3., pinit['a3'] + 3.]})
            fix.update({'a3': False})
    
        if 'a4' in par.keys():
            limits.update({'a4' : [1.5, 2.5] if func.find('expand') > 0 else [0.01,30.]})
            fix.update({'a4': False})
        logging.info("pinit: {0}".format(pinit))

        # perform the fit
        # first a chi2 fit
        snfit.chi2 = True
        snfit(pinit = pinit, limits = limits, fix = fix, profile = False) # no profiling
        pinit = get_pinit_from_bestfit(snfit.mvalues)
        # now a log likelihood fit 
        snfit.chi2 = False
        snfit(pinit = pinit, limits = limits, fix = fix, profile = False) # no profiling
        # now decrease sys uncertainty, redo the fit and 
        # profile over the likelihood
        pinit = get_pinit_from_bestfit(snfit.mvalues)

        logging.info("Setting sys error to {0[fsys_final]:.4f}".format(config))
        snfit.dy = np.sqrt(dy_orig**2. + config['fsys_final']**2. * snfit.y**2.)
        snfit(pinit = pinit, limits = limits, fix = fix, profile = args.profile)

        # chi2 value, p value and degrees of freedom
        snfit.chi2 = True
        chi2 =  snfit.returnLikelihood(snfit.mvalues)
        dof = len(snfit.t) - len(pinit)
        snfit.chi2 = False

        logging.info('chi2 / dof: {0:.2f}, pvalue: {1:.3f}'.format(chi2 / dof,
            pvalue(dof,chi2)))
        logging.info('max L: {1:.3e}, theo. max L: {0:.3e}'.format(snfit.maxL,
            snfit.returnLikelihood(snfit.mvalues)))


        # save the output
        values = []
        for n in snfit.parnames:
            values.append(snfit.mvalues[n])
        cols = OrderedDict({
            'name': [sn['name']],
            'maxLtheo': [snfit.maxL],
            'maxL': [snfit.returnLikelihood(snfit.mvalues)],
            'minos_err': [snfit.m_intervals],
            'lnprof': [np.array([snfit.t0array,snfit.delta_logL])],
            'best_fit_minuit' : [values],
            'parnames': [snfit.parnames],
            'chi2_minuit': [chi2],
            'dof': [dof], 
            'pvalue' : [pvalue(dof,chi2)],
            'last_ul' : [last_ul],
            'ddisc' : [sn['ddisc']],
            'ra' : [sn['ra']],
            'dec' : [sn['dec']],
            'redshift' : [sn['z']],
            'type' : [sn['type']],
        })

        # perform an mcmc sampling of the posterior
        if args.mcmc:
            pinit = get_pinit_from_bestfit(snfit.mvalues)
            snfit.mcmc(pinit = pinit, limits = limits,
                control_plots = config['control_plots'],
                threads = config['threads'],
                start_sample =  config['start_sample'],
                plotdir = tmpdir)

            cols.update({
                'best_fit_mcmc': [snfit.bf_mcmc],
                'posterior_t0': [snfit.samples[:,0]]
                })

        # write fit results to disk
        tnew = Table(cols)
        tname = 'fit_result_{0:s}_{1:s}.fits'.format(sn['name'],func)
        tnew.write(path.join(tmpdir,tname), overwrite = True)
        logging.info('Written astropy table to {0:s}'.format(path.join(tmpdir,tname)))
        
        # generate control plots for minuit fit
        if config['control_plots']:
            import matplotlib.pyplot as plt
            logging.info('Generating control plots')

            t = np.linspace(snfit.t[0] - config['dt'], snfit.t[-1] + 2.*config['dt'], 1000)
            for i,p in enumerate([par, snfit.mvalues]):
                ax = plt.subplot(111)
                ax.set_yscale('log')
                if not i:
                    plt.title('Initial parameters')
                else:
                    plt.title('Minuit best fit,' +\
                        '$\chi^2 / \mathrm{{dof}} = {0:.2f}$, $p$-value $=$ {1:.3f}'.format(
                        chi2 / dof, pvalue(dof,chi2)))
                plt.errorbar(snfit.t,snfit.y,yerr = snfit.dy, marker = 'o', ls = 'None')

                plt.plot(t,snfit.expansion(t,**p), label = 'expansion', ls = '-.',
                    color = 'k')
                plt.plot(t,snfit.breakout(t,**p), label = 'breakout', ls = '--',
                    color = 'k')
                plt.plot(t,snfit.flux_model(t,**p), label = 'total', ls = '-')

                plt.axvline(snfit.mvalues['t0'], label = '$t_0$ best fit', ls = '--', color = 'k')
                plt.axvline(last_ul, label = '$t_\mathrm{Last UL}$', ls = ':', color = 'k')

                plt.legend(loc = 0, title = '$t_0 = {0:.4f}$'.format(snfit.mvalues['t0']))

                plt.gca().set_xlim(sn['ddisc'] - 2., sn['ddisc'] + config['dt'] *1.1)
                plt.gca().set_ylim(np.min(snfit.y) / 2.,np.max(snfit.y) * 1.1)
                #plt.gca().set_ylim(1e-1,np.max(snfit.y) + 1)
                plt.savefig(path.join(tmpdir,'fit_{0:s}_{1:s}.png'.format(sn['name'], func)),
                    format = 'png', dpi = 200)
                plt.close()

            plt.plot((snfit.t0array - snfit.mvalues['t0']) * 24. * 3600.,
                     2 * snfit.delta_logL, lw = 3, color = 'k')
            plt.grid(True)
            v = np.array(plt.axis())
            for i in range(3):
                plt.fill_betweenx([v[2],v[3]], 
                    x1 = snfit.m_intervals[i,0] * 3600. * 24., 
                    x2 = snfit.m_intervals[i,1] * 3600. * 24.,
                    color = plt.cm.Blues(0.6 - i * 0.1),
                    alpha = 0.7,
                    zorder = i * -1.)
            plt.title(func)
            plt.annotate('$t_0 = {0[t0]:.2f}$ MJD'.format(snfit.mvalues),
                xy = (0.5,0.7), xycoords = 'axes fraction', ha = 'center',
                size = 'large')
            plt.xlabel('$\Delta t_0$ (s)')
            plt.ylabel('$2\Delta\log\mathcal{L}$')

            plt.savefig(path.join(tmpdir,'lnlt0_{0:s}_{1:s}.png'.format(sn['name'], func)),
                format = 'png', dpi = 200)
            plt.close()
        del snfit

    utils.copy2scratch(glob(path.join(tmpdir, '*.fits')),config['outdir'])
    utils.copy2scratch(glob(path.join(tmpdir, '*.png')),config['plotdir'])
