import argparse
import numpy as np
import logging
from varFSRQ.manage import SrcList
from os import path
from fermiAnalysis.batchfarm.utils import init_logging, load, save
from glob import glob
from astropy.table import Table
from fermiAnalysis import adaptivebinning as ab
import yaml
from snlc.likelihood import GammaRayLogLike, CalcLimits
from fermiAnalysis.utils import myconf2fermipy,met_to_mjd
from fermipy.gtanalysis import GTAnalysis
from time import time
from scipy.integrate import simps

if __name__ == '__main__':
    usage = "usage: %(prog)s --srcconf srcconfig.yaml --conf config.yaml --interval interval --snconf snconf.yaml"

    description = "Calculate expected limit bands for SNALP analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('--srcconf', required = True)
    parser.add_argument('--conf', required = True)
    parser.add_argument('--snconf', required = True)
    parser.add_argument('--snmodel', default = 'default', required = True)
    parser.add_argument('--select_src', required = False, default = 'none')
    parser.add_argument('--interval',
        #choices = ['weekly','daily','orbit','5min','3min'],
        default = 'weekly', help='time interval')
    parser.add_argument('--Mprog', help = "Progenitor mass in solar masses", type = float, default = 10.)
    parser.add_argument('--m_neV_start', help = "min ALP mass in neV", type = float, default = 0.1)
    parser.add_argument('--m_neV_stop', help = "max ALP mass in neV", type = float, default = 10.)
    parser.add_argument('--m_neV_step', help = "steps for ALP mass", type = int, default = 5)
    parser.add_argument('--overwrite', help = "overwrite existing file", type = int, default = 0)
    parser.add_argument('--nsamples', help = "number of samples to build null distribution",
        type = int, default = 10)
    parser.add_argument('--bfield', help = "Assumed B field", default = 'jansson12')
    args = parser.parse_args()
    src = SrcList(args.srcconf, args.conf, interval = args.interval, usehop = False)
    init_logging("INFO", color = True)
    #init_logging("DEBUG", color = True)

    source_names = list(src.srcconf.keys())

    # photon ALP coupling array
    g11 = np.logspace(-2.,1.5,3*20 + 10 + 1)

    # ALP mass array
    m_neV = np.logspace(np.log10(args.m_neV_start),
                np.log10(args.m_neV_stop), args.m_neV_step)
    logging.info("tested ALP masses: {0}".format(m_neV))
    logging.debug("tested ALP coupling: {0}".format(g11))

    with open(args.snconf) as f:
        snconfig = yaml.load(f)

# === second loop to plot the light curves ==================== #
    for isrc,s in enumerate(source_names):
        if not args.select_src.lower() == 'none': 
            if not s.lower() == args.select_src.lower(): continue

        for i in range(len(src.srcconf[s])):

            filename = path.join(src.srcconf[s][i]['fileio']['outdir'],
                'bands_Mprog{0:n}_{1:s}_{2:s}_{3:s}.npz'.format(args.Mprog,
                    path.basename(args.snconf).split('.')[0],
                    args.snmodel,args.bfield))

            if path.isfile(filename) and not args.overwrite:
                logging.info("file {0:s} exists and overwrite is " \
                    "set to false, continueing".format(filename))
                continue

            logging.info("===== {0:s} t{1:03n} ======".format(s,i + 1))
            if 'min' in args.interval:
                logging.error("LC fitting for minute scales not implemented yet")
                continue
            fname = glob(path.join(src.srcconf[s][i]['fileio']['outdir'], 'avgspec_combined.fits'))
            if len(fname):
                t = Table.read(fname[0])
            else:
                logging.warning("No file found in {0:s}".format(
                    path.join(src.srcconf[s][i]['fileio']['outdir'], 'avgspec_combined.fits')))
                continue

            earray = np.logspace(np.log10(src.srcconf[s][i]['selection']['emin']),
                        np.log10(src.srcconf[s][i]['selection']['emax']),
                        8)
            texp_file = path.join(src.srcconf[s][i]['fileio']['outdir'],
                "exposure_{0[emin]:n}-{0[emax]:n}MeV.npz".format(src.srcconf[s][i]['selection']))

            if path.isfile(texp_file):

                npzfile = np.load(texp_file)
                texp, front, back = npzfile['texp'], npzfile['front'], npzfile['back']
                logging.info("loaded exposure from {0:s}".format(texp_file))

            else:
                # get the gta object
                config = myconf2fermipy(src.srcconf[s][i])
                config['fileio']['scratchdir'] = None
                config['fileio']['usescratch'] = False
                #try:
                gta = GTAnalysis(config,logging={'verbosity' : 3})
                #except Exception as e:
                #    logging.error("{0}".format(e))
                #    config['selection']['target'] = None
                #    gta = GTAnalysis(config,logging={'verbosity' : 3})
                sep = gta.roi.sources[0]['offset'] 
                print (config['selection']['target']) 
                #    logging.warning("Source closets to ROI center is {0:.3f} degree away".format(sep))
                #    if sep < 0.1:
                #        config['selection']['target'] = gta.roi.sources[0]['name']
                #        gta.config['selection']['target'] = config['selection']['target']
                #        logging.info("Set target to {0:s}".format(config['selection']['target']))

                #logging.info("Calculating exposure for {0:.1f} MeV".format(energy))
                front, back = [],[]
                for energy in earray:
                    texp, f, b = ab.comp_exposure_phi(gta, energy = energy)
                    front.append(f)
                    back.append(f)

                front = np.array(front)
                back = np.array(back)
                np.savez(texp_file, texp = texp, front = front,
                    back = back, earray = earray)
                logging.info("Saved exposure to {0:s}".format(texp_file))
# read in likelihood vs norm / flux 
# convert to alp coupling using the generate script 
# plot the color bars for each time bin, check the SED plotting script

            outpath = path.join(snconfig['outbase'],s, args.snmodel, 'products')

            walkerfile = path.join(outpath, 'walkers.json')
            if not path.exists(walkerfile):
                logging.warning("*** File {0:s} not found".format(path.join(outpath, 'walkers.json')))
                continue

            try:
                gloglike = GammaRayLogLike(fname[0], walkerfile,
                    m_neV[0], args.Mprog, 
                    bfield = args.bfield, 
                    #spline = dict(k = 2, s = 1e-3, ext = 'extrapolate'),
                    spline = dict(k = 1, s = 0., ext = 'extrapolate'),
                    min_delay = snconfig.get('min_delay',0.) / 24. / 3600.,
                    max_delay = snconfig.get('max_delay',0.) / 24. / 3600.)
            except ValueError as e:
                logging.error("{0}, continuing".format(e))
                raise
                continue

            # multiply posterior with exposure / max(exposure) and integrate
            # this will get you an estimate of the probability to observe the SN
            texp_max = met_to_mjd(texp[1:])
            texp_min = met_to_mjd(texp[:-1])
            texp_cen = 0.5 * (texp_min + texp_max)

            exp = 0.5 * (front + back)
            # weight by spectrum and integrate
            spec = gloglike.snalpflux.AvgALPflux(EMeV = earray, t_sec = 30., g11 = 1.)
            expw = simps(exp.T * spec, earray, axis = 1) / simps(spec, earray)
            expw = 0.5 * (expw[1:] + expw[:-1])
            aeff = expw / np.diff(texp)

            pobs = gloglike.tpost.integrate_pdf(tmjd_max= texp_max,
                        tmjd_min = texp_min,
                        weights=aeff/ aeff.max())
            pobs = pobs.flatten().sum()
            logging.info("Probability to observe {0:s}: {1:.4f}".format(s, pobs))

            cl = CalcLimits(gloglike)
            q = 0.99
            mdata = CalcLimits.calcmask(gloglike, q = q)
            if mdata.sum():
                t0 = time()

                gband = np.zeros((m_neV.size,5))
                logl = np.zeros((m_neV.size, args.nsamples, g11.size))

                logldata = np.zeros((m_neV.size, g11.size))
                glimdata = np.zeros_like(m_neV)

                logldata[0], tsdata, glimdata[0] = CalcLimits.calc_obs_limits(gloglike,g11 = g11)
                logl[0], ts, gband[0], q = cl.build_null_dist_exp_bands(g11, samples = args.nsamples)

                for imass, m in enumerate(m_neV[1:]):
                    logging.info(" ==== Calculating mass {0:n} / {1:n}: {2:.2f} neV".format(
                                    imass + 1, m_neV.size - 1, m))
                    cl.glnl.m_neV = m
                    logl[imass + 1], ts, gband[imass + 1], q = cl.build_null_dist_exp_bands(g11 = g11, samples = args.nsamples)
                    logldata[imass + 1], tsdata, glimdata[imass + 1] = cl.calc_obs_limits(cl.glnl, g11 = g11)

                np.savez(filename, q = q, m_neV = m_neV, Mprog = args.Mprog, snmodel = args.snmodel,
                        ts = ts, gband = gband, pobs = pobs, max_delay = snconfig.get('max_delay',0.),
                        min_delay = snconfig.get('min_delay',0.), logl = logl, g11 = g11,
                        logldata = logldata, tsdata = tsdata, glimdata = glimdata)
                logging.info("saved bands and ts distribution to {0:s}".format(filename))

            else:
                logging.error("No Fermi LAT data points within {0:.3f} percentile of posterior!" \
                    "Skipping band calculation and not saving file".format(q))
