import argparse
import numpy as np
import logging
from varFSRQ.manage import SrcList
from os import path
from fermiAnalysis.batchfarm.utils import init_logging, load, save
from fermiAnalysis.batchfarm import lsf
from glob import glob
from astropy.table import Table
import yaml
from snlc.likelihood import GammaRayLogLike
from fermiAnalysis.utils import myconf2fermipy, mjd_to_met


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
    parser.add_argument('--seed', help = "random seed", type = int)
    parser.add_argument('--bfield', help = "Assumed B field", default = 'jansson12')
    # kwargs for lsf batchfarm
    parser.add_argument('--dry', default = 0, type = int)
    parser.add_argument('--time', default = '09:59',help='Max time for lsf cluster job')
    parser.add_argument('--concurrent', default = 0,help='number of max simultaneous jobs', type=int)
    parser.add_argument('--maxjobs', default = 500,help='number of max simultaneous running jobs', type=int)
    parser.add_argument('--nmaxjob', help='number of max job id when resubmitting job', type=int)
    parser.add_argument('--nminjob', help='number of min job id when resubmitting job', type=int)
    parser.add_argument('--select_src', default = 'none',help='only run analysis on selected source')
    parser.add_argument('--sleep', default = 10,help='seconds to sleep between job submissions', type=int)
    parser.add_argument('--n', default = 1,help='number of reserved cores', type=int)
    parser.add_argument('--span', default = 'span[ptile=1]',help='spanning of jobs')
    parser.add_argument('--overwrite', default = 0,help='overwrite existing single files', type=int)
    args = parser.parse_args()
    kwargs = {}
    kwargs['dry'] = args.dry
    kwargs['time'] = args.time
    kwargs['concurrent'] = args.concurrent
    kwargs['sleep'] = args.sleep
    kwargs['max_rjobs'] = args.maxjobs
    kwargs['nminjob'] = args.nminjob
    kwargs['nmaxjob'] = args.nmaxjob
    if args.n > 1:
        kwargs['span'] = args.span
        kwargs['n'] = args.n

    src = SrcList(args.srcconf, args.conf, interval = args.interval, usehop = False)
    init_logging("INFO", color = True)

    source_names = list(src.srcconf.keys())

    # photon ALP coupling array
    g11 = np.logspace(-2.,1.5,3*20 + 10 + 1)

    # ALP mass array
    m_neV = np.logspace(np.log10(args.m_neV_start),
                np.log10(args.m_neV_stop), args.m_neV_step)

    logging.info("tested ALP masses: {0}".format(m_neV))
    logging.debug("tested ALP coupling: {0}".format(g11))

    # script to do coverage test
    script = None
    # TODO: change to right name

    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.snconf) as f:
        snconfig = yaml.load(f)

# === second loop to plot the light curves ==================== #
    for isrc,s in enumerate(source_names):
        if not args.select_src.lower() == 'none': 
            if not s.lower() == args.select_src.lower(): continue

        for i in range(len(src.srcconf[s])):

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
                        8 * 10)

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

            # simulate explosion times
            size = 100
            texp_sim, binnum = gloglike.simulate_texp(size = size, apply_obs_times=True)

            # even bins are within light curve, odd bins outside
            # TODO: check if right bins are filled only
            1. / 0.

            # calculate gamma-ray spectra for one coupling
            # which will be tested in coverage test
            # spectrum will scale with gag^4
            refflux = np.zeros(size)
            for j in range(size):
                # TODO this won't work since we have also the bin gaps
                # TODO in binnum, need to think about this more
                dt = gloglike.tmax[binnum[j]] - gloglike.tmin[binnum[j]]
                # TODO: check right units of dt

                refflux[j] = gloglike.snalpflux.AvgALPflux(EMeV=earray,
                               t_sec = dt * 3600. * 24, g11 = 1.)


                filefunction = path.join(src.srcconf[s][i]['fileio']['outdir'],
                                         "spec_Mprog{0:.0f}_g{1:.1f}_m{2:.1f}_dt{3:.1f}_{4:s}.dat".format(
                                         args.Mprog, 1., m_neV[0], dt, args.bfield))

                np.savetxt(filefunction, np.array([earray, refflux[j]]).T, fmt = "%20.5e")
                # TODO: define MJD to MET
                src.srcconf[s][i]['config']['selection']['tmin'] = mjd_to_met(gloglike.tmin[binnum[j]])
                src.srcconf[s][i]['config']['selection']['tmax'] = mjd_to_met(gloglike.tmax[binnum[j]])

                # add new spectrum to model
                src.srcconf[s][i]['model']['sources'][0]['Normalization'] = 1.
                src.srcconf[s][i]['model']['sources'][0]['FileFunction'] = filefunction

                # provide the g11 array
                src.srcconf[s][i]['fit_pars']['garray'] = g11
                src.srcconf[s][i]['fit_pars']['seed'] = j
                src.srcconf[s][i]['fit_pars']['binnum'] = binnum[j]

                # submit to cluster
                # first check if files are present
                missing = []
                njobs = 1

                outfile = "sim_orbit_{0:05n}.fits".format(j + 1)

                if not len(glob(path.join(src.srcconf[src][i]['fileio']['outdir'], outfile))):
                    missing.append(1)
                else:
                    missing = utils.missing_files(path.join(src.srcconf[src][i]['fileio']['outdir'], outfile),
                        njobs, folder = True, split = '')

                kwargs = lsf.lsfDefaults
                kwargs['nolog'] = False
                kwargs['jname'] = "{0:s}{1:03n}".format(s[-4:], j)
                kwargs['logdir'] = path.join(src.srcconf[src][i]['log'], kwargs['jname'])
                kwargs['tmpdir'] = path.join(src.srcconf[src][i]['tmp'], kwargs['jname'])
                kwargs['log'] = path.join(kwargs['logdir'], '{0:s}.out'.format(kwargs['jname']))
                kwargs['err'] = path.join(kwargs['logdir'], '{0:s}.err'.format(kwargs['jname']))

                if len(missing) < njobs and not overwrite:
                    no_events = utils.parse_logfiles(kwargs['logdir'], "***", nostring='glibc', filenames='err.*')
                    if len(no_events):
                        for n in no_events:
                            if n in missing:
                                missing.remove(n)
                    if type(missing) == list and \
                        (kwargs['nminjob'] is not None or kwargs['nmaxjob'] is not None):

                        nminjob = kwargs['nminjob'] if kwargs['nminjob'] is not None else np.min(missing)
                        nmaxjob = kwargs['nmaxjob'] if kwargs['nmaxjob'] is not None else np.max(missing)
                        missing = np.array(missing)
                        m = (missing >= nminjob) & (missing <= nmaxjob)
                        missing = list(missing[m])
                    njobs = missing
                else:
                    missing = [1]

                if len(missing):
                    lsf.submit_lsf(script,
                               src.srcconf[src][i], option,
                               njobs,
                               **kwargs)
                # TODO write function to replace
                # TODO real light curve bin with simulated one
