import argparse
import numpy as np
import logging
from varFSRQ.manage import SrcList
from os import path
from fermiAnalysis.batchfarm.utils import init_logging, parse_logfiles
from fermiAnalysis.batchfarm import lsf
from glob import glob
from astropy.table import Table
import yaml
import snlc
from snlc.likelihood import GammaRayLogLike
from fermiAnalysis.utils import mjd_to_met, collect_lc_results


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
    parser.add_argument('--m_neV_step', help = "steps for ALP mass", type = int, default = 3)
    parser.add_argument('--seed', help = "random seed", type = int)
    parser.add_argument('--bfield', help = "Assumed B field", default = 'jansson12')
    # kwargs for lsf batchfarm
    parser.add_argument('--dry', default = 0, type = int)
    parser.add_argument('--time', default = '09:59',help='Max time for lsf cluster job')
    parser.add_argument('--concurrent', default = 0,help='number of max simultaneous jobs', type=int)
    parser.add_argument('--maxjobs', default = 500,help='number of max simultaneous running jobs', type=int)
    parser.add_argument('--nmaxjob', help='number of max job id when resubmitting job', type=int)
    parser.add_argument('--nminjob', help='number of min job id when resubmitting job', type=int)
    parser.add_argument('--sleep', default = 10,help='seconds to sleep between job submissions', type=int)
    parser.add_argument('--n', default = 1,help='number of reserved cores', type=int)
    parser.add_argument('--span', default = 'span[ptile=1]',help='spanning of jobs')
    parser.add_argument('--overwrite', default = 0,help='overwrite existing single files', type=int)
    parser.add_argument('--overwrite_combined', default = 0,help='overwrite existing combined files', type=int)
    args = parser.parse_args()
    kwargs = lsf.lsfDefaults
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
    init_logging("DEBUG", color = True)

    source_names = list(src.srcconf.keys())

    # photon ALP coupling array
    #binsperdec = 5
    #g11 = np.logspace(-2.,1.,3*binsperdec + 1 - 3*binsperdec % 2)
    g11 = np.append(np.arange(0.1,1.1,0.1), [2.]) 
    logging.info("testing couplings {0}".format(g11))

    # ALP mass array
    m_neV = np.logspace(np.log10(args.m_neV_start),
                np.log10(args.m_neV_stop), args.m_neV_step)

    logging.info("tested ALP masses: {0}".format(m_neV))
    logging.debug("tested ALP coupling: {0}".format(g11))

    # script to do coverage test
    script = path.join(path.dirname(path.realpath(snlc.__file__)),'scripts','run_lc_sim_coverage.py')
    script_result = path.join(path.dirname(path.realpath(snlc.__file__)),'scripts','calc_analysis_w_inj_signal.py')

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

            earray = np.logspace(1.,6., 100)
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
            texp_sim, binnum = gloglike.simulate_texp(size = size, apply_obs_times=True, seed = args.seed)

            # calculate gamma-ray spectra for one coupling
            # which will be tested in coverage test
            # spectrum will scale with gag^4
            refflux = np.zeros((texp_sim.size, earray.size))
            for j in range(texp_sim.size):
                # find the bin that contains texp_sim
                idx = np.where((gloglike.tmin <= texp_sim[j]) & (gloglike.tmax > texp_sim[j]))[0][0]
                logging.info("Simulating signal in light curve bin {0:n}".format(idx))

                dt = gloglike.tmax[idx] - gloglike.tmin[idx] # in days

                # calculate for one minute and then rescale to dt
                refflux[j] = gloglike.snalpflux.dnde_gray(EMeV=earray,
                               t_sec = 60., g11 = 1., m_neV = m_neV[0], bfield = args.bfield) / (dt * 60. * 24)
                refflux[j][refflux[j] < 1e-40] = np.ones(np.sum(refflux[j] < 1e-40)) * 1e-40


                filefunction = path.join(src.srcconf[s][i]['fileio']['outdir'],
                                         "spec_Mprog{0:.0f}_g{1:.1f}_m{2:.1f}_{3:n}_{4:s}.dat".format(
                                         args.Mprog, 1., m_neV[0], idx, args.bfield))

                np.savetxt(filefunction, np.array([earray, refflux[j]]).T, fmt = "%20.5e")
                src.srcconf[s][i]['selection']['tmin'] = mjd_to_met(gloglike.tmin[idx])
                src.srcconf[s][i]['selection']['tmax'] = mjd_to_met(gloglike.tmax[idx])

                # add new spectrum to model
                src.srcconf[s][i]['model']['sources'][0]['Normalization'] = 1.
                src.srcconf[s][i]['model']['sources'][0]['Spectrum_Filename'] = filefunction

                # provide the g11 array
                src.srcconf[s][i]['fit_pars']['gref'] = 1.
                src.srcconf[s][i]['fit_pars']['garray'] = g11
                src.srcconf[s][i]['fit_pars']['seed'] = j
                src.srcconf[s][i]['fit_pars']['binnum'] = idx 
                src.srcconf[s][i]['configname'] = ''
                src.srcconf[s][i]['fit_pars']['avgspec'] = path.join(src.srcconf[s][i]['fileio']['outdir'],'avgspec*.*')
                if not len(glob(src.srcconf[s][i]['fit_pars']['avgspec'])):
                    raise IOError("{0:s} not found".format(src.srcconf[s][i]['fit_pars']['avgspec']))

                # submit to cluster
                # first check if files are present
                missing = []
                njobs = 1

                outfile = 'coverage_test{0:05n}/cov_{1:s}*.fits'.format(j+1,
                            path.basename(src.srcconf[s][i]['model']['sources'][0]['Spectrum_Filename']).split('.dat')[0])

                logging.info("looking for files in {0:s}".format(path.join(src.srcconf[s][i]['fileio']['outdir'], outfile)))

                filespresent = glob(path.join(src.srcconf[s][i]['fileio']['outdir'], outfile.replace('*', '*[!sed]')))
                logging.info("Found {0:n} files".format(len(filespresent)))

                if len(filespresent) < g11.size:
                    missing.append(1)

                kwargs['nolog'] = False
                kwargs['jname'] = "{0:s}{1:03n}".format(s[-4:], j)
                kwargs['logdir'] = path.join(src.srcconf[s][i]['log'], kwargs['jname'])
                kwargs['tmpdir'] = path.join(src.srcconf[s][i]['tmp'], kwargs['jname'])
                kwargs['log'] = path.join(kwargs['logdir'], '{0:s}.out'.format(kwargs['jname']))
                kwargs['err'] = path.join(kwargs['logdir'], '{0:s}.err'.format(kwargs['jname']))

                if args.overwrite:
                    missing = [1]

                if len(missing):
                    option = ''

                    logging.info("Sending: {0:s} {1:s}".format(script, option))
                    lsf.submit_lsf(script,
                               src.srcconf[s][i], option,
                               njobs,
                               **kwargs)
                else:
                    logging.info("All files present")
                    combfile = path.join(src.srcconf[s][i]['fileio']['outdir'],
                        outfile.split('.fits')[0].replace('*','') + '_combined.fits')

                    if not path.isfile(combfile) or args.overwrite_combined:
                        logging.info("Reading light curve files for coverage test for lc bin {0:n} and seed {1:n} ...".format(idx,j))
                        t = collect_lc_results(path.join(src.srcconf[s][i]['fileio']['outdir'],outfile),
                            stripstring = 'coverage_test', createsedlc = True, hdu = 'CATALOG',
                            sortdir = False, sedname_fixed = False)

                        t['g'] = g11
                        t['g'].unit = '1e-11 GeV^-1'

                        logging.info("Done, writing table to {0:s}".format(
                                path.join(src.srcconf[s][i]['fileio']['outdir'], combfile)))
                        t.write(path.join(src.srcconf[s][i]['fileio']['outdir'], combfile), overwrite = True)

                    # now submit script to gather analysis results
                    missing = []
                    njobs = 1

                    outfile = 'coverage_test{0:05n}/cov_results_{1:s}.npz'.format(j+1,
                            path.basename(src.srcconf[s][i]['model']['sources'][0]['Spectrum_Filename']).split('.dat')[0])

                    logging.info("looking for files in {0:s}".format(path.join(src.srcconf[s][i]['fileio']['outdir'], outfile)))

                    filespresent = glob(path.join(src.srcconf[s][i]['fileio']['outdir'], outfile))
                    logging.info("Found {0:n} file(s)".format(len(filespresent)))

                    if not len(filespresent):
                        missing.append(1)
                    if args.overwrite:
                        missing = [1]

                    kwargs['nolog'] = False
                    kwargs['jname'] = "{0:s}{1:03n}r".format(s[-4:], j+1)
                    kwargs['logdir'] = path.join(src.srcconf[s][i]['log'], kwargs['jname'])
                    kwargs['tmpdir'] = path.join(src.srcconf[s][i]['tmp'], kwargs['jname'])
                    kwargs['log'] = path.join(kwargs['logdir'], '{0:s}.out'.format(kwargs['jname']))
                    kwargs['err'] = path.join(kwargs['logdir'], '{0:s}.err'.format(kwargs['jname']))

                    if len(missing):
                        option_list = ['bfield', 'snconf', 'srcconf', 'conf',
                            'snmodel', 'select_src', 'm_neV_stop', 'm_neV_start', 'm_neV_step', 'Mprog']
                        args.select_src = s
                        option = ' '.join(['--{0:s} {1}'.format(k, path.abspath(str(v)) if path.isfile(str(v)) else v) \
                                for k,v in vars(args).items() if k in option_list])
                        option += ' -i {0:n}'.format(j + 1)

                        logging.info("Sending: {0:s} {1:s}".format(script_result, option))
                        lsf.submit_lsf(script_result,
                                   src.srcconf[s][i], option,
                                   njobs,
                                   **kwargs)
