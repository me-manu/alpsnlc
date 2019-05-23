import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerLine2D
import copy
import myplot.funcs as mpf

def plotconflegend(ax, colors = [['limegreen', 'gold']],
    labels = ["68% / 95% expected limits"],
    title = None, fontsize = 'medium', loc = 0):

    handles, labels_all = ax.get_legend_handles_labels()
    hmap = dict(zip(handles, [HandlerLine2D(numpoints = 0) for h in handles]))
    handles += [plt.Rectangle((0,0),1,1) for color  in colors]
    hmap.update(dict(zip(
                handles[-1 * len(colors):], 
                [mpf.Handler(color[0],color[1]) for color in colors]
                )))
    for l in labels:
        labels_all.append(l)
    l = ax.legend(handles=handles, labels=labels_all, handler_map=hmap,
        fontsize = fontsize, loc = loc, 
        title = title)
    plt.setp(l.get_title(), fontsize = fontsize,
        multialignment= 'center')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Function that extracts a subset of a colormap.
    """
    if minval is None:
        minval = 0.0
    if maxval is None:
        maxval = 0.0

    name = "%s-trunc-%.2g-%.2g" % (cmap.name, minval, maxval)
    return LinearSegmentedColormap.from_list(
        name, cmap(np.linspace(minval, maxval, n)))

def get_corner_input(model, return_pars = False):
    corner_input = []
    pars = [x for x in model['setup'] if model['setup'][x].get('kind') == 'parameter' and
        'min_value' in model['setup'][x] and 'max_value' in model['setup'][x]]
    for realization in model['realizations']:
        par_vals = realization['parameters']
        var_names = ['$' + ('\\log\\, ' if par_vals[x].get('log') else '') +
                 par_vals[x]['latex'] + '$' for x in par_vals if x in pars and 'fraction' in par_vals[x]]
        par_names = [x for x in par_vals if x in pars and 'fraction' in par_vals[x]]
        corner_input.append([np.log10(par_vals[x]['value']) if
                         par_vals[x].get('log') else par_vals[x]['value'] for x in par_vals
                         if x in pars and 'fraction' in par_vals[x]])
    if return_pars:
        return corner_input, var_names, par_names
    else:
        return corner_input, var_names

class LightcurvePlotter(object):
    """
    Class to plot light curve where each bin has an associated likelihood
    """
    def __init__(self, lc):
        """Initialize the class with a table with light curve results 
        including the likelihood scan"""

        self._lc = copy.deepcopy(lc)

    @property
    def lc(self):
        return self._lc

    @staticmethod
    def set_points(lc, eflux = False, usesed = False, usebin = 0):
        if usesed:

            f = lc['norm'][:,usebin]
            ful = lc['norm_ul'][:,usebin]
            ferr = lc['norm_err'][:,usebin]

            f *= lc['ref_eflux'][:,usebin] if eflux else lc['ref_flux'][:,usebin]
            ful *= lc['ref_eflux'][:,usebin] if eflux else lc['ref_flux'][:,usebin]
            ferr *= lc['ref_eflux'][:,usebin] if eflux else lc['ref_flux'][:,usebin]
        else:
            if eflux:
                k = 'eflux'
            else:
                k = 'flux'
            f = lc[k]
            ferr = lc[k + '_err'] 
            ful = lc[k] + 2. * ferr

        return f,ferr,ful


    @staticmethod
    def get_ylims(lc):
        f,ferr,ful = LightcurvePlotter.set_points(lc)
        fmin = np.log10(np.nanmin(f)) - 0.5
        fmax = np.log10(np.nanmax(ful)) + 0.5
        fdelta = fmax - fmin
        if fdelta < 2.0:
            fmin -= 0.5 * (2.0 - fdelta)
            fmax += 0.5 * (2.0 - fdelta)

        return fmin, fmax

    @staticmethod
    def plot_lnlscan(lc, **kwargs):

        ax = kwargs.pop('ax', plt.gca())
        eflux = kwargs.pop('eflux', False)
        usebin= kwargs.pop('usebin', 0)
        llhcut = kwargs.pop('llhcut', -2.70)
        cmap = kwargs.pop('cmap', 'BuGn')
        cmap_trunc_lo = kwargs.pop('cmap_trunc_lo', None)
        cmap_trunc_hi = kwargs.pop('cmap_trunc_hi', None)
        t0 = kwargs.pop('t0',0.)

        ylim = kwargs.pop('ylim', None)

        if ylim is None:
            fmin, fmax = LightcurvePlotter.get_ylims(lc)
        else:
            fmin, fmax = np.log10(ylim)

        fluxM = np.arange(fmin, fmax, 0.01)
        fbins = len(fluxM)
        llhMatrix = np.zeros((len(lc['tmin']), fbins))

        cmap = copy.deepcopy(plt.cm.get_cmap(cmap))
        #cmap.set_under("w")
        #cmap.set_over("w")
        if cmap_trunc_lo is not None or cmap_trunc_hi is not None:
            cmap = truncate_colormap(cmap, cmap_trunc_lo, cmap_trunc_hi, 1024)

        # loop over energy bins
        for i in range(len(lc['tmin'])):
            m = lc['norm_scan'][i,usebin] > 0
            fluxscan = lc['norm_scan'][i,usebin][m]
            fluxscan *= lc['ref_eflux'][i,usebin] if eflux else lc['ref_flux'][i,usebin]
            flux = np.log10(fluxscan)
            logl = lc['dloglike_scan'][i,usebin][m]
            logl -= np.max(logl)
            try:
                fn = interpolate.interp1d(flux, logl, fill_value='extrapolate')
                logli = fn(fluxM)
            except:
                logli = np.interp(fluxM, flux, logl)
            llhMatrix[i, :] = logli

            xedge = [lc['tmin'][i] - t0,lc['tmax'][i] - t0]
            yedge = np.logspace(fmin, fmax, fbins)
            xedge, yedge = np.meshgrid(xedge, yedge)
            im = ax.pcolormesh(xedge, yedge, np.array([logli]).T,
                           vmin=llhcut, vmax=0, cmap=cmap,
                           linewidth=0)


        if kwargs.pop('plotcb', True):
            cb = plt.colorbar(im)
            cb.set_label('Delta LogLikelihood')

        ax.set_ylim(10 ** fmin, 10 ** fmax)
        ax.set_yscale('log')
        ax.set_xlim(lc['tmin'][0] - t0, lc['tmax'][-1] - t0)
        return im

    @staticmethod
    def plot_flux_points(lc, **kwargs):

        ax = kwargs.pop('ax', plt.gca())
        eflux = kwargs.pop('eflux', False)
        usesed = kwargs.pop('usesed', False)
        usebin = kwargs.pop('usebin', 0)
        f,ferr,ful = LightcurvePlotter.set_points(lc, eflux = eflux, usesed = usesed)
        kts = 'ts_sed' if usesed else 'ts'
        t0 = kwargs.pop('t0',0.)

        ul_ts_threshold = kwargs.pop('ul_ts_threshold', 9)

        if usesed:
            m = lc[kts][:,usebin] < ul_ts_threshold
        else:
            m = lc[kts] < ul_ts_threshold
        x = 0.5 * (lc['tmin'] + lc['tmax'])

        delo = x - lc['tmin']
        dehi = lc['tmax'] - x 
        xerr0 = np.vstack((delo[m], dehi[m]))
        xerr1 = np.vstack((delo[~m], dehi[~m]))

        if (~m).sum():
            ax.errorbar(x[~m] - t0, f[~m], xerr=xerr1,
                     yerr=ferr[~m], **kwargs['fluxp'])
        if m.sum():
            ax.errorbar(x[m] - t0, ful[m], xerr=xerr0,
                     yerr=ful[m] * 0.2, uplims=True, **kwargs['fluxulp'])

        ax.set_yscale('log')
        ax.set_xlim(lc['tmin'][0] - t0, lc['tmax'][-1] - t0)
