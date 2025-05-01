#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Like plotgnu, but each column of the input file(s) is placed on a separate plots.
'''

import copy
import math
import argparse
import os
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt

import plotgnu

# Cycle through these styles, by default
styles = ['k-o', 'r-o', 'g-o', 'b-o', 'y-o']
# Maps --scaling string to pyplot function name
scalings_map = {'linear':'plot', 'logx':'semilogx',  'logy':'semilogy',
                'loglog':'loglog'}
# options are saved in this file.  See also --load.
optionsfile = '.plotgrid'

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    # Initially parse just the --load option.  Then set up the main parser, based on
    # its value.
    init_parser = argparse.ArgumentParser(add_help=False)
    init_parser.add_argument('--load', nargs='?', default=None, const=optionsfile,
                            metavar='LOADFILE',
                            help='Load options from file %(metavar)s. Command-line options'
                            ' override loaded options. If --load is specified with'
                            ' no argument, then %(const)s is used as %(metavar)s, which'
                            ' is also written after each invocation.')

    init_args, remains = init_parser.parse_known_args()

    parser = argparse.ArgumentParser('plotgrid.py',
                                    description='Plots gnuplot files.',
                                    parents=[init_parser])

    nargs = '+'
    if init_args.load:
        nargs = '*'
    parser.add_argument('file', nargs=nargs,
                        help='Gnuplot file(s). Optional if --load is specified.')
    parser.add_argument('--abscissa', type=int, default=0,
                        help='Column in each file to plot as abscissa.'
                        ' (default: %(default)s)')
    parser.add_argument('--xlabel', type=str,
                        help='x-axis label.'
                        ' Default is no label.')
    parser.add_argument('--ylabel', nargs='*', type=str,
                        help='y-axis titles.'
                        ' For each plot, labels will cycle through the those specified.'
                        ' Default is no label.')
    parser.add_argument('--figsize', nargs=2, default=[8,8], type=float,
                        metavar=('XSIZE', 'YSIZE'),
                        help='Figure size. (default: %(default)s)')
    parser.add_argument('--fontsize', default=12, type=int,
                        help='Font size.'
                        ' (default: %(default)s)')
    parser.add_argument('--grid', default=True, action='store_true',
                        help='If true, add grid to each plot. '
                        '(default: %(default)s)')
    parser.add_argument('--no-grid', dest='grid',
                        action='store_false', help='Opposite of --grid')
    parser.add_argument('-l', '--label', nargs='*',
                        help='Label strings for each file.'
                        ' For each column (ordinate), labels will cycle through the those specified.'
                        ' The default is to use the file name.')
    parser.add_argument('--legend', default=True, action='store_true',
                        help='If true, add a legend to the figure. '
                        '(default: %(default)s)')
    parser.add_argument('--legend_loc', default='upper right',
                        help='Location of legend on figure.'
                        ' See matlib documentation for values.'
                        ' (default: %(default)s)')
    parser.add_argument('--no-legend', dest='legend',
                        action='store_false', help='Opposite of --legend')
    parser.add_argument('--linewidth', nargs='*', default=[2],
                        help='Line width for each line that is plotted.'
                        ' For each column (ordinate), widths will cycle through those specified.'
                        ' (default: %(default)s)')
    parser.add_argument('--markersize', default=4, type=int,
                        help='Marker (symbol) size.'
                        ' (default: %(default)s)')
    parser.add_argument('--numcols', default=1, type=int,
                        help='Number of columns of plots.  If the number of plots exceeds numcols * numrows,'
                        ' then numcols is increased to accomodate the number of plots.'
                        ' (default: %(default)s)')
    parser.add_argument('--numrows', default=1, type=int,
                        help='Number of rows of plots.'
                        ' (default: %(default)s)')
    parser.add_argument('-o', '--output', nargs='?', default=None,
                        const='plotgrid.png', metavar='OUTPUTFILE',
                        help='Send grahics output to file %(metavar)s, with no screen '
                        'output. If --output is specified with no argument, '
                        'then %(const)s is used as %(metavar)s.')
    parser.add_argument('--ordinate', type=int, nargs='*', action='append',
                        help='Columns in each gnuplot file to plot as ordinate.'
                        ' Each column is plotted on a separate plot.'
                        ' If not specified, then all columns are plotted.'
                        ' (default: %(default)s)')
    parser.add_argument('--scaling', nargs='*', choices=scalings_map.keys(), default=['linear'],
                        help='Scaling type.'
                        ' For each column (ordinate), scaling will cycle through those specified.'
                        ' (default: %(default)s)')
    parser.add_argument('--style', nargs='*', default=styles,
                        help='Plot style for each line.'
                        ' For each column (ordinate), style will cycle through those specified.'
                        ' (default: %(default)s)')
    parser.add_argument('-t', '--title', default=None, type=str,
                        help='Plot main title. The default is to concatanate '
                        'the file labels.')
    parser.add_argument('--xlim', nargs='*', type=float,
                        metavar='XMIN XMAX [XSTEP]',
                        help='x-axis limits. Default is autoscale.')
    parser.add_argument('--ylim', nargs='*', type=float, action='append',
                        metavar='YMIN YMAX [YSTEP]',
                        help='y-axis limits. Default is autoscale.'
                        ' May be specified multiples times, for each ordinate.'
                        ' If no arguments are given, then autoscale will be used.'
                        ' For each column (ordinate), limits will cycle through those specified.')
    parser.add_argument('--ylabelrotation', default='vertical', choices=['vertical, horizontal'],
                        help='Rotation of all y-axis labels. (default: %(default)s)')

    if init_args.load:
        with open(init_args.load, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            saved_files = t_args.file[:]
            args = parser.parse_args(remains, namespace=t_args)
            if len(args.file) == 0:
                args.file = saved_files
    else:
        args = parser.parse_args(remains)

    if len(args.file) == 0:
        parser.error('Must supply FILE arguments on command line or in the --load file.')

    if args.label is None:
        labels = ['_']
    else:
        labels = args.label

    scalings = args.scaling
    if scalings is None:
        scalings = ['linear']

    if args.style is not None:
        styles = args.style

    font = {'size':args.fontsize}
    plt.rc('font', **font)

    fig = None
    iabs = args.abscissa
    factory = plotgnu.Gnuplot_Data_Factory()
    data_list = []
    for fn in args.file:
        data_list.append(factory(fn))
    istyles = itertools.cycle(styles)
    ilinewidths = itertools.cycle(args.linewidth)
    ilabels = itertools.cycle(labels)
    ordinates = args.ordinate
    num_ordinates = None
    for i, gdata in enumerate(data_list):
        filename = gdata.filename
        style = next(istyles)
        linewidth = next(ilinewidths)
        if i == 0:
            basename = os.path.basename(filename)
        elif os.path.isdir(filename):
            filename = os.path.join(filename, basename)
        if args.label is None:
            label = filename
        else:
            label = next(ilabels)
        for curve in gdata.curves:
            for iseg, seg in enumerate(curve.segments):
                ncolumns = np.shape(seg)[1]
                if num_ordinates is None:
                    # If the ordinates were not specified, plot all the columns
                    if ordinates is None:
                        ncolumns = np.shape(seg)[1]
                        ordinates = list(range(1, ncolumns))
                    num_ordinates = len(ordinates)
                maxcol = np.max(ordinates)
                if maxcol > ncolumns-1:
                    mesg = f'File {filename} has {ncolumns} columns; specified ordinate index {maxcol} > {ncolumns-1}'
                    if i > 0:
                        mesg += '\nAll files must have the equal or more columns than the first file specified.'
                    raise ValueError(mesg)

                if fig is None:
                    n = args.numcols * args.numrows
                    if n < num_ordinates:
                        args.numcols = math.ceil(num_ordinates / args.numrows)
                    fig, ax = plt.subplots(args.numrows, args.numcols, sharex=True, figsize=args.figsize,
                                           squeeze=False)

                iscaling = itertools.cycle(scalings)
                for iax, iord in enumerate(ordinates):
                    row = iax // args.numcols
                    col = iax - row * args.numcols
                    s = next(iscaling)
                    scaling = getattr(ax[row, col], scalings_map[s])

                    # Put only the first axis in the legend
                    if iax == 0:
                        seg_label = label
                    else:
                        seg_label = ''

                    # Add this segment to the plot
                    scaling(seg[:,iabs], seg[:,iord], style, label=seg_label,
                                linewidth=linewidth,
                                markersize=args.markersize, fillstyle='none')

    # Dump the command-line options
    with open(optionsfile, 'wt') as f:
        json.dump(vars(args), f, indent=4)

    # Set the options for each plot
    if args.ylabel is None:
        ylabels = itertools.cycle([None])
    else:
        ylabels = itertools.cycle(args.ylabel)
    if args.ylim is None:
        ylims = itertools.cycle([[]])
    else:
        ylims = itertools.cycle(args.ylim)
    for i in range(num_ordinates):
        row = i // args.numcols
        col = i - row * args.numcols
        axis = ax[row, col]
        gs = axis.get_subplotspec()
        plotopts = copy.deepcopy(args)
        plotopts.legend = False
        ylabel = next(ylabels)
        ylim = next(ylims)
        if gs.is_last_row():
            xlabel = args.xlabel
        else:
            xlabel = None
        plotopts.axislabels = [xlabel, ylabel]
        if len(ylim) == 0:
            plotopts.ylim = None
        else:
            plotopts.ylim = ylim
        plotopts.title = None
        plotgnu.set_plot_opts(plotopts, axis)
    # turn off unused plots in the plot grid
    emptycols = args.numrows * args.numcols - num_ordinates
    for i in range(args.numcols - emptycols, args.numcols):
        ax[args.numrows-1, i].axis('off')
    # add the legend
    if args.legend:
        fig.legend(draggable=True, loc=args.legend_loc, framealpha=1,
                   handlelength=4.0)
    plotgnu.finalize(args, fig)

