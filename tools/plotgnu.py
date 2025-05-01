#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Plots gnuplot files.  Specifically, the data format is

x1 y1 [z1 ...]
x2 y2 [z2 ...]
# Comment line
.
.
xN yN [zN ...]

By default, x1...xN are the abscissa, and all columns are plotted.
If there is a single blank line, then it denotes a discontinuity
in the curves. A segment of a curve is between two discontinuities or curve
endpoints.  If there is two consecutive blank lines, then a new
curve is started, the same as if the new curve's data was in a
separate file.

If this file is loaded as a module, it contains helper classes
that may be used for additional matplotlib features, such as subplots and
adding text objects. For example, say you have a file named file.gnu
in 3-column format (x,y,z) with 1 segment (i.e., continuous).
Then the following will plot y vs. x and z vs. x on separate axes:

    # Load the data input a Gnuplot_Data object
    data = Gnuplot_Data_Factory('file.gnu')
    # defaults arguments of get_segment() will return all columns of data
    # for the first segment (the only segment, in this case). Here,
    #    x = segment[:,1]
    #    y = segment[:,2]
    #    z = segment[:,3]
    segment = data.get_segment()
    # Use default options
    Figure_Opts opts
    # Figure layout is one plot on top of another, sharing the x-axis
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=opts.figsize)
    # plot y vs. x in the top plot
    ax[0].plot(segment[:,0], segment[:1], style='k-')
    # plot z vs. x in the bottom plot
    ax[1].plot(segment[:,0], segment[:2], style='r-')
    # Set the options for each plot
    Plot_Opts popts1(axislabels=['x','y'])
    Plot_Opts popts2(axislabels=['x','z'])
    set_plot_opts(popts1, ax[0])
    set_plot_opts(popts2, ax[1])
    # Set the figure options and generate the figure
    finalize(opts, fig)
'''

import argparse
import codecs
import os
import itertools
import json
import shlex
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Cycle through these styles, by default
styles = ['k-o', 'r-o', 'g-o', 'b-o', 'y-o']
# Maps --scaling string to pyplot function name
scalings = {'linear':'plot', 'logx':'semilogx',  'logy':'semilogy',
            'loglog':'loglog'}
# options are saved in this file.  See also --load.
optionsfile = '.plotgnu'

class Plot_Opts:
    '''
    Options for each individual plot. See function
    set_plot_options().
    '''
    def __init__(self, title=None, axislabels=None,
                 ylabelrotation='vertical', grid=False,
                 xlim=None, ylim=None, legend=True,
                 legend_loc='upper right', rylim=None,
                 ryaxislabel=None):
        self.title = title
        self.axislabels = axislabels
        self.ylabelrotation = ylabelrotation
        self.grid = grid
        self.xlim = xlim
        self.ylim = ylim
        self.legend = legend
        self.legend_loc = legend_loc
        self.ryaxislabel = ryaxislabel
        self.rylim = rylim

class Figure_Opts:
    '''
    Options for the overall figure, which may have multiple subplots.
    See function finalize().
    '''
    def __init__(self, title=None, output=None,
                 figsize=[6,6],fontsize=10):
        self.title = title
        self.output = output
        self.figsize = figsize
        self.fontsize = fontsize

def set_plot_opts(opts, ax, axRight=None):
    '''
    Sets the options for a single plot.

    opts: Plot_Opts object.
    ax: matplotlib.axes.Axes object, for the left axis.
    axRight: matplotlib.axes.Axes object, for the right axis (defaut is None).
    '''
    if opts.legend:
        lines, labels = ax.get_legend_handles_labels()
        if axRight is not None:
            lines2, labels2 = axRight.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        # big legend:
        #ax.legend(prop={'size':16}, draggable=True)
        ax.legend(lines, labels, draggable=True, loc=opts.legend_loc, 
                  framealpha=1, handlelength=4.0)
    if opts.title is not None:
        ax.set_title(codecs.decode(opts.title, 'unicode_escape'))
    if opts.axislabels is not None:
        if opts.axislabels[0] is not None:
            ax.set_xlabel(opts.axislabels[0])
        if opts.axislabels[1] is not None:
            ax.set_ylabel(opts.axislabels[1], rotation=opts.ylabelrotation)
    ax.grid(visible=opts.grid)
    if axRight is not None:
        axRight.grid(visible=opts.grid)
        if opts.ryaxislabel is not None:
            axRight.set_ylabel(opts.ryaxislabel, rotation=opts.ylabelrotation)
        if opts.rylim is not None:
            n = len(opts.rylim)
            if n not in (2, 3):
                raise ValueError('--rylim must have 2 or 3 arguments.')
            v = [float(x) for x in opts.rylim]
            axRight.set_ylim(v[:2])
            if len(opts.rylim) == 3:
                axRight.set_yticks(np.arange(v[0], v[1]+0.1*v[2], step=v[2]))
    if opts.xlim is not None:
        n = len(opts.xlim)
        if n not in (2, 3):
            raise ValueError('--xlim must have 2 or 3 arguments.')
        v = [float(x) for x in opts.xlim]
        ax.set_xlim(v[:2])
        if len(opts.xlim) == 3:
            ax.set_xticks(np.arange(v[0], v[1]+0.1*v[2], step=v[2]))
    if opts.ylim is not None:
        n = len(opts.ylim)
        if n not in (2, 3):
            raise ValueError('--ylim must have 2 or 3 arguments.')
        v = [float(x) for x in opts.ylim]
        ax.set_ylim(v[:2])
        if len(opts.ylim) == 3:
            ax.set_yticks(np.arange(v[0], v[1]+0.1*v[2], step=v[2]))

def finalize(opts, fig):
    '''
    Sets the figure options and outputs the figure.

    opts: Figure_Opts object
    fig: matplotlib.fig.Figure object
    '''
    if opts.title is not None:
        plt.suptitle(codecs.decode(opts.title, 'unicode_escape'))
    fig.tight_layout()
    plt.show()
    if opts.output is not None:
        #plt.set_loglevel("debug")
        if opts.output[-4:] == '.pgf':
            matplotlib.use('pgf')
            #print(matplotlib.rcParams.keys())
            matplotlib.rcParams.update({
                'pgf.texsystem': 'pdflatex',
                'font.family': 'serif',
                'font.size' : 11,
                'text.usetex': True,
                'pgf.rcfonts': False,
                'pgf.preamble': r'\usepackage{pgfplots}\usepackage{amsmath}',
            })
            print(f'Using {matplotlib.get_backend()}')
        fig.savefig(opts.output, bbox_inches='tight')
        print(f'Image written to {opts.output}')

class Curve_Set:
    '''
    Container for two or more curves.  A single curve is one or more segments.
    A segment is an array of ordered points, and a curve is discontinuous at
    the segment endpoints. 
    '''
    def __init__(self, name, column_names):
        self.name = name
        self.segments = []
        self.column_names = column_names
        self.num_columns = len(column_names)
    def add_segment(self, segment):
        if np.size(segment, 1) != self.num_columns:
            raise ValueError(f'add_segment: Wrong number of columns')
        self.segments.append(segment)
    def set_column_names(self, names=None):
        if names is None:
            self.column_names = [''] * self.num_columns
        else:
            if len(names) != self.num_columns:
                raise ValueError(f'set_column_names: Wrong number of columns')
            self.column_names = names[:]

class Gnuplot_Data:
    '''
    A collection of Curve_Sets and associated filename
    '''
    def __init__(self, filename):
        self.filename = filename
        self.curves = []
    def get_curve(self, index=0):
        return self.curves[index]
    def get_segment(self, cindex=0, sindex=0):
        return self.curves[cindex].segments[sindex]

class Gnuplot_Data_Factory():
    '''
    Creates a Gnuplot_Data object. This class is called as a function; for example

      gnuplot_data = Gnuplot_Data_Factory(filename)
    
    where the return value is a Gnuplot_Data object.
    '''
    def __call__(self, filename):
        gdata = Gnuplot_Data(filename)
        self.filename = filename
        self.fd = open(filename, 'r')
        self.curve_set_name = filename
        self.column_names = None
        line, num_blank = self._parse_header()
        if num_blank > 0:
            raise ValueError(f'Blank lines not permitted in inital header, file {filename}')
        status = 'segment'
        num_curves = 0
        gdata.curves = [Curve_Set(self.curve_set_name, self.column_names)]
        while status != 'eof':
            curve = gdata.curves[-1]
            d, status, line = self._get_segment(line)
            curve.add_segment(d)
            if status == 'eof':
                break
            elif status == 'blank_line':
                line, num_blank = self._parse_header(line)
                if num_blank == 0:
                    break
                if num_blank == 2:
                    if num_curves == 0 and gdata.curves[0].name == self.curve_set_name:
                        gdata.curves[0].name += '[0]'
                    num_curves += 1
                    num_columns = len(line.split())
                    if len(self.column_names) != num_columns:
                        self.column_names = [''] * num_columns
                    name = f'{self.curve_set_name}[{num_curves}]sucky'
                    gdata.curves.append(Curve_Set(name, self.column_names))
        return gdata
    def _parse_header(self, line=None):
        num_blank = 0
        if line is None:
            line = self.fd.readline()
        while True:
            if len(line) == 0: # then end of file
                return line, 0 # use num_blank = 0 to indicate eof
            line = line.strip()
            if len(line) == 0: # then blank line
                num_blank += 1
            elif line[0] == '#': # then comment line
                s = shlex.split(line)
                if s[0] == '#COLUMNS':
                    if len(s) < 2:
                        raise ValueError(f'Must supply column names: {line}')
                    self.column_names = s[1:]
                elif s[0] == '#NAME':
                    if len(s) != 2:
                        raise ValueError(f'NAME must have single argument (surround in quotes if there are spaces): {line}')
                    self.curve_name = s[1]
            else: # then data line.
                s = line.split()
                num_columns = len(s)
                if self.column_names is None:
                    self.column_names = [''] * num_columns
                elif num_columns != len(self.column_names):
                    print(f'Data line: {line}')
                    print(f'Column names: {self.column_names}')
                    raise ValueError(f'Number of column names not equal to number of data columns')
                break
            line = self.fd.readline()
        return line, num_blank
    def _get_segment(self, line):
        data = []
        status = None
        while True:
            sline = line.strip()
            if len(sline) == 0: # blank line
                status = 'blank_line'
                break
            elif sline[0] == '#':
                continue
            s = sline.split()
            if len(s) != len(self.column_names):
                raise ValueError(f'File {self.filename} has varying number of columns, line "{line}"')
            linefloat = []
            for c in s:
                linefloat.append(float(c))
            data.append(linefloat)
            line = self.fd.readline()
            if len(line) == 0:
                status = 'eof'
                break
        
        return np.array(data), status, line

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

    parser = argparse.ArgumentParser('plotgnu.py',
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
    parser.add_argument('--axislabels', nargs=2,
                        metavar=('XTITLE', 'YTITLE'), type=str,
                        help='Axis titles.'
                        ' Default is no titles.')
    parser.add_argument('--figsize', nargs=2, default=[9,9], type=float,
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
                        ' Lines will cycle through the styles specified.'
                        ' The default is to use the file name + the'
                        ' column index in the file.')
    parser.add_argument('--legend', default=True, action='store_true',
                        help='If true, add a legend to each plot. '
                        '(default: %(default)s)')
    parser.add_argument('--legend_loc', default='upper right',
                        help='Location of legend on plot.'
                        ' See matlib documentation for values.'
                        ' (default: %(default)s)')
    parser.add_argument('--no-legend', dest='legend',
                        action='store_false', help='Opposite of --legend')
    parser.add_argument('--linewidth', nargs='*', default=[2],
                        help='Line width for each line that is plotted.'
                        ' Lines will cycle through the widths specified.'
                        ' (default: %(default)s)')
    parser.add_argument('--markersize', default=4, type=int,
                        help='Marker (symbol) size.'
                        ' (default: %(default)s)')
    parser.add_argument('-o', '--output', nargs='?', default=None,
                        const='plotgnu.png', metavar='OUTPUTFILE',
                        help='Send grahics output to file %(metavar)s, with no screen '
                        'output. If --output is specified with no argument, '
                        'then %(const)s is used as %(metavar)s.')
    parser.add_argument('--ordinate', type=int, nargs='*', action='append',
                        help='Column in each gnuplot file to plot as ordinate.'
                        ' If negative, place on a right ordinate axis.'
                        ' (default: %(default)s)')
    parser.add_argument('--ryaxislabel', type=str,
                        help='Axis title for right axis.'
                        ' Default is no title.')
    parser.add_argument('--rylim', nargs='*', type=float,
                        metavar='YMIN YMAX [YSTEP]',
                        help='y-axis limits for right axis. Default is autoscale.')
    parser.add_argument('--rscaling', choices=scalings.keys(), default='linear',
                        help='Scaling type for right axis.'
                        ' (default: %(default)s)')
    parser.add_argument('--scaling', choices=scalings.keys(), default='linear',
                        help='Scaling type.'
                        ' (default: %(default)s)')
    parser.add_argument('--style', nargs='*', default=styles,
                        help='Plot style for each line.'
                        ' Lines will cycle through the styles specified.'
                        ' (default: %(default)s)')
    parser.add_argument('-t', '--title', default=None, type=str,
                        help='Plot main title. The default is to concatanate '
                        'the file labels.')
    parser.add_argument('--xlim', nargs='*', type=float,
                        metavar='XMIN XMAX [XSTEP]',
                        help='x-axis limits. Default is autoscale.')
    parser.add_argument('--ylim', nargs='*', type=float,
                        metavar='YMIN YMAX [YSTEP]',
                        help='y-axis limits. Default is autoscale.')
    parser.add_argument('--ylabelrotation', default='vertical', choices=['vertical, horizontal'],
                        help='Rotation of y-axis label. (default: %(default)s)')

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

    ordinates = args.ordinate
    if ordinates is None:
        ordinates = [[]] * len(args.file)
    elif len(ordinates) == 1:
        ordinates = args.ordinate * len(args.file)
    elif len(ordinates) != len(args.file):
        parser.error('If --ordinate is repeated, it must be specified for all'
                    ' files.')

    if args.label is not None:
        labels = itertools.cycle(args.label)
    else:
        labels = itertools.cycle(['_'])

    if args.style is not None:
        styles = args.style
    styles = itertools.cycle(styles)
    linewidths = itertools.cycle(args.linewidth)
    font = {'size':args.fontsize}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=args.figsize)

    iabs = args.abscissa
    axRight = None
    factory = Gnuplot_Data_Factory()
    data_list = []
    for fn in args.file:
        data_list.append(factory(fn))
    for i, gdata in enumerate(data_list):
        filename = gdata.filename
        if i == 0:
            basename = os.path.basename(filename)
        elif os.path.isdir(filename):
            filename = os.path.join(filename, basename)
        for curve in gdata.curves:
            for iseg, seg in enumerate(curve.segments):
                # Use the same line styles for all segments
                styles, seg_styles = itertools.tee(styles)
                linewidths, seg_linewidths = itertools.tee(linewidths)
                labels, seg_labels = itertools.tee(labels)

                # If the ordinates were not specified, plot all the columns
                if len(ordinates[i]) == 0:
                    ordinates[i] = list(range(1, np.shape(seg)[1]))
                num_ordinates = len(ordinates[i])

                for iord in ordinates[i]:
                    aiord = iord
                    scaling = getattr(ax, scalings[args.scaling])
                    if iord < 0:
                        aiord = -iord
                        if axRight is None:
                            axRight = ax.twinx()
                        scaling = getattr(axRight, scalings[args.rscaling])
                    style = next(seg_styles)
                    linewidth = next(seg_linewidths)

                    # Put only the first segment in the legend
                    if iseg == 0:
                        if args.label is None:
                            label = f'{filename}[{iord}]'
                        else:
                            label = next(seg_labels)
                    else:
                        label = ''

                    # Add this segment to the plot
                    scaling(seg[:,iabs], seg[:,aiord], style, label=label,
                            linewidth=linewidth,
                            markersize=args.markersize, fillstyle='none')
            labelbase = f'{curve.name}'
            for _ in range(num_ordinates):
                next(styles)
                next(linewidths)
                next(labels)
        # Advance the line styles for the next file (if there is one)
#        for _ in range(num_ordinates):
#            next(styles)
#            next(linewidths)
#            next(labels)

    # Dump the command-line options
    with open(optionsfile, 'wt') as f:
        json.dump(vars(args), f, indent=4)

    set_plot_opts(args, ax, axRight)
    args.title = None # there's only 1 plot, so force only 1 title
    finalize(args, fig)
