import numpy as np
from numpy.random import multivariate_normal

import argparse
import astropy
import os,  os.path, sys
import chainconsumer
from chainconsumer import ChainConsumer
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.pyplot as plt

names = {
'RAJ': 'RA (J2000)',
'DECJ': 'DEC (J2000)',
'F0' : 'Spin Frequency',
'F1' : 'Spin down',
'PMRA': r'$\mu_{RA}$',
'PMDEC' : r'$\mu_{DEC}$',
'PB' : r'$P_{\rm b}$',
'T0' : r'$T_0 $',
'A1' : r'$x$\,(lt-s)',
'OM' : r'$ \omega$',
'PX': 'PX',
'ECC' : 'e',
'XDOT' : r'$\dot{x}$~( lt-$s~s^{-1}$)',
'EDOT' : r'$\dot{e}$',
'M2' : r'$M_{\rm c}$',
'MTOT': r'$M_{\rm TOT}$',
'PBDOT': r'$\dot{P_{\rm b}}$',
'XPBDOT': r'$\dot{P}^{\rm excess}_{\rm b}$',
'OMDOT': r'$\dot{\omega}$',
'GAMMA': r'$\gamma~(s)$',
'H3': r'$h_3$',
'SINI': 'SINI',
'DM': 'DM',
'DM1': 'DM1',
'DM2': 'DM2',
'DM001': 'DM1',
'DM002': 'DM2',
'STIG' :r"$\zeta$",
'GLPH_1': r'Glitch Phase',
'GLF0_1': r'GlitchF0\_1',
'GLF1_1' : r'GlitchF1\_1',
'GLF0_2': r'GlitchF0\_2',
'GLF1_2' : r'GlitchF1\_2',
'RedAmp' : r'$A_{red}$',
'RedSlope': r'$\alpha_{red}$',
'DMAmp' : r'$A_{DM}$',
'DMSlope': r'$\alpha_{DM}$',
'EFAC1': 'EFAC1',
'EFAC2': 'EFAC2',
'EFAC3': 'EFAC3',
'EFAC4': 'EFAC4',
'EFAC5': 'EFAC5',
'EFAC6': 'EFAC6',
'EFAC7': 'EFAC7',
'EFAC8': 'EFAC8',
'EQUAD1':'EQUAD1',
'EQUAD2':'EQUAD2',
'EQUAD3':'EQUAD3',
'EQUAD4':'EQUAD4',
'EQUAD5':'EQUAD5',
'EQUAD6':'EQUAD6',
'EQUAD7':'EQUAD7',
'EQUAD8':'EQUAD8',
'Likelihood' : 'Likelihood'
}

#rc('text', usetex=False)
#rc('text.latex', preamble='\\\\usepackage{amsmath},\\\\usepackage{amssymb},\\\\usepackage{bm}')

rcParams["axes.labelpad"] = 6
rcParams["axes.formatter.limits"] = [-3,3]
rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot TN output posterior distributions')

parser.add_argument('-p','--plot', help='get the subset of params to plot', default='', nargs='?')
parser.add_argument('-d','--device', help='say live or give a file name',  default='live')
parser.add_argument('-s', '--rescale', action='store_true')
parser.add_argument('-r','--results', help='directory of results', default='')
parser.add_argument('-f','--fig_size', help='PAGE OR COLUMN OR GROW', default='GROW')
parser.add_argument('-n','--psr_name', help='J or B name of psr', default='', required=True)
parser.add_argument('-l','--use_live', help='Use phys live points', action="store_true")
parser.add_argument('-x','--print_summary', help='print summary table', action="store_true")
args = parser.parse_args()




results_dir = str(args.results)
prefix=args.psr_name


dir_contents=os.listdir(results_dir)
if len(dir_contents) == 0:
    print("no files in directory", args.results)
    sys.exit(1)

else:

    # Get all the parameter names
    param_names_file = str(os.path.join(results_dir,prefix+'-.paramnames'))
    col_names = [x.split()[0] for x in open(param_names_file).read().splitlines()]
    col_names.append("Likelihood")

    arg_cols = [i for i, e in enumerate(col_names)]

    #Shortlist requested parameters
    if(args.plot != ""):
        required = args.plot.strip().replace(","," ").split()
        print("shortlisting requested parameters:", required)
        arg_cols = [i for i, e in enumerate(col_names) if e in required]
        print(col_names)




    chain_file = str(os.path.join(results_dir,prefix+'-post_equal_weights.dat'))

    #choose between post_equal_weights and phys_live.points
    if(not os.path.exists(chain_file) or args.use_live):
        chain_file = str(os.path.join(results_dir,prefix+'-phys_live.points'))

        if(not os.path.exists(chain_file)):
            print("No post_equal_weights.dat or phys_live.points files found. Aborting.")
            sys.exit(1)

        else:
            if( not args.use_live):
                print("No st_equal_weights.dat file. Using phys-live points instead. The run is probably still going on.")

            else:
                print("Proceeding with using phys-live.points")

    data = np.loadtxt(chain_file)
    mean = [0] * len(col_names)
    sigma = [1] * len(col_names)


    if args.rescale:
        scaling_file  = str(os.path.join(results_dir,prefix+'-T2scaling.txt'))

        if not os.path.exists(scaling_file):
            print("no scaling file. Are you sure the run is over?")
            sys.exit(1)


        for line in open(scaling_file, 'r').read().splitlines():
            chunks = line.strip().split(" ")
            name = chunks[0]
            m = 0
            s = 1
            if name in col_names:
                m = float(chunks[-2])
                s = float(chunks[-1])

                index = col_names.index(name)

                mean[index] = m
                sigma[index] = s


            else:
                print("skipping unused column: ",name)

    print ("means:", mean)
    print ("Sigmas: ", sigma)
    print (arg_cols)      

    #rescale data accordingly
    x = []
    for i in arg_cols:
       x.append( mean[i] + sigma[i] *  data[:,i])
    col_names = [col_names[i] for i in arg_cols]
    x = np.transpose(np.array(x))

    cc = ChainConsumer().add_chain(x,parameters= col_names,walkers = np.shape(x)[0],num_eff_data_points=np.shape(x)[0],num_free_params =np.shape(x)[1], name = args.psr_name)

    if(args.print_summary):
        analysis_summary = cc.analysis.get_summary()
        keylist = analysis_summary.keys()
        for s in keylist:
            print(s,analysis_summary[s])

        print(cc.analysis.get_correlation_table(chain=0,caption='Parameter Correlations', label='tab:parameter_correlations'))
        print(c.analysis.get_latex_table())

    params = col_names
    clr = "#5F9EA0" #cadetblue
    #clr = "#6199AE" #steel blue
    cc.configure(max_ticks=3,colors=clr, tick_font_size=14, label_font_size=12,spacing=1.0,diagonal_tick_labels=True,
        contour_labels='confidence', contour_label_font_size=14, shade_gradient=[3.0],sigmas=[1,2,3], shade_alpha=0.6, linewidths=1.5, summary=False,sigma2d=True)
    fig = cc.plotter.plot(figsize=args.fig_size,legend=False, filename=args.device)
    axes = np.array(fig.axes).reshape(len(params), len(params))
    for ax in axes[-1, :]:
        offset = ax.get_xaxis().get_offset_text()
        print("{0} {1}".format(ax.get_xlabel(), "[{0}]".format(offset.get_text())))
        ax.set_xlabel("{0} \n {1}".format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
        offset.set_visible(False)
    for ax in axes[:, 0]:
        offset = ax.get_yaxis().get_offset_text()
        ax.set_ylabel("{0} \n {1}".format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
        offset.set_visible(False)
    fig.set_size_inches(2* fig.get_size_inches())
    fig.savefig(args.device)        




