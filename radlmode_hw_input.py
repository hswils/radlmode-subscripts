from radlmode_hw import *

''' 
RADLMODE INPUT SCRIPT

An input file to be used with radlmode main script. To scan a variable, enter it with starting and
ending values as [start, end] and use simple_plot function.
    
    numtrials:  number of trials you want to scan over using simple_plot, enter 1 if using a different built in plotting function
    Rmaj:       major radius in m
    A:          aspect ratio
    delta:      triangularity, appropriate range is -0.8 to 0.8
    kappa:      elongation, appropriate range is 1 to 3
    xi:         plasma "squareness," appropriate range is -0.5 to 0.65
    B0:         magnetic field at plasma axis in T
    Ip:         plasma current in MA
    fgr:        Greenwald fraction
    fuel:       fuel type used, always enter as list, choose up to three: ['DT','DD','DHe3']
    impurity:   impurity type used, always enter as list, choose up to six: ['pure','ash','Ar','Kr','Xe','Ne']
    scalinglaw: scaling law used, always enter as list, choose up to three: ['L89','98Y2','ISS04']
    T0:         plasma temperature in keV
    alpha_T:    temperature scaling factor
    alpha_n:    density scaling factor
    alpha_imp:  impurity density scaling factor
    H:          H factor
    f_ash:      fraction of ash in plasma
    f_LH:       fraction of LH power threshold    
    
    
    Built-in Plotting Functions:
    --------------------------
     simple_plot: Scans over all variables entered as a range in this input file, plots their output.
     If more than 1 fuel/impurity/scaling law is inputed, will plot a line for every combination.
        > plot_cs: Plots plasma cross sections. For circular cs, set delta = 0, kappa = 1, xi = 0.
        > contourplot: Must scan exactly two variables, plots one on the horizontal axis and one on the
                       vertical axis while making a contour of "zaxis" input. If used with multiple 
                       fuels/impurities/scaling laws, will only evaluate for the first in each list.
            
    plotprofiles: Does not work if scanning variables.
        > plotprof: Plots normalized plasma radius against density, temperature, and impurity profiles. Prints
                    Prints power outputs on plot.
        > plotprad: Plots normalized plasma radius against Prad, prints impurity fraction on plot.
        > plotlhscan: Makes three plots: f_LH versus Pfus, Pext, and Psol for each impurity
        > plott0scan: Plots T0 agains Pfus and f_imp for each impurity.
        > plotDHe3scan: Plots H89 versus Pfus and Ip versus Pfus. 
        
    arc_calc_popcon_lmode: Does not work if scanning variables.
        > T0_range: range over which you want to evaluate popcon
        > verbose: if 'True,' prints plot.
     
    '''

'''Default geometry (ARC-like)'''

numtrials = 4        

Rmaj = [2,3.7,6]
A = 3.7/1.14          
delta = 0.54      
kappa = 1.75                 
xi = 0.097
B0 = 12.16                   
Ip = 17.4                 
fgr = 0.9                    
fuel = ['DT']                 
impurity = ['pure']
scalinglaw = ['ISS04']          
T0 = 21
alpha_T = 1.5 
alpha_n = 1.5  
alpha_imp = 1.5
H = 1.15     
f_ash = 0.02
f_LH = 0.5             

xaxis = 'Rmaj'                
zaxis = 'n20'         

 # Choose zaxis string from below:
 # output_dict = {'Pfus/volume':ps.Pfus/ps.vol,'Pfus':ps.Pfus,'Palpha':ps.Palpha,'P_LH':ps.P_LH,'Pext':ps.Pext,'Psol':ps.Psol,'Prad':ps.Prad,'Pohm':ps.Pohm,'Ploss':ps.Ploss, \
 #                           'qstar':ps.qstar,'q95':ps.q95,'volume':ps.vol,'surface':ps.surf,'n_Gr':ps.n_Gr,'n20':ps.n20,'tau_E':ps.tau_E,'f_imp':ps.f_imp,'beta_n':ps.beta_n,'Qp':ps.Qp}

plot_cs = True
contourplot = False   

plotprof = False  
plotprad = False 
plotlhscan = False 
plott0scan = False 
plotDHe3scan = False 

T0_range = [2,30]
verbose = False

        


# Nothing needs edited below this line---------------------------------------------------------------

input_dict = {'Rmaj':Rmaj,'A':A,'delta':delta,'kappa':kappa,'B0':B0,'Ip':Ip,'fgr':fgr,'fuel':fuel,'impurity':impurity,'T0':T0,'alpha_T':alpha_T,\
              'alpha_n':alpha_n,'alpha_imp':alpha_imp,'H':H,'xi':xi,'f_ash':f_ash,'f_LH':f_LH,'scalinglaw':scalinglaw}

    
if numtrials > 1:
    simple_plot(input_dict,numtrials,xaxis,zaxis,plot_cs,contourplot) 


if numtrials == 1:
    # plot profiles
    plotprofiles(input_dict,plotprof,plotprad,plotlhscan,plott0scan,plotDHe3scan)
    
    # popcon
    arc_calc_popcon_lmode(input_dict, T0_range, verbose)