from radlmode_profiles import *

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

'''DIII-D Geometry'''

numtrials = 1

Rmaj = 1.68
A = 1.68/0.59          
delta = 0.47  
kappa = 1.686                 
xi = 0.0
B0 = 2.14                   
Ip = 1.22               
fgr = 0.35                 
fuel = ['DT']                 
impurity = ['Ar']
scalinglaw = ['L89']   
T0 = 1.3
alpha_T = 2.2
alpha_n = 0.7
alpha_imp = alpha_n
T_sep = 0.03
amp_n = 0.425
amp_imp = amp_n
b_T =  0.1 #0.15
b_n = 0.1
b_imp = b_n
c_T = 1.6
c_n = 1.98
c_imp = c_n
H_array = np.linspace(0.235,0.236,200) 
f_ash = 0.02
f_LH = 3.4


xaxis = 'T0'                
zaxis = 'f_imp'         

 # Choose zaxis string from below:
 # output_dict = {'Pfus/volume':ps.Pfus/ps.vol,'Pfus':ps.Pfus,'Palpha':ps.Palpha,'P_LH':ps.P_LH,'Pext':ps.Pext,'Psol':ps.Psol,'Prad':ps.Prad,'Pohm':ps.Pohm,'Ploss':ps.Ploss, \
 #                           'qstar':ps.qstar,'q95':ps.q95,'volume':ps.vol,'surface':ps.surf,'n_Gr':ps.n_Gr,'n20':ps.n20,'tau_E':ps.tau_E,'f_imp':ps.f_imp,'beta_n':ps.beta_n,'Qp':ps.Qp}

plot_cs = False
contourplot = True   

for H in H_array:
    ps = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel[0],impurity[0],T0,alpha_T,alpha_n,alpha_imp,T_sep,amp_n,amp_imp,b_T,b_n,b_imp,c_T,c_n,c_imp,H,xi,f_ash,f_LH,scalinglaw[0])
    ps.calc_lmode()
    if ps.f_imp < 0.01472 and ps.f_imp > 0.01471:
        print(ps.f_imp,H)

# Nothing needs edited below this line---------------------------------------------------------------

input_dict = {'Rmaj':Rmaj,'A':A,'delta':delta,'kappa':kappa,'B0':B0,'Ip':Ip,'fgr':fgr,'fuel':fuel,'impurity':impurity,'T0':T0,'alpha_T':alpha_T,\
              'alpha_n':alpha_n,'alpha_imp':alpha_imp,'T_sep':T_sep,'amp_n':amp_n,'amp_imp':amp_imp,'b_T':b_T,'b_n':b_n,'b_imp':b_imp,'c_T':c_T,'c_n':c_n,'c_imp':c_imp,\
                  'H':H,'xi':xi,'f_ash':f_ash,'f_LH':f_LH,'scalinglaw':scalinglaw}

    
if numtrials > 1:
    simple_plot(input_dict,numtrials,xaxis,zaxis,plot_cs,contourplot) 
