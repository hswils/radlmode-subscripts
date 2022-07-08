import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

class radlmode:

    def __init__(self,Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,impurity,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH,scalinglaw):
        
        self.scalinglaw = scalinglaw
        
        if self.scalinglaw not in ['L89','98Y2','ISS04']:
            print("Invalid scaling law.")
            
        '''
        creates the structure with the inputs that will be used to predict ARC
        fusion performance...
        These are all inputs required to calculate ARC performance
        '''
        self.Rmaj = Rmaj        # major radius in m
        self.epsilon = 1/A      # epsilon is defined as inverse aspect ratio in this script
        
        self.delta = delta      # triangularity
        self.kappa = kappa      # elongation
        self.xi = xi
    
        self.B0 = B0            # magnetic field at plasma axis in T
        self.Ip = Ip            # plasma current in MA
        self.fgr = fgr          # Greenwald fraction
        
        '''
        parameters for what was previously arc_calc_core_lmode and arc_solve_lmode
        '''
        self.fuel = fuel
        self.impurity = impurity
        self.T0 = T0
        self.alpha_T = alpha_T
        self.alpha_n = alpha_n
        self.alpha_imp = alpha_imp
        self.xi = xi       
        self.H = H
        self.f_ash = f_ash
        self.f_LH = f_LH
        
        if self.fuel == 'DD':
            self.M_avg = 2.0
        else:
            self.M_avg = 2.5




        # @hw
        # Calculate normalized plasma squareness from xi as described in Appendix C of Sauter et al
        if self.xi == 0:    # xi = 0 causes division by zero, so use limit as xi -> 0 for second term of theta_07 eqn (which is 0)
            theta_07 = np.arcsin(0.7)
        else:
            theta_07 = np.arcsin(0.7)+(1-np.sqrt(1+8*self.xi**2))/(4*self.xi)
        
        self.w_07 = np.cos(theta_07 - self.xi*np.sin(2*theta_07))*(1-(0.49/2)*self.delta**2)/np.sqrt(0.51)
        
    def rate_coeff_BH(self, Ti, fuel):
        '''
        calculates the fusion rate coefficient in m3/s for
        temperature Ti (in keV) based on the Bosch-Hale rates
        https://iopscience.iop.org/article/10.1088/0029-5515/32/4/I07
        '''

        if fuel == 'DT':
            Bg = 34.3827        # keV**1/2
            mu_c2 = 1124656

            C1 = 1.17302e-9    # cm3/s
            C2 = 1.51361e-2
            C3 = 7.51886e-2
            C4 = 4.60643e-3
            C5 = 1.35e-2
            C6 = -1.06750e-4
            C7 = 1.366e-5

        elif fuel == 'DD':
            Bg=31.39702         # keV^1/2
            mu_c2 = 937814.0e0

            C1=5.43360e-12      # cm3/s
            C2=5.85778e-3
            C3=7.68222e-3
            C4=0.0
            C5=-2.96400e-6
            C6=0.0
            C7=0.0

        elif fuel == 'DHe3':
            Bg = 68.7508        # keV^1/2
            mu_c2 = 1124572

            C1 = 5.51036e-10    # cm^3/s
            C2 = 6.41918e-3
            C3 = -2.02896e-3
            C4 = -1.91080e-5
            C5 = 1.35776e-4
            C6 = 0
            C7 = 0

        elif fuel == 'DHe3SP':
            Bg = 68.7508        # keV^1/2
            mu_c2 = 1124572

            C1 = 1.5 * 5.51036e-10    # cm^3/s   # Scaled by 50% to account for SP
            C2 = 6.41918e-3
            C3 = -2.02896e-3
            C4 = -1.91080e-5
            C5 = 1.35776e-4
            C6 = 0
            C7 = 0

        theta = Ti / (1-  (Ti * (C2 + Ti * (C4 + Ti * C6)))/(1 + Ti * (C3 + Ti * (C5 + Ti * C7))))

        psi=(Bg**2/(4.0*theta))**(1./3.)

        return C1*theta * np.sqrt(psi/(mu_c2*Ti**3.0)) * np.exp(-3.0*psi) / 1e6

    def poly(self, X, C):
        '''
        https://www.l3harrisgeospatial.com/docs/poly.html
        Needed for get_noble_lrad
        '''
        return C[0] + C[1]*X + C[2]*X**2 + C[3]*X**3 + C[4]*X**4

    def get_noble_lrad(self,t,impurity):
        '''
        provides the L_rad in W m3 for a species specified by the user by its
        elemental name, impurity source data: "Improved fits of coronal
        radiative cooling rates for high-temperature plasmas," A.A. Mavrin,
        Radiation Effects and Defects in Solids, 173:5-6, 388-398
        '''

        if impurity not in ['Ar', 'Kr', 'Xe', 'Ne']:
            print('Impurity not recognized!!! Error in impurity calculation!!!')
            return 0

        nt = np.shape(t)
        lrad = np.zeros(nt)
        z_avg = np.zeros(nt)

        # first check that all the sent T are in valid range

        if np.any(t < 0.1) or np.any(t>100):
            print('Bad T out of range!!! Error in impurity calculation!!!')
            return 0

        if impurity == 'Kr':    # Krypton

            N_fit_range = 3

            aa = np.zeros((N_fit_range,5))  # fit coefficients for log-log Lrad
            bb = np.zeros((N_fit_range,5))  # fit coefficients for linear log (T) <Z>

            trange = np.zeros((N_fit_range,2))
            zrange = np.zeros((N_fit_range,2))

            trange[0,:] = [0.1 , 0.447]
            aa[0,:] = [-3.4512E+01,-2.1484E+01,-4.4723E+01,-4.0133E+01,-1.3564E+01]
            zrange[0,:] = [0.1 , 0.447]
            bb[0,:] = [7.7040E+01,3.0638E+02,5.6890E+02,4.6320E+02,1.3630E+02]

            trange[1,:] = [0.447, 2.364]
            aa[1,:] = [-3.1399E+01,-5.0091E-01,1.9148E+00,-2.5865E+00,-5.2704E+00]
            zrange[1,:] = [0.447 , 4.117]
            bb[1,:] = [2.4728E+01,1.5186E+00,1.5744E+01,6.8446E+01,-1.0279E+02]

            trange[2,:] = [2.364, 100]
            aa[2,:] = [-2.9954E+01,-6.3683E+00,6.6831E+00,-2.9674E+00,4.8356E-01]
            zrange[2,:] = [4.117,100]
            bb[2,:] = [2.5368E+01,2.3443E+01,-2.5703E+01,1.3215E+01,-2.4682E+00]

        elif impurity == 'Ar':  # Argon

            N_fit_range = 3

            aa = np.zeros((N_fit_range,5))  # fit coefficients for log-log Lrad
            bb = np.zeros((N_fit_range,5))  # fit coefficients for linear log (T) <Z>

            trange = np.zeros((N_fit_range,2))
            zrange = np.zeros((N_fit_range,2))

            trange[0,:] = [0.1 , 0.6]
            aa[0,:] = [-3.2155E+01,6.5221E+00,3.0769E+01,3.9161E+01,1.5353E+01]
            zrange[0,:] = [0.1 , 0.6]
            bb[0,:] = [1.3171E+01,-2.0781E+01,-4.3776E+01,-1.1595E+01,6.8717E+00]

            trange[1,:] = [0.6 , 3]
            aa[1,:] = [-3.2530E+01,5.4490E-01,1.5389E+00,-7.6887E+00,4.9806E+00]
            zrange[1,:] = [0.6 , 3]
            bb[1,:] = [1.5986E+01,1.1413E+00,2.5023E+00,1.8455E+00,-4.8830E-02]

            trange[2,:] = [3, 100]
            aa[2,:] = [-3.1853E+01,-1.6674E+00,6.1339E-01,1.7480E-01,-8.2260E-02]
            zrange[2,:] = [3, 100]
            bb[2,:] = [1.4948E+01,7.9986E+00,-8.0048E+00,3.5667E+00,-5.9213E-01]

        elif impurity == 'Xe':  # Xenon

            N_fit_range = 4

            aa = np.zeros((N_fit_range,5))  # fit coefficients for log-log Lrad
            bb = np.zeros((N_fit_range,5))  # fit coefficients for linear log (T) <Z>

            trange = np.zeros((N_fit_range,2))
            zrange = np.zeros((N_fit_range,2))

            trange[0,:] = [0.1 , 0.5]
            aa[0,:] = [-2.9303E+01,1.4351E+01,4.7081E+01,5.9580E+01,2.5615E+01]
            zrange[0,:] = [0.1 , 0.3]
            bb[0,:] = [3.0532E+02,1.3973E+03,2.5189E+03,1.9967E+03,5.8178E+02]

            trange[1,:] = [0.5 , 2.5]
            aa[1,:] = [-3.1113E+01,5.9339E-01,1.2808E+00,-1.1628E+01,1.0748E+01]
            zrange[1,:] = [0.3 , 1.5]
            bb[1,:] = [3.2616E+01,1.6271E+01,-4.8384E+01,-2.9061E+01,8.6824E+01]

            trange[2,:] = [2.5 , 10]
            aa[2,:] = [-2.5813E+01,-2.7526E+01,4.8614E+01,-3.6885E+01,1.0069E+01]
            zrange[2,:] = [1.5 , 8]
            bb[2,:] = [4.8066E+01,-1.7259E+02,6.6739E+02,-9.0008E+02,4.0756E+02]

            trange[3,:] = [10 , 100]
            aa[3,:] = [-2.2138E+01,-2.2592E+01,1.9619E+01,-7.5181E+00,1.0858E+00]
            zrange[3,:] = [8 , 100]
            bb[3,:] = [-5.7527E+01,2.4056E+02,-1.9931E+02,7.3261E+01,-1.0019E+01]

        elif impurity == 'Ne':  # Neon

            N_fit_range = 3

            aa = np.zeros((N_fit_range,5))  # fit coefficients for log-log Lrad
            bb = np.zeros((N_fit_range,5))  # fit coefficients for linear log (T) <Z>

            trange = np.zeros((N_fit_range,2))
            zrange = np.zeros((N_fit_range,2))

            trange[0,:] = [0.1 , 0.7]
            aa[0,:] = [-3.3132E+01,1.7309E+00,1.5230E+01,2.8939E+01,1.5648E+01]
            zrange[0,:] = [0.1 , 0.5]
            bb[0,:] = [8.9737E+00,-1.3242E+01,-5.3631E+01,-6.4696E+01,-2.5303E+01]

            trange[1,:] = [0.7 , 5]
            aa[1,:] = [-3.3290E+01,-8.7750E-01,8.6842E-01,-3.9544E-01,1.7244E-01]
            zrange[1,:] = [0.5 , 2]
            bb[1,:] = [9.9532+00,2.1413E-01,-8.0723E-01,3.6868E+00,-7.0678E+00]

            trange[2,:] = [5, 100]
            aa[2,:] = [-3.3410E+01,-4.5345E-01,2.9731E-01,4.3960E-02,-2.6930E-02]
            zrange[2,:] = [2, 100]
            bb[2,:] = [1.000E+01,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00]

        # and now calculate Lrad and <Z>

        for l in np.arange(N_fit_range):
            sel = (t > trange[l,0]) * (t < trange[l,1])
            if np.any(sel):
                lrad[sel] = self.poly(np.log10(t[sel]), aa[l,:])

            sel = (t > zrange[l,0]) * (t < zrange[l,1])
            if np.any(sel):
                z_avg[sel] = self.poly(np.log10(t[sel]), bb[l,:])

        lrad = 10**lrad

        # fix boundary condition
        lrad[-1] = lrad[-2]

        result = {'t':t,
                  'impurity':impurity,
                  'lrad':lrad, # Wm^3
                  'z_avg':z_avg}

        return result


        
    def calc_lmode(self):
        '''
        arc_calc_geom: calculates basic geomtry and operating conditions given values provided in
        structure d, which contains the ARC geomtry from arc_make_input
        '''
        self.a = self.Rmaj*self.epsilon  # minor radius
        
        # @hw: Negative triangularity additions. Delta and kappa must be at the last closed flux surface
        self.L_p = 2*np.pi*self.a*(1+0.55*(self.kappa-1))*(1+0.08*self.delta**2)*(1+0.2*(self.w_07-1)) # poloidal length around the plasma cross-section
        self.surf_phi = np.pi*self.a**2*self.kappa*(1+0.52*(self.w_07-1))                              # surface of plasma cross section in the radial and poloidal direction
        
        # @hw: surf and vol are the new equations to account for -delta from Sauter paper
        self.surf = 2*np.pi*self.Rmaj*(1-0.32*self.delta*self.epsilon)*self.L_p                        # plasma surface area in m2
        self.vol = 2*np.pi*self.Rmaj*(1-0.25*self.delta*self.epsilon)*self.surf_phi                    # plasma volume in m3

        # radial grid of a 1000 points for profiles
        self.rho = (np.arange(1000)+1.0) * 0.001                                              # normalized radius
        self.r = self.rho*self.a   
        self.dL_p = 2*np.pi*self.r*(1+0.55*(self.kappa-1))*(1+0.08*self.delta**2)*(1+0.2*(self.w_07-1))                                                           # radial position in m
        self.dsurf_phi = np.pi*2*self.r*(self.r[1]-self.r[0])*self.kappa*(1+0.52*(self.w_07-1))
        self.dvol = 2*np.pi*self.Rmaj*(1-0.25*self.delta*self.epsilon)*self.dsurf_phi  # differential volume in m3
        self.dsurf = 2*np.pi*self.Rmaj*(1-0.32*self.delta*self.epsilon)*self.dL_p

        # plasma operationa parameters
        self.qstar = 5*(1+self.kappa**2) * (self.B0/self.Rmaj) * self.a**2/(2*self.Ip)        # effective safety factor, kink limit=2
        
        # @hw: Negative triangularity q95
        self.q95 = (4.1*self.a**2*self.B0/(self.Rmaj*self.Ip))*(1+1.2*(self.kappa-1)+0.56*(self.kappa-1)**2)* \
            (1+0.09*self.delta+0.16*self.delta**2)*((1+0.45*self.delta*self.epsilon)/(1-0.74*self.epsilon))*(1+0.55*(self.w_07-1))  #changed to formula from Sauter
        
        self.n_Gr = self.Ip / (np.pi*self.a**2)                                               # Greenwald density in n20
        self.n20 = self.fgr * self.n_Gr                                                       # operating line averaged density in 1e20 m-3
        self.P_LH = 0.0488 * self.n20**0.717 * self.B0**0.803 * self.surf**0.941              # L-H power threshold in MW

        # @hw: q profile to calculate iota profile (from Stroth paper) 
        if self.scalinglaw == 'ISS04':
            self.q_r = self.rho**2*self.qstar/(1-(1-self.rho**2)**4)
            self.iota_bar_r = 1/self.q_r 
            self.iota_bar = self.iota_bar_r[666] #produces iota_bar_2/3
          


        '''
        calculates power and equilibrium conditions for the ARC core with parameters from arc_calc_geom
        for an L-mode temperature profile T0(1-rho**2)**alpha_T, so no pedestal default shaping is parabolic to 1.5 power
        '''

        # the profile for the density has a shape where the boundary density is
        # at a fixed ratio to central density user though can change the parabolic power with alpha_n

        self.n20_r = ( 0.6*(1-self.rho**2)**self.alpha_n  + 0.4 )
        dum = np.average(self.n20_r)

        # normalize so that the line averaged density is n20
        self.n20_r *= self.n20 / dum
        self.n20_avg = np.sum(self.n20_r*self.dvol) / np.sum(self.dvol) # volume averaged density
        self.n_peaking = self.n20_r[0] / self.n20_avg

        # @hw: impurity density profile is the same as n20 profile except for alpha_imp, will be changed later by a factor of f_imp
        self.nimp = self.n20                                       
        self.nimp_r = ( 0.6*(1-self.rho**2)**self.alpha_imp  + 0.4 )
        dum = np.average(self.nimp_r)
        self.nimp_r *= self.nimp / dum

        # temperature profile.
        # separatrix temperature is set at 0.1 keV
        self.T_r = (self.T0-0.1) * (1-self.rho**2)**self.alpha_T + 0.1
        self.T_avg = np.sum(self.T_r * self.dvol) / np.sum(self.dvol)   # volume averaged temperature

        # calculate some standard parameters from the density and T profiles + geometry

        self.p_r = 2*(self.n20_r*1e20) * 1.6e-19*(1e3*self.T_r) / 1e6   # pressure profile in MPa
        self.p_avg = np.sum(self.p_r*self.dvol) / np.sum(self.dvol)     # volume averaged pressure in MPa

        self.beta_t = 100 * self.p_avg / (0.3979*self.B0**2)            # toroidal beta
        self.beta_n = self.beta_t / ( self.Ip/ (self.a*self.B0) )            # normalized beta

        self.Wth = np.sum(1.5 * self.p_r * self.dvol)                   # stored thermal energy in MJ
        self.T_avg_Wth = 1e-3*(self.p_avg*1e6) / (2*(self.n20*1e20)*1.6e-19)  # averaged temperature from <p> and line average density

        self.Coulomb_log = np.log( 4.9e7*self.T_r**1.5 / self.n20_r**0.5 )    # coulomb log profile

        self.sigmas_r = self.T_r**1.5 / ( self.Coulomb_log * 1.65e-9 )  # plasma conductivity in S/m radial profile
        self.sigmas_avg = np.sum(self.sigmas_r*self.dvol) / np.sum(self.dvol) # surface averaged conductivity in S/m
        self.Seff = self.sigmas_avg * ( np.pi * self.kappa * self.a**2 ) / (2 * np.pi * self.Rmaj) # effective conductance Sievert

        self.Reff = 1.0/self.Seff                                       # effective plasma resistance, ohm
        self.Pohm = self.Reff * self.Ip**2 * 1e6                        # ohmic heating in MW
        self.Vloop = self.Ip*1e6 * self.Reff                            # loop voltage in V

        # and fusion power using DT rate coefficient, assuming pure D-T

        self.SDT_r = self.rate_coeff_BH(Ti = self.T_r, fuel=self.fuel)               # in m3/s
        self.pfus_r = 0.25*(self.n20_r*1e20)**2*self.SDT_r*1.6e-19*17.6 # in MW/m3
        self.Pfus_orig = np.sum(self.pfus_r*self.dvol)                       # total fusion power in MW
        
        if self.fuel in ['DT','DHe3','DHe3SP']:
            self.Palpha_orig = 0.2*self.Pfus_orig                                     # alpha power in MW
        elif self.fuel == 'DD':
            self.Palpha_orig = 0.25*self.Pfus_orig
        else:
            print('Fuel not recognized...')

        if self.fuel in ['DHe3','DHe3SP']:
            self.pure_zeff = 0.7*1**2 + 0.3*2**2 # 70:30 Fuel mix
        else:
            self.pure_zeff = 1

        # and since we're here, we can also calculate the radiated power and
        # average Z of the seeded impurity required for keeping the plasma in
        # L-mode and/or power exhaust
        # for simplicity this assumes the impurity fraction is 1e-3

        # krypton
        c = self.get_noble_lrad(self.T_r,'Kr')
        dum = 1e-3*(self.nimp_r*1e20)**2 * c['lrad']/1e6               # in MW/m3
        prad = np.sum(dum*self.dvol)                               # radiated power in MW at f_Z=1e-3
        zdilute = np.sum(c['z_avg']*self.pfus_r*self.dvol) / np.sum(self.pfus_r*self.dvol) # fusion power weighted average Z
        self.krypton = {'c':c,'prad':prad,'zdilute':zdilute}

        # argon
        c = self.get_noble_lrad(self.T_r,'Ar')
        dum = 1e-3*(self.nimp_r*1e20)**2 * c['lrad']/1e6               # in MW/m3
        prad = np.sum(dum*self.dvol)                               # radiated power in MW at f_Z=1e-3
        zdilute = np.sum(c['z_avg']*self.pfus_r*self.dvol) / np.sum(self.pfus_r*self.dvol) # fusion power weighted average Z
        self.argon = {'c':c,'prad':prad,'zdilute':zdilute}

        # xenon
        c = self.get_noble_lrad(self.T_r,'Xe')
        dum = 1e-3*(self.nimp_r*1e20)**2 * c['lrad']/1e6               # in MW/m3
        prad = np.sum(dum*self.dvol)                               # radiated power in MW at f_Z=1e-3
        zdilute = np.sum(c['z_avg']*self.pfus_r*self.dvol) / np.sum(self.pfus_r*self.dvol) # fusion power weighted average Z
        self.xenon = {'c':c,'prad':prad,'zdilute':zdilute}

        # neon
        c = self.get_noble_lrad(self.T_r,'Ne')
        dum = 1e-3*(self.nimp_r*1e20)**2 * c['lrad']/1e6               # in MW/m3
        prad = np.sum(dum*self.dvol)                               # radiated power in MW at f_Z=1e-3
        zdilute = np.sum(c['z_avg']*self.pfus_r*self.dvol) / np.sum(self.pfus_r*self.dvol) # fusion power weighted average Z
        self.neon = {'c':c,'prad':prad,'zdilute':zdilute}
        
        '''
        calculates power and equilibrium conditions using profiles from
        arc_calc_core_lmode(T0) with input of H89 confinement

        impurity is a string keyword that indicates the identity of the
        radiating impurity, if any:
            # impurity = 'pure'    no impurity
            # impurity = 'ash'  He ash only
            # impurity = 'Kr'  Krypton
            # impurity = 'Ar' Argon
            # impurity = 'Xe' Xenon

        f_ash is a keyword for the He ash impurity fraction
        default is f_ash=0.02 (because L-mode?? need to check)

        f_Lh is the requested fraction of the LH power threshold
        f_LH = 1 means the Ploss-Prad = Plh#  if f_LH = 0.5, Ploss-Prad=0.5*PLh
        '''

        if self.impurity not in [ "pure", "ash", "Kr", "Ar", "Xe", "Ne"]:
            print('WARNING: IMPURITY NOT RECOGNIZED. ERROR IN calc_lmode')

        # SCALING LAWS
        if self.scalinglaw == 'L89':
            # first calculate taue_E*P**0.5  in seconds - (MW**0.5) for L-mode
    
            dum = 0.048 * self.H * self.n20**0.1 * np.sqrt(self.M_avg) * self.Ip**0.85 * \
                    self.Rmaj**1.2 * self.a**0.3 * self.kappa**0.5 * self.B0**0.2 # tau_E*sqrt(P)

            self.Ploss = (self.Wth / dum)**2.0         # this is equilibrium PLoss (MW) by definition (with no prad...)
            # Ploss = Wth^2 / tau_E^2 / Ploss
    
            self.tau_E = self.Wth / self.Ploss          # energy confinement time in seconds

        elif self.scalinglaw == '98Y2':
            
            dum = 0.0562*self.H*self.Ip**0.93*self.B0**0.15*self.n20**0.43* \
                self.Rmaj**1.39*self.M_avg**0.19*self.kappa**0.78*self.a**0.58
            
            self.Ploss = (self.Wth/dum)**(1/(1-0.69))  # exponent should be (1/1+x) where x is the power exp in scaling law
            self.tau_E = self.Wth/self.Ploss
            
        elif self.scalinglaw == 'ISS04':
            
            # first calculate taue_E*P**-0.61 in seconds for L-mode
            dum = 0.134 * self.H * self.a**2.28 * self.Rmaj**0.64 * (self.n20*10)**(0.54) * self.B0**0.84 * self.iota_bar**0.41  # tau_E*P**0.61
            self.Ploss = (self.Wth / dum)**(1/(1-0.61)) 
            self.tau_E = self.Wth / self.Ploss 

            # @hw stellarator density limit from Sudo et al 1990 
            self.n_lim = 0.25*self.B0**0.5*self.a**(-1)*self.Rmaj**(-0.5)*self.Ploss**0.5
            
            if self.n20 > self.n_lim:
                print('Particle density exceeds stellarator limit.')


        # now solve self-consistently for the requested radiation and impurity
        # fraction. note that by definition Ploss and PLH do not vary as a
        # function of f_imp because they are defined by the geometry and plasma parameters

        # create a sub-structure to record the profile data

        self.profile={
                  'r':self.r,             # minor radius in meters
                  'rho':self.rho,
                  'T':self.T_r,           # temperature in keV
                  'n20':self.n20_r,       # density in n20
                  'palpha':np.zeros(1000),  # alpha heating power MW/m3
                  'prad':np.zeros(1000),    # radiated power MW/m3
                  'Pnet':np.zeros(1000),    # net power in MW
                  'qloss':np.zeros(1000),   # conducted areal power density (MW/m2)
                  'chi':np.zeros(1000)      # effective heat diffusivity m2/s
                  }

        # based on the impurity, preform different calculations:

        if self.impurity == 'pure': # pure D-T, can just use pre-calculated alpha power
            self.Prad = 0
            self.f_imp = 0
            self.zeff = self.pure_zeff
            self.Palpha = self.Palpha_orig
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0
            self.Pext = self.Ploss - self.Palpha # external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] *= 0.0

        elif self.impurity == 'ash': # D-T + He ash impurity at f_ash concentration, no radiating impurities
            self.Prad = 0
            self.f_imp = 0
            self.zeff = (1-self.f_ash)*self.pure_zeff + self.f_ash*2**2
            self.Palpha = (1-self.f_ash*2.0)**2 * self.Palpha_orig
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0 # this is overwritten from above? (# TODO: )
            self.Pext = self.Ploss - self.Palpha # external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * (1.0-self.f_ash*2.0)**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] *= 0.0

        elif self.impurity == 'Kr': # D-T + He ash impurity at f_ash concentration + Krypton for radiation
            self.Prad = self.Ploss - self.f_LH*self.P_LH # required radiation to keep plasma in L-mode in MW
            self.f_imp = 1e-3*(self.Prad/self.krypton['prad']) # required impurity fraction to obtain Prad
            self.zeff = (1-self.f_ash-self.f_imp)*self.pure_zeff + self.f_ash*2**2 +\
                self.f_imp * (np.sum(self.krypton['c']['z_avg']*self.dvol) / np.sum(self.dvol))**2            
            if 1-self.f_ash*2.0-self.f_imp*self.krypton['zdilute'] > 0:
                self.Palpha = (1-self.f_ash*2.0-self.f_imp*self.krypton['zdilute'])**2 * self.Palpha_orig
            else:
                self.Palpha = 0
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0 # this is overwritten from above? (# TODO: )
            self.Pext = self.Ploss - self.Palpha #  external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * \
                                     (1.0-self.f_ash*2.0 - self.f_imp*self.krypton['zdilute'])**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] = self.f_imp*(self.profile['n20']*1e20)**2 * self.krypton['c']['lrad']/1e6

        elif self.impurity == 'Ar': # D-T + He ash impurity at f_ash concentration + Argon for radiation
            self.Prad = self.Ploss - self.f_LH*self.P_LH # required radiation to keep plasma in L-mode in MW
            self.f_imp = 1e-3*(self.Prad/self.argon['prad']) # required impurity fraction to obtain Prad
            self.zeff = (1-self.f_ash-self.f_imp)*self.pure_zeff + self.f_ash*2**2 +\
                self.f_imp * (np.sum(self.argon['c']['z_avg']*self.dvol) / np.sum(self.dvol))**2
            if 1-self.f_ash*2.0-self.f_imp*self.argon['zdilute'] > 0:
                self.Palpha = (1-self.f_ash*2.0-self.f_imp*self.argon['zdilute'])**2 * self.Palpha_orig
            else:
                self.Palpha = 0
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0 # this is overwritten from above? (# TODO: )
            self.Pext = self.Ploss - self.Palpha #  external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * \
                                     (1.0-self.f_ash*2.0 - self.f_imp*self.argon['zdilute'])**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] = self.f_imp*(self.profile['n20']*1e20)**2 * self.argon['c']['lrad']/1e6

        elif self.impurity == 'Xe': # D-T + He ash impurity at f_ash concentration + Xenon for radiation
            self.Prad = self.Ploss - self.f_LH*self.P_LH # required radiation to keep plasma in L-mode in MW
            self.f_imp = 1e-3*(self.Prad/self.xenon['prad']) # required impurity fraction to obtain Prad
            self.zeff = (1-self.f_ash-self.f_imp)*self.pure_zeff + self.f_ash*2**2 +\
                self.f_imp * (np.sum(self.xenon['c']['z_avg']*self.dvol) / np.sum(self.dvol))**2            
            if 1-self.f_ash*2.0-self.f_imp*self.xenon['zdilute'] > 0:
                self.Palpha = (1-self.f_ash*2.0-self.f_imp*self.xenon['zdilute'])**2 * self.Palpha_orig
            else:
                self.Palpha = 0
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0 # this is overwritten from above? (# TODO: )
            self.Pext = self.Ploss - self.Palpha #  external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * \
                                     (1.0-self.f_ash*2.0 - self.f_imp*self.xenon['zdilute'])**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] = self.f_imp*(self.profile['n20']*1e20)**2 * self.xenon['c']['lrad']/1e6

        elif self.impurity == 'Ne': # D-T + He ash impurity at f_ash concentration + Neon for radiation
            self.Prad = self.Ploss - self.f_LH*self.P_LH # required radiation to keep plasma in L-mode in MW
            self.f_imp = 1e-3*(self.Prad/self.neon['prad']) # required impurity fraction to obtain Prad
            self.zeff = (1-self.f_ash-self.f_imp)*self.pure_zeff + self.f_ash*2**2 +\
                self.f_imp * (np.sum(self.neon['c']['z_avg']*self.dvol) / np.sum(self.dvol))**2            
            if 1-self.f_ash*2.0-self.f_imp*self.neon['zdilute'] > 0:
                self.Palpha = (1-self.f_ash*2.0-self.f_imp*self.neon['zdilute'])**2 * self.Palpha_orig
            else:
                self.Palpha = 0
            if self.fuel == 'DD':
                self.Pfus = self.Palpha*4.0
            else:
                self.Pfus = self.Palpha*5.0 # this is overwritten from above? (# TODO: )
            self.Pext = self.Ploss - self.Palpha #  external heating power in MW in equilibrium
            self.Qp = self.Pfus / self.Pext                # plasma Q
            self.profile['palpha'] = 0.25*(self.profile['n20']*1e20)**2 * \
                                     (1.0-self.f_ash*2.0 - self.f_imp*self.neon['zdilute'])**2 * \
                                     self.rate_coeff_BH(Ti = self.profile['T'], fuel=self.fuel) * 1.6e-19*3.5
            self.profile['prad'] = self.f_imp*(self.profile['n20']*1e20)**2 * self.neon['c']['lrad']/1e6


        # @hw: recalculate impurity profile with calculated f_imp
        self.nimp = self.f_imp*self.n20
        dum = np.average(self.nimp_r)
        self.nimp_r *= self.nimp / dum
        self.nimp_avg = np.sum(self.nimp_r * self.dvol) / np.sum(self.dvol)

        # assume external power is uniformly distributed

        self.profile['Pnet'][0] = (self.profile['palpha'][0] + self.Pext/self.vol - self.profile['prad'][0]) * self.dvol[0]

        for i in np.arange(1,1000):
            self.profile['Pnet'][i] = self.profile['Pnet'][i-1] + \
                   (self.profile['palpha'][i] + self.Pext/self.vol - self.profile['prad'][i]) * self.dvol[i]

        self.profile['qloss'] = self.profile['Pnet']/self.dsurf
        dTdr = 1e3*np.gradient(self.profile['T'],self.profile['r'])   # dT/dr in eV/m
        self.profile['chi'] = (-1.0)*(1e6*self.profile['qloss']) / ( 1.6e-19*(self.profile['n20']*1e20)*dTdr )

        self.Psol = self.Ploss-self.Prad         # power to SOL
        
        self.Pwall = (self.Prad+self.Pfus) / self.surf # first wall power load (MW/m^2)

        # volume-averaged bremsstrahlung radiation power loss
        # from https://doi.org/10.1088/0741-3335/47/8/011

        Tm = 511. # keV
        xrel_r = (1+2*self.T_r/Tm) * (1 + (2/self.zeff) * (1-1/(1+self.T_r/Tm)))
        Kb = self.n20_r**2 * self.T_r**0.5 * xrel_r # I think these units are right?
        self.P_brem = 5.35e-3 * self.zeff * (np.sum(Kb * self.dvol) / np.sum(self.dvol)) * np.sum(self.dvol) # Volume-averaged bremsstrahlung radiaton in MW

        # volume-averaged synchrotron radiation power loss
        # from https://doi.org/10.1088/0741-3335/47/8/011

        rhoa = 6.04e3 * self.a * self.n20_r / self.B0 # optical thickness
        Rw = 0.9 # wall reflectivity
        gammaT = 2
        Ks = (self.alpha_n + 3.87*self.alpha_T + 1.46)**(-0.79) * (1.98 + self.alpha_n)**(1.36) * gammaT**2.14 * (gammaT**1.53 + 1.87*self.alpha_T - 0.16)**(-1.33)
        Gs = 0.93 * (1 + 0.85 * np.exp(-0.82*self.Rmaj/self.a))
        phi = 6.86E-5 * self.kappa**(-0.21) * (16+self.T_r)**(2.61) * ((rhoa/(1-Rw))**(0.41) + 0.12*self.T_r)**(-1.51) * Ks * Gs
        P_sync_r = 6.25e-3 * self.n20_r * self.T_r * self.B0**2 * phi
        self.P_sync = np.sum(P_sync_r * self.dvol)
        
        if self.delta == 0:
            print(self.tau_E)


       
def simple_plot(in_dict,numtrials,xaxis,zaxis,plot_cs = False,contourplot = False):
        
    '''
    If contourplot = False, plots a number of points given by numtrials in between the range the user
    has inputted for one or more variables in the input file. If multiple variables are "scanned" (entered
    as a range in the input file) they are each split into numtrials amount of points and the output is
    evaluated at each point.
    If contourplot = True, exactly two variables must be entered as a range and the first is plotted on
    the horizontal axis, the second is on the vertical, and the output is a contour. 
    
    Inputs:
    ----------------
        in_dict:   dictionary that holds all the input variables 
        numtrials: number of trials you would like to scan over
        xaxis:     the variable plotted on the xaxis automatically done when contourplot = True
        zaxis:     output variable on the vertical axis of 2D plot, or output variable when contourplot = True
        plotcs:    if 'True', plot the plasma cross sections
        contourplot: if 'True', make a contour plot with two scanned variables. '''
    
    # Assigning variables from input file
    fuels,impurity,scalinglaws = in_dict['fuel'],in_dict['impurity'],in_dict['scalinglaw']
    
    # Make temporary dictionary of numerical variables
    var_dict = copy.deepcopy(in_dict)           # Deep copy because there are lists in in_dict
    del var_dict['fuel']; del var_dict['impurity']; del var_dict['scalinglaw'] #deleting non-numerical variables

    # The following makes the var_list into a list of lists with some variables scanning over a range
    for var in var_dict:
        if isinstance(var_dict[var],list):                                              # If the user entered a list for the variable instead of a single value                             
            var_dict[var] = np.linspace(var_dict[var][0],var_dict[var][-1],numtrials)           # Make that variable range over the values given in input         
        else:
            var_dict[var] = var_dict[var]*np.ones(numtrials)                                    # Otherwise, make an array of that value repeated numtrials times

    z_array = []                                # Holds outputs for different fuel, impurity, and scaling law combinations, if applicable
    z_label = []                                # An array to later label the fuel, impurity, and scaling law for each output

    # The following runs the radlmode class for each variable combination
    for fuel in fuels:                # Scan over each value entered in fuels
        for imp in impurity:          # Scan over each value entered in impurity
            for sl in scalinglaws:    # Scan over each value entered in scalinglaws    
                z=[]                                        # Initialize y for each fuel/impurity/scaling law combination
                for i in np.arange(numtrials):              # Scan over every trial
                    # Assign variables for the trial:
                    Rmaj,A,delta,kappa = var_dict['Rmaj'][i],var_dict['A'][i],var_dict['delta'][i],var_dict['kappa'][i]
                    B0,Ip,fgr,T0 = var_dict['B0'][i],var_dict['Ip'][i],var_dict['fgr'][i],var_dict['T0'][i]
                    alpha_T,alpha_n,alpha_imp = var_dict['alpha_T'][i],var_dict['alpha_n'][i],var_dict['alpha_imp'][i]
                    H,xi,f_ash,f_LH = var_dict['H'][i],var_dict['xi'][i],var_dict['f_ash'][i],var_dict['f_LH'][i]
                    
                    
                    # Initialize class and run calc_lmode for the trial:         
                    ps = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH,sl)
                    ps.calc_lmode()
                    
                    # A dictionary to pull the value the user specified in inputs
                    output_dict = {'Pfus/volume':ps.Pfus/ps.vol,'Pfus':ps.Pfus,'Palpha':ps.Palpha,'P_LH':ps.P_LH,'Pext':ps.Pext,'Psol':ps.Psol,'Prad':ps.Prad,'Pohm':ps.Pohm,'Ploss':ps.Ploss, \
                            'qstar':ps.qstar,'q95':ps.q95,'volume':ps.vol,'surface':ps.surf,'n_Gr':ps.n_Gr,'n20':ps.n20,'tau_E':ps.tau_E,'f_imp':ps.f_imp,'beta_n':ps.beta_n,'beta_t':ps.beta_t,'Qp':ps.Qp,'Wth':ps.Wth}
                        
                    # Save the user-specified yaxis value for the trial
                    z.append(output_dict[zaxis])            
                    
                    # Plot cross section if applicable
                    if plot_cs == True:
                        theta = np.linspace(0,2*np.pi,1000)
                        R = Rmaj + ps.a*np.cos(theta+delta*np.sin(theta)-xi*np.sin(2*theta))
                        Z = kappa*ps.a*np.sin(theta+xi*np.sin(2*theta))
                        plt.plot(R,Z,label = '$a$ = %.2f, $R$ = %.2f' % (ps.a,Rmaj))
                        #plt.plot(R,Z,label = '$\delta$ = %.2f' % (delta))

               
                
                z_array.append(z)                       # Save the y for this scaling law/fuel/impurity combo
                z_label.append([sl,fuel,imp])   # Save the label for this scaling law/fuel/impurity combo
                
                if plot_cs == True:
                    plt.xlabel('R (m)')
                    plt.ylabel('Z (m)')
                    plt.axis('equal')
                    #plt.legend()
                    plt.savefig('crosssections.pdf')

    # Assign x from user-specifed xaxis value using input_dict
    x = np.linspace(in_dict[xaxis][0],in_dict[xaxis][-1],numtrials)
  
    # Plotting 
    
    # Dictionary to label plots
    label_dict = {'Rmaj':'$R_{maj}$ (m)','A':'$A$','delta':'$\delta$','kappa':'$\kappa$','B0':'$B_0$ (T)','Ip':'$I_p$ (MA)',\
                  'fgr':'$f_{Gr}$','T0':'$T_0$ (keV)','alpha_T':'$\\alpha_T$','alpha_n':'$\\alpha_n$','alpha_imp':'$\\alpha_{imp}$',\
                      'H':'$H$','xi':'$\\xi$','f_ash':'$f_{ash}$','f_LH':'$f_{LH}$','Pfus/volume':'$P_{fus}/volume$ (MW/m$^3$)',\
                          'Pfus':'$P_{fus}$ (MW)','Palpha':'$P_{\\alpha}$ (MW)','P_LH':'$P_{LH}$ (MW)','Pext':'$P_{ext}$ (MW)',\
                              'Psol':'$P_{SOL}$ (MW)','Pohm':'$P_{ohm}$ (MW)','Ploss':'$P_{loss}$ (MW)','qstar':'$q^{\\ast}$',\
                                  'q95':'$q_{95}$','volume':'$volume$ (m$^3$)','surface':'$surface$ $area$ (m$^2$)','n_Gr':'$n_{Gr}$',\
                                      'n20':'$n_{20}$ (m$^{-3}$)','tau_E':'$\\tau_E$ (s)','f_imp':'$f_{imp}$','beta_n':'$\\beta_N$','beta_t':'$\\beta_T$','Qp':'$Q_p$','Wth':'$W_{th}$ (MJ)'}
                     

    
    plt.figure(dpi = 150)
    for i,zz in enumerate(z_array):
        plt.plot(x,zz,lw = 2,label = z_label[i])

    plt.xlabel(label_dict[xaxis])
    plt.ylabel(label_dict[zaxis])
    plt.legend()
    plt.show() 

    ## Plot contours if applicable    
    if contourplot:
        fuel = in_dict['fuel'][0]; imp = in_dict['impurity'][0]; sl = in_dict['scalinglaw'][0] # Chooses first item in each string list if user entered multiple for the 2D plot
        var_matrix_dict = copy.deepcopy(in_dict)
        var_matrix_dict.pop('fuel'); var_matrix_dict.pop('impurity'); var_matrix_dict.pop('scalinglaw')
        
        axis = []                                      # List to keep scanned variable names in for labeling the plot
        count = 1                                      # Count of scanned variables
        for var in var_matrix_dict:
            A = np.empty((numtrials,numtrials))        # Empty matrix to hold scanned variable values.
            if isinstance(var_matrix_dict[var],list):
                axis.append(var)
                a = np.linspace(var_matrix_dict[var][0],var_matrix_dict[var][-1],numtrials) # An array made from the range entered for the scanned variable

                for i in np.arange(len(a)):
                    A[i,:] = a                         # Repeats 'a' for every row

                if count == 1:                         
                    var_matrix_dict[var] = A                  # The first scanned variable will be represented by A (value increases over rows)
                elif count == 2:
                    var_matrix_dict[var] = np.transpose(A)    # The second scanned variable will be represented by A-transpose (value increases over columns)
                count+=1
            
            else:
                var_matrix_dict[var] = var_matrix_dict[var]*np.ones((numtrials,numtrials)) # Every other variable is represented by a matrix of the variable's value repeated

        # A matrix to hold values
        zfield = np.empty((numtrials,numtrials))       # Field we will be plotting
        
        # Scan over every trial
        for i in np.arange(numtrials):
            for j in np.arange(numtrials):
                # Pull values from the i,j point of the variable matrix stored in var_matrix_dict
                
                Rmaj,A,delta,kappa = var_matrix_dict['Rmaj'][i][j],var_matrix_dict['A'][i][j],var_matrix_dict['delta'][i][j],var_matrix_dict['kappa'][i][j]
                B0,Ip,fgr,T0 = var_matrix_dict['B0'][i][j],var_matrix_dict['Ip'][i][j],var_matrix_dict['fgr'][i][j],var_matrix_dict['T0'][i][j]
                alpha_T,alpha_n,alpha_imp = var_matrix_dict['alpha_T'][i][j],var_matrix_dict['alpha_n'][i][j],var_matrix_dict['alpha_imp'][i][j]
                H,xi,f_ash,f_LH = var_matrix_dict['H'][i][j],var_matrix_dict['xi'][i][j],var_matrix_dict['f_ash'][i][j],var_matrix_dict['f_LH'][i][j]
                
                # Initialize class and run calc_lmode for the trial: 
                ps = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH,sl)
                ps.calc_lmode()
                
                output_dict = {'Pfus/volume':ps.Pfus/ps.vol,'Pfus':ps.Pfus,'Palpha':ps.Palpha,'P_LH':ps.P_LH,'Pext':ps.Pext,'Psol':ps.Psol,'Prad':ps.Prad,'Pohm':ps.Pohm,'Ploss':ps.Ploss, \
                        'qstar':ps.qstar,'q95':ps.q95,'volume':ps.vol,'surface':ps.surf,'n_Gr':ps.n_Gr,'n20':ps.n20,'tau_E':ps.tau_E,'f_imp':ps.f_imp,'beta_n':ps.beta_n,'beta_t':ps.beta_t,'Qp':ps.Qp,'Wth':ps.Wth}
                    
                # Save the user-specified yaxis value for the trial
                zfield[i,j] = (output_dict[zaxis])



        # Assign x from user-specifed xaxis values using input_dict
        x1 = np.linspace(in_dict[axis[0]][0],in_dict[axis[0]][-1],numtrials) #in_dict is somehow redefined as var_matrix_dict, don't know why, so it's now a matrix
        x2 = np.linspace(in_dict[axis[1]][0],in_dict[axis[1]][-1],numtrials)    
        
        # Plot contours
        X1,X2 = np.meshgrid(x1,x2)
        fig, ax1 = plt.subplots(dpi = 150)
        CS = ax1.contourf(X1,X2,zfield,levels = 10, cmap = cm.winter)
        CS2 = ax1.contour(X1,X2,zfield,levels = 10,cmap = cm.GnBu)
        ax1.set_xlabel(label_dict[axis[0]])
        ax1.set_ylabel(label_dict[axis[1]])    
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel(label_dict[zaxis])
        cbar.add_lines(CS2)
        plt.show()
        

def arc_calc_popcon_lmode(in_dict, T0_range=[2,30], verbose=True):
    '''
    this solves for, records and plots the Popcon for the ARC lmode
    scenarios, using the geometry and device inputs from plasma structure,
    which is obtained by calling arc_make_input()
    H89 sets confinement quality
    default space is fgr = 0.1 to 1.0, and T0 = 2 to 50 keV
    '''
    Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,impurity = in_dict['Rmaj'],in_dict['A'],in_dict['delta'],in_dict['kappa'],in_dict['B0'],in_dict['Ip'],in_dict['fgr'],in_dict['fuel'],in_dict['impurity']
    T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH = in_dict['T0'],in_dict['alpha_T'],in_dict['alpha_n'],in_dict['alpha_imp'],in_dict['H'],in_dict['xi'],in_dict['f_ash'],in_dict['f_LH']
    scalinglaw = in_dict['scalinglaw']

    # grid has resolution of 0.25 keV for T0
    Tres = 1
    T0 = np.arange( (T0_range[1]-T0_range[0])/Tres + 1 ) * Tres + T0_range[0]

    #grid has resolution of 0.01 in density as set by Greenwald fraction
    fgr = np.arange(9)*0.1 + 0.05

    n_T0 = len(T0)
    n_dens = len(fgr)

    # create arrays that will hold information
    pfus = np.zeros((n_T0,n_dens))
    pext = np.zeros((n_T0,n_dens))
    prad = np.zeros((n_T0,n_dens))
    psol = np.zeros((n_T0,n_dens))
    qpar = np.zeros((n_T0,n_dens))
    Qp = np.zeros((n_T0,n_dens))

    for i in np.arange(n_T0):
        for j in np.arange(n_dens):
            ps = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr[j],fuel[0],impurity[0],T0[i],alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH,scalinglaw[0])
            ps.calc_lmode()
            pfus[i,j] = ps.Pfus
            pext[i,j] = ps.Pext
            prad[i,j] = ps.Prad
            psol[i,j] = ps.Ploss - ps.Prad # power to SOL in MW
            Bpol = 0.2 * ps.Ip / ( ps.a * np.sqrt(0.5*(1+ps.kappa**2)) )  # Bpol in T
            lambdaq = 3 * 6.3e-4/Bpol**1.19 # in m from Eich scaling, x3 for L-mode
            Asol = 4*np.pi * ps.Rmaj * lambdaq * Bpol / ps.B0 # // area of SOL in m2
            qpar[i,j] = psol[i,j] / Asol
            Qp[i,j] = ps.Pfus / ps.Pext

            if Qp[i,j] < 0.0:  Qp[i,j] = 1000.0 # for plotting purposes clamp max Q at 1000.
            if Qp[i,j] > 1000.0:  Qp[i,j] = 1000.0  # for plotting purposes clamp max Q at 1000.

    print('lambda q (mm) = {}'.format(lambdaq/1e-3))

    if verbose:
        pfus_level=[30,50,100,300,500,1000,2000,3000]
        pext_level=[0,1,3,5,7,10,12,15,20,50]
        Qp_level=[1,3,5,10,20,30,50,100,1000]
        prad_surf_level=[0.1,0.3,0.5,1,2,3,4,5]
        qpar_level=[100,300,500,700,1000,2000,3000,4000]

        plt.figure()
        ax = plt.subplot(111)
        cs = plt.contour(T0, fgr, pfus.transpose(), levels=pfus_level, colors='k')
        ax.clabel(cs)

        cs = plt.contour(T0, fgr, pext.transpose(), levels=pext_level, colors='r')
        ax.clabel(cs)
        ax.set_xlabel('T0')
        ax.set_ylabel('fgr')
        # plt.contour(T0, fgr, Qp.transpose(), levels=Qp_level)
        plt.contour(T0, fgr, prad.transpose()/ps.surf, levels=prad_surf_level)
        # plt.contour(T0, fgr, qpar.transpose(), levels=qpar_level)
        plt.legend()
        plt.show()

    # result={a:a,impurity:impurity,f_LH:f_LH,H89:H89, $
    #         T0:T0,fgr:fgr,pfus:pfus,pext:pext,Qp:Qp,prad:prad,psol:psol,qpar:qpar}

    

  
def plotprofiles(in_dict,plotprof,plotprad,plotlhscan,plott0scan,plotDHe3scan):  

    Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,impurity = in_dict['Rmaj'],in_dict['A'],in_dict['delta'],in_dict['kappa'],in_dict['B0'],in_dict['Ip'],in_dict['fgr'],in_dict['fuel'],in_dict['impurity']
    T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH = in_dict['T0'],in_dict['alpha_T'],in_dict['alpha_n'],in_dict['alpha_imp'],in_dict['H'],in_dict['xi'],in_dict['f_ash'],in_dict['f_LH']
    scalinglaw = in_dict['scalinglaw']
    
    # Initialize plasma state
    ps = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel[0],impurity[0],T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,f_LH,scalinglaw[0])
    ps.calc_lmode()

    try:
        rho = ps.rho
        dens = ps.n20_r
        temp = ps.T_r
        nimp = ps.nimp_r
        
    except AttributeError:
        print('Profiles not available... ')

    else:
        if plotprof:
            plt.figure(dpi = 150)
            ax1 = plt.subplot(111)
        
            ax1.plot(rho, dens, c='k', lw=2, label='n (10$^{20}$ m$^{-3}$)')
            ax1.plot(rho, temp, c='b', lw=2, label='T (keV)')
            ax1.plot(rho, nimp, c='g', lw=2, label = 'n_Z (10$^{20}$ m$^{-3}$)')
            ax1.set_xlim(0,1)
            ax1.set_xlabel(r'$\rho$')
        
            ax1.text(0.7,0.9,r'$P_{fus}$ = ' + '{:.0f}MW'.format(ps.Pfus), transform = ax1.transAxes)
            ax1.text(0.7,0.85,r'$<p>$ = ' + '{:.2f}MPa'.format(ps.p_avg), transform = ax1.transAxes)
            ax1.text(0.7,0.8,r'$\beta_{n}$ = ' + '{:.2f}'.format(ps.beta_n), transform = ax1.transAxes)
            ax1.text(0.7,0.75,r'$P_{\alpha}$ = ' + '{:.0f}MW'.format(ps.Palpha), transform = ax1.transAxes)
            ax1.text(0.7,0.7,r'$P_{L-H}$ = ' + '{:.0f}MW'.format(ps.P_LH), transform = ax1.transAxes)
            ax1.text(0.7,0.65,r'$P_{ext}$ = ' + '{:.0f}MW'.format(ps.Pext), transform = ax1.transAxes)
            ax1.text(0.7,0.6,r'$P_{SOL}$ = ' + '{:.0f}MW'.format(ps.Psol), transform = ax1.transAxes)
            ax1.text(0.7,0.55,r'$P_{rad}$ = ' + '{:.0f}MW'.format(ps.Prad), transform = ax1.transAxes)
            ax1.text(0.7,0.5,r'$f_{Gr}$ = ' + '{:.2f}'.format(ps.fgr), transform = ax1.transAxes)
        
            plt.title('H89={:1.2f}, f_LH={:1.2f}, impurity={:s}'.format(ps.H, ps.f_LH, impurity[0]))
        
            plt.legend(ncol=2,loc=3)
            plt.savefig('fig_profiles.png')
    
     
        if plotprad:
            plt.figure(dpi = 150)
            ax1 = plt.subplot(111)
        
            ax1.text(0.5,0.7,'Impurity fraction:',color='k',transform=ax1.transAxes)
            
            i = 0
            for Z in ['Ar','Kr','Xe','Ne']:
                plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel[0],Z,T0,alpha_T,alpha_n, alpha_imp,H,xi,f_ash,f_LH,scalinglaw[0])                 
                plasma_state.calc_lmode()
                ax1.plot(plasma_state.rho, plasma_state.profile['prad'], lw=2, label=Z)
                ax1.text(0.5,0.65-i*0.05,'{:s}: {:1.3f}%'.format(Z,plasma_state.f_imp*100),transform=ax1.transAxes)
                i+=1
                   
            ax1.set_xlim(0,1)
            ax1.set_xlabel(r'$\rho$')
            ax1.set_ylabel(r'P$_{rad}$ (MW/m$^{3}$)')
    
            ax1.semilogy()
        
            plt.legend(ncol=3,loc=2)
            plt.savefig('fig_prad.png')
    
        
    
        if plotlhscan:
            plt.figure(figsize=(6.4,10),dpi = 150)
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312)
            ax3 = plt.subplot(313)
    
            lhs = np.linspace(0,2,10)
            imps = ['pure','ash','Ar','Kr','Xe','Ne']
            pfus = {}
            pext = {}
            psol = {}
    
            for imp in imps:
                pfus[imp] = np.zeros(len(lhs))
                pext[imp] = np.zeros(len(lhs))
                psol[imp] = np.zeros(len(lhs))
    
                for i,lh in enumerate(lhs):
                    plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel[0],imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                    plasma_state.calc_lmode()
                    pfus[imp][i] = plasma_state.Pfus
                    pext[imp][i] = plasma_state.Pext
                    psol[imp][i] = plasma_state.Psol
    
                    if plasma_state.qstar < 2:
                        print('QSTAR NOT SATISFIED!')
    
            ax1.plot(lhs, pfus['pure'], ls='--', lw=2, c='m', label='pure DT')
            ax1.plot(lhs, pfus['ash'], ls='--', lw=2, c='c', label='2% ash')
            ax1.plot(lhs, pfus['Ar'], ls='-', lw=2, c='k', label='Ar')
            ax1.plot(lhs, pfus['Kr'], ls='-', lw=2, c='b', label='Kr')
            ax1.plot(lhs, pfus['Xe'], ls='-', lw=2, c='r', label='Xe')
            #ax1.plot(lhs, pfus['Ne'], ls='-', lw=2, c='g', label='Ne')
    
            ax2.plot(lhs, pext['pure'], ls='--', lw=2, c='m', label='pure DT')
            ax2.plot(lhs, pext['ash'], ls='--', lw=2, c='c', label='2% ash')
            ax2.plot(lhs, pext['Ar'], ls='-', lw=2, c='k', label='Ar')
            ax2.plot(lhs, pext['Kr'], ls='-', lw=2, c='b', label='Kr')
            ax2.plot(lhs, pext['Xe'], ls='-', lw=2, c='r', label='Xe')
            #ax2.plot(lhs, pext['Ne'], ls='-', lw=2, c='g', label='Ne')
    
    
            ax3.plot(lhs, psol['pure'], ls='--', lw=2, c='m', label='pure DT')
            ax3.plot(lhs, psol['ash'], ls='--', lw=2, c='c', label='2% ash')
            ax3.plot(lhs, psol['Ar'], ls='-', lw=2, c='k', label='Ar')
            ax3.plot(lhs, psol['Kr'], ls='-', lw=2, c='b', label='Kr')
            ax3.plot(lhs, psol['Xe'], ls='-', lw=2, c='r', label='Xe')
            #ax3.plot(lhs, psol['Ne'], ls='-', lw=2, c='g', label='Ne')
    
    
            ax1.legend(ncol=2,loc=4)
    
            ax2ymin = ax2.get_ylim()[0]
            if ax2ymin < 0:
                ax2.axhspan(ax2ymin*2,0,color='k',alpha=0.3)
                ax2.set_ylim(ax2ymin)
                ax2.text(0.75,0.1,'ignited',ha='center',va='bottom',transform=ax2.transAxes)
    
            ax1.axvspan(min(lhs),1,color='g',alpha=0.3)
            ax2.axvspan(min(lhs),1,color='g',alpha=0.3)
            ax3.axvspan(min(lhs),1,color='g',alpha=0.3)
    
            ax1.text(0.25,1,'L-mode',ha='center',va='bottom',transform=ax1.transAxes,color='g')
            ax2.text(0.25,1,'L-mode',ha='center',va='bottom',transform=ax2.transAxes,color='g')
            ax3.text(0.25,1,'L-mode',ha='center',va='bottom',transform=ax3.transAxes,color='g')
    
            ax3.set_ylim(0)
    
            ax1.set_ylabel('$P_{fus}$ (MW)')
            ax2.set_ylabel('$P_{ext}$ (MW)')
            ax3.set_ylabel('$P_{sol}$ (MW)')
            ax1.set_xlabel('$f_{LH}$')
            ax2.set_xlabel('$f_{LH}$')
            ax3.set_xlabel('$f_{LH}$')
            ax1.set_xlim(min(lhs),max(lhs))
            ax2.set_xlim(min(lhs),max(lhs))
            ax3.set_xlim(min(lhs),max(lhs))
            plt.savefig('fig_lhscan.png')
        if plott0scan:
            plt.figure(figsize=(6.4,6), dpi = 150)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
    
            T0s =  np.arange(5,80,2)
            imps = ['Ar','Kr','Xe']
            pfus = {}
            prad = {}
            psol = {}
            flhs = {}
            fimp = {}
    
            psolconstraint = 30
    
            for imp in imps:
                pfus[imp] = np.zeros(len(T0s))
                psol[imp] = np.zeros(len(T0s))
                prad[imp] = np.zeros(len(T0s))
                flhs[imp] = np.zeros(len(T0s))
                fimp[imp] = np.zeros(len(T0s))
    
                for i,T0 in enumerate(T0s):
                    lh = 0.175
                    plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel[0],imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                    plasma_state.calc_lmode() 
    

                    if plasma_state.Psol > psolconstraint:
                        print('psol contraint not met! Psol={:.2f}MW'.format(plasma_state.Psol))
    
                    flhs[imp][i] = lh
                    pfus[imp][i] = plasma_state.Pfus
                    psol[imp][i] = plasma_state.Psol
                    prad[imp][i] = plasma_state.Prad
                    fimp[imp][i] = plasma_state.f_imp
    
            ax1.plot(T0s, pfus['Ar'], ls='-', lw=2, c='k', label='Ar')
            ax1.plot(T0s, pfus['Kr'], ls='-', lw=2, c='b', label='Kr')
            ax1.plot(T0s, pfus['Xe'], ls='-', lw=2, c='r', label='Xe')
    
            ax2.plot(T0s, fimp['Ar']*100, ls='-', lw=2, c='k', label='Ar')
            ax2.plot(T0s, fimp['Kr']*100, ls='-', lw=2, c='b', label='Kr')
            ax2.plot(T0s, fimp['Xe']*100, ls='-', lw=2, c='r', label='Xe')
    
            ax2.legend(ncol=3,loc=2)
    
            ax1.set_title('$P_{SOL}$ constraint: %d MW, fuel = %s, H89 = %0.1f' % (psolconstraint,fuel[0],H))
    
            ax1.set_ylabel('$P_{fus}$ (MW)')
            ax2.set_ylabel('$f_{imp}$ (%)')
            ax1.set_xlabel('$T_{0}$')
            ax2.set_xlabel('$T_{0}$')
    
            plt.tight_layout()
    
            plt.savefig('fig_t0scan.png')
    
       
        if plotDHe3scan:
            imp = 'Xe' # local variable
            fuel = 'DHe3'
    
            plt.figure(figsize=(6.4,6),dpi = 150)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
    
            T0 = 30
            psolconstraint = 30
    
            H89s = np.linspace(0.9,2.5)
    
            pfus = np.zeros(len(H89s))
            prad = np.zeros(len(H89s))
            psol = np.zeros(len(H89s))
            flhs = np.zeros(len(H89s))
            fimp = np.zeros(len(H89s))
    
            for i,H in enumerate(H89s):
                lh = 1
                plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                plasma_state.calc_lmode()
    
                while plasma_state.Psol > psolconstraint:
                    lh *= 0.9
                    plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                    plasma_state.calc_lmode()                    
    
                flhs[i] = lh
                pfus[i] = plasma_state.Pfus
                psol[i] = plasma_state.Psol
                prad[i] = plasma_state.Psol
                fimp[i] = plasma_state.f_imp
    
            ax1.plot(H89s, pfus, ls='-', lw=2, c='k', label='DH3')
    
            Ips = np.linspace(10,30)
            H = 2
    
            pfus = np.zeros(len(Ips))
            prad = np.zeros(len(Ips))
            psol = np.zeros(len(Ips))
            flhs = np.zeros(len(Ips))
            fimp = np.zeros(len(Ips))
    
            for i,Ip in enumerate(Ips):
                lh = 1
                plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                plasma_state.calc_lmode()
    
                while plasma_state.Psol > psolconstraint:
                    lh *= 0.9
                    plasma_state = radlmode(Rmaj,A,delta,kappa,B0,Ip,fgr,fuel,imp,T0,alpha_T,alpha_n,alpha_imp,H,xi,f_ash,lh,scalinglaw[0])
                    plasma_state.calc_lmode()
    
                flhs[i] = lh
                pfus[i] = plasma_state.Pfus
                psol[i] = plasma_state.Psol
                prad[i] = plasma_state.Psol
                fimp[i] = plasma_state.f_imp
    
            ax2.plot(Ips, pfus, ls='-', lw=2, c='k', label='DHe3')
    
            ax1.set_title(fuel+': $P_{SOL}$='+' {}MW'.format(psolconstraint)+', impurity='+imp+ ', T0={}'.format(T0))
    
            ax1.set_ylabel('$P_{fus}$ (MW)')
            ax2.set_ylabel('$P_{fus}$ (MW)')
            ax1.set_xlabel('$H_{89}$   $(I_{p}=17.4$MA)')
            ax2.set_xlabel('$I_{p}$[MA]   $(H_{89}=2)$')
    
            plt.tight_layout()
    
            plt.savefig('fig_DHe3scan.png')
            

