#Code used in master thesis project
#Dust in the intergalactic medium and Type Ia supernovae
#Karl Johanströmmer
#Stockholm University
#Department of Physics


#Used modules
import numpy as np
import pandas as pd
import pymultinest as pmn
import glob
import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import w0waCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import wCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
from time import time
from numpy import median

#note that pymultinest must be installed and pointed to the directory
#the following command works (ubuntu 16.04)
#export LD_LIBRARY_PATH=/home/path/to/MultinNest/lib:$LD_LIBRARY_PATH

#Physical constants
cc = 299792458            #speed of light
G = 6.67408*10**(-11)	  #Newton's gravitational constant
c_sec = 6.626e-25 		  #photon-electron cross-section cm**2
c_secsi = c_sec*1e-4      #cross-section m**2
mp = 1.6726219e-27        #proton mass (kg)
Y = 0.2467                #Helium/Hydrogen ratio


#DATA

#JLA 
#(zcmb dz mb dmb x1 dx1 color dcolor hm)
jla = pd.read_csv('sn_data/jla_lcparams.txt',
	delim_whitespace = True, header = 0)

#Pantheon (binned sample)
#(zcmb, mb, dmb)
pb = pd.read_csv('sn_data/lcparam_DS17f.txt',
	delim_whitespace = True, header = 0)
pb_stat = np.diag((pb.dmb.values)**2)                   #statistical uncertainty
pb_sys = np.loadtxt('sn_data/syscov_panth.txt')         #systematic uncertainty
pb_cm = pb_stat + pb_sys                                #covariance matrix
pb_icm = np.linalg.inv(pb_cm)                           #inverted covariance matrix

#ZTF (simulated data)
ztf = pd.read_csv('sn_data/ztf_msip.dat', 
	delim_whitespace = True, header = 0)
#Combining the ZTF + Pantheon samples
lowz = ztf.z.values
lowd = ztf.dmu.values
C1 = np.diag(lowd**2)
C2 = pb_cm[12:, 12:]
C3 = np.bmat([[C1, np.zeros((8, 28))], [np.zeros((28, 8)), C2]])  #covariance matrix of the combined sample
C41 = np.linalg.inv(C3)
C4 = np.asarray(C41)                                              #inverted covariance matrix

#WFIRST + Foundation
wfirst = pd.read_csv('/home/kj/dust/sn_data/lcparam_WFIRST_G10.txt', 
	delim_whitespace = True, header = 0)






#Functions


#Extinction law (CCM 89)
class ccm:
    def __init__(self, rv, av):
        self.av=av
        self.rv=rv
    def a(self, lam):
        x=1./lam
        if x >= .3 and x <= 1.1:
            a = .574* pow(x, 1.61)
        elif x >= 1.1 and x <= 3.3:
            y= x -1.82
            a=1 + .17699 *y - .50447*y**2 - 0.02427* y**3 +.72085*y **4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7            
        else:
            print("x is not in range")
        return a        

    def b(self, lam):
        x=1./lam
        if x >= .3 and x <= 1.1:
            b= -0.527 * pow(x, 1.61)
        elif x >= 1.1 and x <= 3.3:
            y= x-1.82
            b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260* y**6 - 2.09002*y**7
        else: 
            print("x is not in range")
        return b
        
    def alam(self, lam):
        alav=self.a(lam)+self.b(lam)/self.rv
        al=alav*self.av
        return al

    def kappa_lam(self, lam): #wavelenght dependent mass absorption coefficient 
        alav=self.a(lam)+self.b(lam)/self.rv
        kappa = 1.54e3*alav
        return kappa

Rv = 3.1 #total to selective absorption
Av = 0 #Total extinction in V-band. Not used
exti = ccm(Rv, Av)




### COSMOLOGY 
def E(om, z, w0, wa):
    w = w0 + wa*z/(1+z)
    return np.sqrt(om*(1.+z)**3.+(1.-om)*(1.+z)**(3.*w+3))

#Luminosity distance. 
def DL(om, z_max, w0, wa):
    res = integrate.quad(lambda z: 1 / E(om, z, w0, wa), 0, z_max)
    return cc/1000*(1.+z_max) * res[0]



#Attenuation from dust, in B-band supernova rest frame as function of redshift
#see equation 1.34 in thesis
def attenuation(om, h0, z_max, w0, wa, gamma):
    h0si = h0/(3.09*10**(19)) #h0 in SI
    res = quad(lambda z: exti.kappa_lam(0.44*(1+z_max)/(1+z))*(1+z)**(gamma-1)/E(om, z, w0, wa), 0, z_max)
    return 1.086*3*h0si*cc/(8*np.pi*G)*res[0]

#Attenuation from compton scattering on free electrons
def compt(om, h0, w0, wa, ob, z_max):
	h0si = h0/(3.09*10**(19))
	rho_c = 3*h0si**2/(8*np.pi*G)
	nH = (1-Y)*ob*rho_c/mp
	nHe = Y*ob*rho_c/(4*mp)
	ne = nH + 2*nHe
	res = quad(lambda z: (1+z)**2/E(om, z, w0, wa), 0, z_max)
	return 1.086*cc*c_secsi*ne/h0si*res[0]



#JLA covariance matrix (from sample_jla_cov.py)
def mu_cov(alpha, beta):
    """ Assemble the full covariance matrix of distance modulus

    See Betoule et al. (2014), Eq. 11-13 for reference
    """
    Ceta = sum([fits.getdata(mat) for mat in glob.glob('sn_data/jla_cov/C*.fits')])
    
    Cmu = np.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

    # Add diagonal term from Eq. 13
    sigma = np.loadtxt('sn_data/jla_cov/sigma_mu.txt')

    sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    
    return Cmu



#GENERATED SETS
#generation parameters
om_gen = 0.3   #matter density parameter
h0_gen = 70    #Hubble constant
w0_gen = -1    #DE EoS
wa_gen = 0     #extension EoS
od_gen = 8e-5  #dust density parameter
g_gen = -1     #dust redshift distribution exponent



#ZTF + Pantheon

z_zp = np.concatenate((lowz, pb.zcmb.values[12:])) #redshift points of ZTF + Pantheon sample
z_zp = np.asarray(z_zp) #asarray

dg_zp = [] #dust contribution to distance modulus
for m in range(0, len(z_zp)):
	dgg = od_gen*attenuation(om_gen, h0_gen, z_zp[m], w0_gen, wa_gen, g_gen)
	dg_zp.append(dgg)

ldg_zp = FlatLambdaCDM(h0_gen, om_gen).luminosity_distance(z_zp).value #luminosity distance of generated points
mod0_zp = 5*np.log10(ldg_zp)+25 #distance moduli of generated points, without dust contribution
mod1_zp = mod0_zp + dg_zp #distance moduli of generated points, with dust contribution




#WFIRST+Foundation

z_wf = wfirst.zcmb.values

dg_wf = [] #dust contribution to distance modulus
for n in range(0, len(z_wf)):
	dgw = od_gen*attenuation(om_gen, h0_gen, z_wf[n], w0_gen, wa_gen, g_gen)
	dg_wf.append(dgw)

ldg_wf = FlatLambdaCDM(h0_gen, om_gen).luminosity_distance(z_wf).value #luminosity distance of generated points
mod0_wf = 5*np.log10(ldg_wf)+25 #distance moduli of generated points, without dust contribution
mod1_wf = mod0_wf + dg_wf #distance moduli of generated points, with dust contribution




#instead of evaluating the attenuation integral
#for each of the 740 SNe in the JLA sample, we
#use interpolation.
#redshift for use in interpolation
z50 = np.linspace(0.001, 2.26, 50)



#MULTINEST


#naming:
#jlac = JLA Clean (no dust)
#jlad = JLA Dust
#pbc = Pantheon binned Clean
#pbd = Pantheon binned Dust
#zp = ZTF + Pantheon
#wf = WFIRST + Foundation
#flcdm = FlatLambdaCDM
#wcdm = wCDM
#wzcdm = w0waCDM

#########################################################
##################        JLA         ###################
#########################################################

#FlatlambdaCDM
#no dust
#log likelihood function
def llhood_jlac_flcdm(model_param, ndim, nparam):
	#parameters in use in this case
	om, a, b, MB, DM = [model_param[i] for i in range(5)]

	#h0 fixed to 70 to conform to Betoule et al. 2014
	h0 = 70
	#w0 = -1
	#wa = 0

	#luminosity distance to redshift z given h0 and om
	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jla.zcmb.values).value
	#convert to distance modulus
	mod_th = 5*np.log10(dist_th) + 25

	#hubble residuals (equation 2.8 in thesis for general case)
	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	#host mass correction
	hub_res[jla.hm.values >= 10.] += DM
	#assembling covariance matrix given alpha and beta (see Betoule et al. 2014)
	C = mu_cov(a, b)
	#invert covariance matrix
	iC = np.linalg.inv(C)
	#calculate chi-square
	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	#add CMB prior on matter density, uncomment to include
	#mu = 0.308
	#sigma = 0.012
	#cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2


	return -0.5*chisq# + cmb_prior

#flat prior function, in order listed at beginning of log likelihood function
#cube[n] defines a flat prior in the range [0,1]. 
#It can then be scaled and shifted by mutliplication and addition
#For example, the matter density parameter is assumed to be somewhere between 0 and 2, 
#and the Type Ia supernova absolute peak magnitude in B-band somewhere between -25 and -15 
def prior_jlac_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*0.3
	cube[2] = cube[2]*5
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*0.5-0.25


#wCDM
#no dust
def llhood_jlac_wcdm(model_param, ndim, nparam):
	om, w0, a, b, MB, DM = [model_param[i] for i in range(6)]

	h0 = 70
	wa = 0

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2


	return -0.5*chisq + cmb_prior

def prior_jlac_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*0.3
	cube[3] = cube[3]*5
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*0.5-0.25


#wzCDM
#no dust
def llhood_jlac_wzcdm(model_param, ndim, nparam):
	om, w0, wa, a, b, MB, DM = [model_param[i] for i in range(7)]

	h0 = 70

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2


	return -0.5*chisq + cmb_prior

def prior_jlac_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*6-3
	cube[3] = cube[3]*0.3
	cube[4] = cube[4]*5
	cube[5] = cube[5]*10-25
	cube[6] = cube[6]*0.5-0.25


#FlatlambdaCDM
#with dust
def llhood_jlad_flcdm(model_param, ndim, nparam):
	om, a, b, MB, DM, od, g = [model_param[i] for i in range(7)]

	h0 = 70
	w0 = -1
	wa = 0

	A = []
	for k in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[k], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)


	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_jlad_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*0.3
	cube[2] = cube[2]*5
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*0.5-0.25
	cube[5] = cube[5]*10-10
	cube[6] = cube[6]*9-6


#wCDM
#with dust
def llhood_jlad_wcdm(model_param, ndim, nparam):
	om, w0, a, b, MB, DM, od, g = [model_param[i] for i in range(8)]

	h0 = 70
	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_jlad_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*0.3
	cube[3] = cube[3]*5
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*0.5-0.25
	cube[6] = cube[6]*10-10
	cube[7] = cube[7]*9-6


#wzCDM
#with dust
def llhood_jlad_wzcdm(model_param, ndim, nparam):
	om, w0, wa, a, b, MB, DM, od, g = [model_param[i] for i in range(9)]

	h0 = 70

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_jlad_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*6-3
	cube[3] = cube[3]*0.3
	cube[4] = cube[4]*5
	cube[5] = cube[5]*10-25
	cube[6] = cube[6]*0.5-0.25
	cube[7] = cube[7]*10-10
	cube[8] = cube[8]*9-6




#######################################################
################## Pantheon (binned) ##################
#######################################################



def llhood_pbc_flcdm(model_param, ndim, nparam):
	om, h0, MB = [model_param[i] for i in range(3)]

	w0 = -1
	wa = 0

	#jlac_wz_c1 = np.loadtxt('posteriors/jla/jlac_wzcdm_cmb1.dat')
	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_pbc_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*10-25



def llhood_pbc_wcdm(model_param, ndim, nparam):
	om, h0, w0, MB = [model_param[i] for i in range(4)]

	wa = 0

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_pbc_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*10-25

def llhood_pbc_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa, MB = [model_param[i] for i in range(5)]

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior


def prior_pbc_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3
	cube[4] = cube[4]*10-25



def llhood_pbd_flcdm(model_param, ndim, nparam):
	om, h0, MB, od, g = [model_param[i] for i in range(5)]

	w0 = -1
	wa = 0
	#od = np.log10(8e-5)
	A = []
	for i in range(0,len(pb.zcmb.values)):
		At = 10**(od)*attenuation(om, h0, pb.zcmb.values[i], w0, wa, g)
		A.append(At)
	

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25
	hub_res = pb.mb.values - mod_th - MB - A
	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))
	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_pbd_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*10-25
	cube[3] = cube[3]*10-10
	cube[4] = cube[4]*9-6
 




def llhood_pbd_wcdm(model_param, ndim, nparam):
	om, h0, w0, MB, od, g = [model_param[i] for i in range(6)]

	wa = 0

	A = []
	for i in range(0,len(pb.zcmb.values)):
		At = 10**(od)*attenuation(om, h0, pb.zcmb.values[i], w0, wa, g)
		A.append(At)

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - A

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	
	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_pbd_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*10-10
	cube[5] = cube[5]*9-6

def llhood_pbd_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa, MB, od, g = [model_param[i] for i in range(7)]

	A = []
	for i in range(0,len(pb.zcmb.values)):
		At = 10**(od)*attenuation(om, h0, pb.zcmb.values[i], w0, wa, g)
		A.append(At)
	#A_inter = interp1d(z50, A)
	#DUST = A_inter(pb.zcmb.values)

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - A

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	mu = 0.308
	sigma = 0.012
	cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_pbd_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*10-10 #log od
	cube[6] = cube[6]*9-6 #gamma




#Attenuation including compton scattering on free electrons

def llhood_pbd_compton_flcdm(model_param, ndim, nparam):
	om, h0, MB, od, g, ob = [model_param[i] for i in range(6)]

	h = h0/100
	w0 = -1
	wa = 0
	A = []
	B = []
	for i in range(0,len(pb.zcmb.values)):
		At = 10**(od)*attenuation(om, h0, pb.zcmb.values[i], w0, wa, g)
		Bt = compt(om, h0, w0, wa, ob, pb.zcmb.values[i])
		A.append(At)
		B.append(Bt)
	

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - B - A

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))
	#gaussian prior on matter density parameter (Planck 2018)
	mu_cmb = 0.315
	sigma_cmb = 0.007
	cmb_prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_cmb))-0.5*(om-mu_cmb)**2/sigma_cmb**2
	#gaussian prior on baryon density parameter (Planck 2018)
	mu_ob = 0.0224
	sigma_ob = 0.0001
	ob_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma_ob))-0.5*(ob*h**2-mu_ob)**2/sigma_ob**2

	return -0.5*chisq + ob_prior + cmb_prior

def prior_pbd_compton_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2 #om
	cube[1] = cube[1]*50+50 #h0
	cube[2] = cube[2]*10-25 #MB
	cube[3] = cube[3]*10-10 #log od
	cube[4] = cube[4]*9-6 #gamma	
	cube[5] = cube[5]*0.5






#####################################################
################# ZTF + Pantheon ####################
#####################################################

def llhood_zp_flcdm(model_param, ndim, nparam):
	om, od, g = [model_param[i] for i in range(3)]

	h0 = 70
	A = []
	for k in range(0, len(z_zp)):
		At = 10**(od)*attenuation(om, h0, z_zp[k], -1, 0, g)
		A.append(At)

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(z_zp).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = mod1_zp - mod_th - A

	chisq = np.dot(hub_res.T, np.dot(C4, hub_res))

	if cmb == 1:
		mu = 0.3
		sigma = 0.005
		cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_zp_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*10-10
	cube[2] = cube[2]*9-6



###################################################
############### WFIRST + Foundation ###############
###################################################


def llhood_wf_flcdm(model_param, ndim, nparam):
	om, od, g = [model_param[i] for i in range(3)]

	h0 = 70
	A = []
	for k in range(0, len(z_wf)):
		At = 10**(od)*attenuation(om, h0, z_wf[k], -1, 0, g)
		A.append(At)

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(z_wf).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = mod1_wf - mod_th - A

	chisq = np.sum(hub_res**2/(wfirst.dmb.values)**2)

	if cmb == 1:
		mu = 0.3
		sigma = 0.005
		cmb_prior =  np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(om-mu)**2/sigma**2

	return -0.5*chisq + cmb_prior

def prior_wf_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]*2
	cube[1] = cube[1]*10-10
	cube[2] = cube[2]*9-6






### MultiNest
#uncomment whichever case is to be ran.
#only one case can be ran each time, clear the chains folder between runs.

npoints = 5000
start = time()
#FULL JLA
pmn.run(llhood_jlac_flcdm, prior_jlac_flcdm, 5, verbose = True, n_live_points = npoints)
#pmn.run(llhood_jlac_wcdm, prior_jlac_wcdm, 6, verbose = True, n_live_points = npoints)
#pmn.run(llhood_jlac_wzcdm, prior_jlac_wzcdm, 7, verbose = True, n_live_points = npoints)
#FULL JLA + DUST
#pmn.run(llhood_jlad_flcdm, prior_jlad_flcdm, 7, verbose = True, n_live_points = npoints)
#pmn.run(llhood_jlad_wcdm, prior_jlad_wcdm, 8, verbose = True, n_live_points = npoints)
#pmn.run(llhood_jlad_wzcdm, prior_jlad_wzcdm, 9, verbose = True, n_live_points = npoints)
#BINNED PANTHEON
#pmn.run(llhood_pbc_flcdm, prior_pbc_flcdm, 3, verbose = True, n_live_points = npoints)
#pmn.run(llhood_pbc_wcdm, prior_pbc_wcdm, 4, verbose = True, n_live_points = npoints)
#pmn.run(llhood_pbc_wzcdm, prior_pbc_wzcdm, 5, verbose = True, n_live_points = npoints)
#BINNED PANTHEON + DUST
#pmn.run(llhood_pbd_flcdm, prior_pbd_flcdm, 5, verbose = True, n_live_points = npoints)
#pmn.run(llhood_pbd_wcdm, prior_pbd_wcdm, 6, verbose = True, n_live_points = npoints)
#pmn.run(llhood_pbd_wzcdm, prior_pbd_wzcdm, 7, verbose = True, n_live_points = npoints)
#COMPTON
#pmn.run(llhood_pbd_compton_flcdm, prior_pbd_compton_flcdm, 6, verbose = True, n_live_points = npoints)
#ZTF + PANTHEON
#pmn.run(llhood_zp_flcdm, prior_zp_flcdm, 3, verbose = True, n_live_points = npoints)
#WFIRST + Foundation
#pmn.run(llhood_wf_flcdm, prior_wf_flcdm, 3, verbose = True, n_live_points = npoints)
end = time()
print('sampler time', end-start)