#corner plots of posteriors
#uncomment to load and plot
import numpy as np
import matplotlib.pyplot as plt
import corner

#legend
##clean = no dust term in hubble residual
####pbc = pantheon binned clean
####pbd = panhteon binned dust
###jlac = JLA clean
###jlad = JLA dust
######f = FlaLambdaCDM
######w = wCDM
#####wz = w0waCDM


###PANTHEON BINNED
###FLATLAMBDA
pbc_f_c0 = np.loadtxt('posteriors/pbin/pbc_flcdm_cmb0.dat')
pbc_f_c1 = np.loadtxt('posteriors/pbin/pbc_flcdm_cmb1.dat')
pbd_f_c0 = np.loadtxt('posteriors/pbin/pbd_flcdm_cmb0.dat')
pbd_f_c1 = np.loadtxt('posteriors/pbin/pbd_flcdm_cmb1.dat')

###WCDM
#pbc_w_c0 = np.loadtxt('posteriors/pbin/pbc_wcdm_cmb0.dat')
#pbc_w_c1 = np.loadtxt('posteriors/pbin/pbc_wcdm_cmb1.dat')
#pbd_w_c0 = np.loadtxt('posteriors/pbin/pbd_wcdm_cmb0.dat')
#pbd_w_c1 = np.loadtxt('posteriors/pbin/pbd_wcdm_cmb1.dat')

###W0WA
#pbc_wz_c0 = np.loadtxt('posteriors/pbin/pbc_wzcdm_cmb0.dat')
#pbc_wz_c1 = np.loadtxt('posteriors/pbin/pbc_wzcdm_cmb1.dat')
#pbd_wz_c0 = np.loadtxt('posteriors/pbin/pbd_wzcdm_cmb0.dat')
#pbd_wz_c1 = np.loadtxt('posteriors/pbin/pbd_wzcdm_cmb1.dat')


#JLA
#FLATLAMBDA
#jlac_f_c0 = np.loadtxt('posteriors/jla/jlac_flcdm_cmb0.dat')
#jlac_f_c1 = np.loadtxt('posteriors/jla/jlac_flcdm_cmb1.dat')
#jlad_f_c0 = np.loadtxt('posteriors/jla/jlad_flcdm_cmb0.dat')
#jlad_f_c1 = np.loadtxt('posteriors/jla/jlad_flcdm_cmb1.dat')

#WCDM
#jlac_w_c0 = np.loadtxt('posteriors/jla/jlac_wcdm_cmb0.dat')
#jlac_w_c1 = np.loadtxt('posteriors/jla/jlac_wcdm_cmb1.dat')
#jlad_w_c0 = np.loadtxt('posteriors/jla/jlad_wcdm_cmb0.dat')
#jlad_w_c1 = np.loadtxt('posteriors/jla/jlad_wcdm_cmb1.dat')

#W0WA
#jlac_wz_c0 = np.loadtxt('posteriors/jla/jlac_wzcdm_cmb0.dat')
#jlac_wz_c1 = np.loadtxt('posteriors/jla/jlac_wzcdm_cmb1.dat')
#jlad_wz_c0 = np.loadtxt('posteriors/jla/jlad_wzcdm_cmb0.dat')
#jlad_wz_c1 = np.loadtxt('posteriors/jla/jlad_wzcdm_cmb1.dat')







#labels for the different plots
label_pbc_f = ["$\Omega_m$", "$H_0$", "$M_B$"]
label_pbd_f = ["$\Omega_m$", "$H_0$", "$M_B$", "$\log_{10}\Omega_d$", "$\gamma$"]

label_pbc_w = ["$\Omega_m$", "$H_0$", "$w_0$", "$M_B$"]
label_pbd_w = ["$\Omega_m$", "$H_0$", "$w_0$", "$M_B$", "$\log_{10}\Omega_d$", "$\gamma$"]

label_pbc_wz = ["$\Omega_m$", "$H_0$", "$w_0$", "$w_a$", "$M_B$"]
label_pbd_wz = ["$\Omega_m$", "$H_0$", "$w_0$", "$w_a$", "$M_B$", "$\log_{10}\Omega_d$", "$\gamma$"]



label_jc_f = ["$\Omega_m$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$"]
label_jd_f = ["$\Omega_m$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$", "$\log_{10}\Omega_d$", "$\gamma$"]

label_jc_w = ["$\Omega_m$", "$w_0$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$"]
label_jd_w = ["$\Omega_m$", "$w_0$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$", "$\log_{10}\Omega_d$", "$\gamma$"]

label_jc_wz = ["$\Omega_m$", "$w_0$", "$w_a$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$"]
label_jd_wz = ["$\Omega_m$", "$w_0$", "$w_a$", "$\\alpha$", "$\\beta$", "$M_B$", "$\Delta_M$", "$\log_{10}\Omega_d$", "$\gamma$"]






###PANTHEON
###FLAT CLEAN
corner.corner(pbc_f_c0[:,:-1],labels = label_pbc_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
plt.title('PANTHEON CLEAN FLAT')
plt.show()
corner.corner(pbc_f_c1[:,:-1],labels = label_pbc_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
plt.title('PANTHEON CLEAN FLAT CMB')
plt.show()
###FLAT DUST
corner.corner(pbd_f_c0[:,:-1],labels = label_pbd_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
plt.title('PANTHEON DUST FLAT')
plt.show()
corner.corner(pbd_f_c1[:,:-1],labels = label_pbd_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
plt.title('PANTHEON DUST FLAT CMB')
plt.show()


###WCDM CLEAN
#corner.corner(pbc_w_c0[:,:-1],labels = label_pbc_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON CLEAN W')
#plt.show()
#corner.corner(pbc_w_c1[:,:-1],labels = label_pbc_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON CLEAN W CMB')
#plt.show()
###WCDM DUST
#corner.corner(pbd_w_c0[:,:-1],labels = label_pbd_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON DUST W')
#plt.show()
#corner.corner(pbd_w_c1[:,:-1],labels = label_pbd_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON DUST W CMB')
#plt.show()


###WZCDM CLEAN
#corner.corner(pbc_wz_c0[:,:-1],labels = label_pbc_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON CLEAN WZ')
#plt.show()
#corner.corner(pbc_wz_c1[:,:-1],labels = label_pbc_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON CLEAN WZ CMB')
#plt.show()
###WZCDM DUST
#corner.corner(pbd_wz_c0[:,:-1],labels = label_pbd_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON DUST WZ')
#plt.show()
#corner.corner(pbd_wz_c1[:,:-1],labels = label_pbd_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('PANTHEON DUST WZ CMB')
#plt.show()




#JLA
###FLAT CLEAN
#corner.corner(jlac_f_c0[:,:-1],labels = label_jc_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN FLAT')
#plt.show()
#corner.corner(jlac_f_c1[:,:-1],labels = label_jc_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN FLAT CMB')
#plt.show()
###FLAT DUST
#corner.corner(jlad_f_c0[:,:-1],labels = label_jd_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST FLAT')
#plt.show()
#corner.corner(jlad_f_c1[:,:-1],labels = label_jd_f, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST FLAT CMB')
#plt.show()


###WCDM CLEAN
#corner.corner(jlac_w_c0[:,:-1],labels = label_jc_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN W')
#plt.show()
#corner.corner(jlac_w_c1[:,:-1],labels = label_jc_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN W CMB')
#plt.show()
###WCDM DUST
#corner.corner(jlad_w_c0[:,:-1],labels = label_jd_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST W')
#plt.show()
#corner.corner(jlad_w_c1[:,:-1],labels = label_jd_w, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST W CMB')
#plt.show()


###WZCDM CLEAN
#corner.corner(jlac_wz_c0[:,:-1],labels = label_jc_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN WZ')
#plt.show()
#corner.corner(jlac_wz_c1[:,:-1],labels = label_jc_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA CLEAN WZ CMB')
#plt.show()
###WZCDM DUST
#corner.corner(jlad_wz_c0[:,:-1],labels = label_jd_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST WZ')
#plt.show()
#corner.corner(jlad_wz_c1[:,:-1],labels = label_jd_wz, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.title('JLA DUST WZ CMB')
#plt.show()






### ZTF + Pantheon

###offset refers to an added 0.01 mag to the ZTF sample relative to Pantheon

###no offset, with dust
#ztf1 = np.loadtxt('posteriors/zp/ztf_nooffset_dust.dat')
#zt1 = ztf1[:,[1,2]]
#lbl = ["$\log_{10}\Omega_d$", "$\gamma$"]
#trt = [np.log10(8e-5), -1]
#corner.corner(zt1, labels = lbl, truths = trt, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.show()

###offset, with dust
#ztf2 = np.loadtxt('posteriors/zp/ztf_offset_dust.dat')
#zt2 = ztf2[:,[1,2]]
#lbl = ["$\log_{10}\Omega_d$", "$\gamma$"]
#trt = [np.log10(8e-5), -1]
#corner.corner(zt2, labels = lbl, truths = trt, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.show()

###offset, without dust
#ztf3 = np.loadtxt('posteriors/zp/ztf_offset_clean.dat')
#zt3 = ztf3[:,[1,2]]
#lbl = ["$\log_{10}\Omega_d$", "$\gamma$"]
#trt = [np.log10(8e-5), -1]
#corner.corner(zt3, labels = lbl, plot_contours = False, plot_density = False, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.show()

###no offset, without dust
#ztf4 = np.loadtxt('posteriors/zp/ztf_nooffset_clean.dat')
#zt4 = ztf4[:,[1,2]]
#lbl = ["$\log_{10}\Omega_d$", "$\gamma$"]
#trt = [np.log10(8e-5), -1]
#corner.corner(zt4, labels = lbl, plot_contours = False, plot_density = False, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.show()



### WFIRST + Foundation


#wfirst = np.loadtxt('posteriors/wfirst/wfirst.dat')
#lblw = ["$\Omega_m$", "$\log_{10}\Omega_d$", "$\gamma$"]
#trt = [0.3, np.log10(8e-5), -1]
#corner.corner(wfirst[:,:-1], labels = lblw, truths = trt, plot_contours = True, plot_density = True, levels=(1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
#plt.show()
