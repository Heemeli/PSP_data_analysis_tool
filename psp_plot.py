
#Tab size preference: 4

import pyspedas
from hyperlink import URL
import datetime
import time
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import ticker
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy import constants

url	= URL.from_text(u'https://cdaweb.gsfc.nasa.gov/sc_inventory_plots/')
def ts(): #timestamps
	ts	= datetime.datetime.fromtimestamp(time.time()).strftime('%d-%b-%y %H:%M:%S:')
	return ts

#Variables to plot. True = plot, False = no plot. RTN B and v are plotted together.

#SWEAP and FIELDS products
BR = RVEL	= True #Magnetic field and proton speed, radial
BT = TVEL	= False #Magnetic field and proton speed, tangential
BN = NVEL	= False #Magnetic field and proton speed, normal
MAGB		= False #Magnetic field strength
DENS		= False #Solar wind proton and electron densities
AVEL		= False #Alfvén speed
AMACH		= True #Alfvén Mach number
PAD			= True #Electron pitch angle distribution for specific energy bin
EED			= False #Electron energy distribution 2eV-30keV
PED			= True #Proton energy distribution 2eV-30keV
HELIODIST	= True #Heliocentric distance

SPEFLAG		= False #SPAN-E quality flag
SPIFLAG 	= False #SPAN-I quality flag
SPCFLAG		= False #SPC quality flag
FIELDSFLAG	= False #FIELDS quality flag

#ISOIS products
#Proton data:
#EPILO IC 20keV–15MeV; LET1 1-20MeV; HET ->100MeV
#Electron data:
#EPILO IC 25–1000keV; LET1 0.5–2MeV; HET ->6MeV
#Ion data (nuc>1):
#EPILO IC 20keV/nuc–15MeV/nuc; LET1 1-20MeV/nuc; HET ->100MeV/nuc
HET			= False
HETFLAG		= False #EPI-Hi HET quality flag
LET1		= True
LET1FLAG	= False #EPI-Hi LET1 quality flag
EPILO		= False
EPILOFLAG	= False #EPI-Lo quality flag

H			= True #Protons
E			= True #Electrons
He			= False
C			= False
N			= False
O			= False
Ne			= False
Na			= False #EPI-Hi only
Mg			= False
Al			= False
Si			= False
S			= False #EPI-Hi only
Ar			= False #EPI-Hi only
Ca			= False #EPI-Hi only
Cr			= False #EPI-Hi only
Fe			= False
Ni			= False #EPI-Hi only

#Options
NORM_BRTN	= False #If true, RTN components of B will be normalized to au values (B*au^2)
EPIHI_ENC	= True #(EPI-Hi) If True, 1 minute encounter data will be downloaded instead of 1 hour cruise data 
SPI_OPT		= True #Choose between SPAN-I and SPC for plasma data. True = SPI, False = SPC.

#Plot cosmetic options
hspace			= 0.0 #Vertical space between subplots. Plots cannot touch as long as 'constrained_layout' is opted 
labelsize		= 15 #Plot label sizes
ticklabelsize 	= 15 #Plot tick label sizes
font			= 'Times New Roman' #Plot font
dist_tick_freq	= 4000 #Every how often heliodist ticks are placed. Note: cadence varies with the orbit phase

#User input
print('\nPlease ensure data-availability at', url, 'before plotting.')
print('Give start date, press enter, and give end date. Use format "YYYY-MM-DD": \n')
Start	= input()
End		= input()
print('\nSelected time period: ', Start, '-', End)
if PAD:
	print('Give approximate electron energy to be studied in the PAD. Valid energies 0-4000 eV.\n')
	enumber = input()
print('\n'+ts(), 'Downloading data from CDAWeb:', '\n')

########################################Downloading data########################################

#Data products from SWEAP suite
SWEAP		= [PAD, EED, PED, HELIODIST, (RVEL, TVEL, NVEL), DENS]
if any(SWEAP) or AVEL or AMACH or NORM_BRTN or SPIFLAG or SPEFLAG or SPCFLAG:
	SPI				= pyspedas.psp.spi(trange=[Start, End], datatype='spi_sf00_l3_mom',
			 	      notplot=True, level='l3', time_clip=True)
	heliodist_x		= SPI['psp_spi_SUN_DIST']['x'] #Time
	heliodist_y		= SPI['psp_spi_SUN_DIST']['y'] #Heliocentric distance
	PED_x			= SPI['psp_spi_EFLUX_VS_ENERGY']['x'] #Time
	PED_y			= SPI['psp_spi_EFLUX_VS_ENERGY']['y'] #Proton intensity array
	PED_v			= SPI['psp_spi_EFLUX_VS_ENERGY']['v'][0] #Proton energy bins array
	spiflag_x		= SPI['psp_spi_QUALITY_FLAG']['x']
	spiflag_y		= SPI['psp_spi_QUALITY_FLAG']['y']
	if PAD or EED or SPEFLAG:
		SPE			= pyspedas.psp.spe(trange=[Start, End], datatype='spe_sf0_pad',
				  	  get_support_data=False, notplot=True, level='l3', time_clip=True)
		PAD_x		= SPE['psp_spe_EFLUX_VS_PA_E']['x'] #Time
		PAD_y		= SPE['psp_spe_EFLUX_VS_PA_E']['y'] #Intensity array
		PAD_v1		= SPE['psp_spe_EFLUX_VS_PA_E']['v1'][0] #Pitch angles
		PAD_v2		= SPE['psp_spe_EFLUX_VS_PA_E']['v2'][0] #Electron energies
		EED_x		= SPE['psp_spe_EFLUX_VS_ENERGY']['x'] #Time
		EED_y		= SPE['psp_spe_EFLUX_VS_ENERGY']['y'] #Intensity array
		EED_v		= SPE['psp_spe_EFLUX_VS_ENERGY']['v'][0] #Electron energies
		speflag_x	= SPE['psp_spe_QUALITY_FLAG']['x']
		speflag_y	= SPE['psp_spe_QUALITY_FLAG']['y']
	if not(SPI_OPT) or SPCFLAG:
		SPC			= pyspedas.psp.spc(trange=[Start, End], datatype='l3i',
					  get_support_data=True, notplot=True, level='l3', time_clip=True)
		VRTN_x		= SPC['psp_spc_vp_moment_RTN']['x'] #Time
		VRTN_y		= SPC['psp_spc_vp_moment_RTN']['y'] #RTN proton speed, SPC
		protdens_x	= SPC['psp_spc_np_moment']['x'] #Time
		protdens_y	= SPC['psp_spc_np_moment']['y'] #Proton density, SPC
		spcflag_x	= SPC['psp_spc_general_flag']['x']
		spcflag_y	= SPC['psp_spc_general_flag']['y']
		Vlabel		= 'SPC'
	else:
		VRTN_x		= SPI['psp_spi_VEL_RTN_SUN']['x'] #Time
		VRTN_y		= SPI['psp_spi_VEL_RTN_SUN']['y'] #RTN speed, SPI
		protdens_x	= SPI['psp_spi_DENS']['x'] #Time
		protdens_y	= SPI['psp_spi_DENS']['y'] #Proton density, SPI
		Vlabel		= 'SPI'

#Data products from FIELDS suite
FIELDS		= [(BR, BT, BN, MAGB), DENS]
if any(FIELDS) or AVEL or AMACH or FIELDSFLAG:
	FIELDS_B	= pyspedas.psp.fields(trange=[Start, End], datatype='mag_RTN_4_per_cycle',
			  	  get_support_data=True, notplot=True, level='l2', time_clip=True)
	FIELDS_QTN	= pyspedas.psp.fields(trange=[Start, End], datatype='sqtn_rfs_v1v2',
				  varnames=['electron_density'], notplot=True, level='l3', time_clip=True)
	QTN 		= 'electron_density'
	ELECDENS	= True
	if FIELDS_QTN == {}:
		FIELDS_QTN	= pyspedas.psp.fields(trange=[Start, End], datatype='rfs_lfr_qtn',
					  varnames=['N_elec'], notplot=True, level='l3', time_clip=True)
		QTN			= 'N_elec'
		if FIELDS_QTN == {}:
			print(ts, 'No QTN data found. Unable to get electron density.')
			ELECDENS = False
	FIELDSdict	= FIELDS_B | FIELDS_QTN
	BRTN_x		= FIELDSdict['psp_fld_l2_mag_RTN_4_Sa_per_Cyc']['x'] #Time
	BRTN_y		= FIELDSdict['psp_fld_l2_mag_RTN_4_Sa_per_Cyc']['y'] #RTN magnetic flux density
	elecdens_x	= FIELDSdict[f'{QTN}']['x'] #Time
	elecdens_y	= FIELDSdict[f'{QTN}']['y'] #Electron density
	fieldsflag_x= FIELDSdict['psp_fld_l2_quality_flags']['x']
	fieldsflag_y= FIELDSdict['psp_fld_l2_quality_flags']['y']

#Data products from ISOIS suite
ISOIS		= [H, E, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Cr, Fe, Ni]
			  #['HET/LET1 name specifier', 'EPILO channel specifier', 'EPILO name specifier']
ISOISnames	= [['H','ChanP','H'], ['Electrons','ChanE','Electron'], ['He','ChanC','He4'],
			   ['C','ChanD','C'], ['N','ChanD','N'], ['O','ChanC','O'], ['Ne','ChanD','Ne'],
			   ['Na',None,None], ['Mg','ChanD','Mg'], ['Al',None,None], ['Si','ChanD','Si'],
			   ['S',None,None], ['Ar',None,None], ['Ca',None,None], ['Cr',None,None],
			   ['Fe','ChanC','Fe'], ['Ni',None,None]]
ISOISvars	= list(zip(ISOIS, ISOISnames))
False_list = [] #List for appending "bad" variables to. This might be used for skipping over said variables 
if any(ISOIS) or HETFLAG or LET1FLAG or EPILOFLAG:
	if HET or HETFLAG:
		hetdtype = 'het_rates1h'
		if EPIHI_ENC: hetdtype = 'het_rates1min'
		HETd		= pyspedas.psp.epihi(trange=[Start, End], datatype=hetdtype,
					  get_support_data=True, notplot=True, level='l2', time_clip=True)
		HETdict	= {}
		if HETFLAG:
			hetflag_x	= HETd['psp_epihi_Quality_Flag']['x']
			hetflag_y	= HETd['psp_epihi_Quality_Flag']['y']
		for i in ISOISvars:
			if i[0]:
				if np.count_nonzero(HETd[f'psp_epihi_A_{i[1][0]}_Rate']['y']) == 0: #Invalidate element if rate array is filled with zeros
					print(ts(), f'No EPI-Hi data found for element {i[1][0]}.')
					False_list.append(i[1][0])
				else:
					HETdict[f'HET_{i[1][0]}time']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['x'] #Time
					HETdict[f'HET_{i[1][0]}rate']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['y'] \
													+ HETd[f'psp_epihi_B_{i[1][0]}_Rate']['y'] #Count rate array
					HETdict[f'HET_{i[1][0]}ergs']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['v'] #Energy bins array
	if LET1 or LET1FLAG:
		let1dtype = 'let1_rates1h'
		if EPIHI_ENC: let1dtype = 'let1_rates1min'
		LET1d		= pyspedas.psp.epihi(trange=[Start, End], datatype=let1dtype, 
				      get_support_data=True, notplot=True, level='l2', time_clip=True)
		LET1dict	= {}
		if LET1FLAG:
			let1flag_x	= LET1d['psp_epihi_Quality_Flag']['x']
			let1flag_y	= LET1d['psp_epihi_Quality_Flag']['y']
		for i in ISOISvars:
			if i[0]:
				if np.count_nonzero(LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['y']) == 0: #Invalidate element if rate array is filled with zeros
					print(ts(), f'No EPI-Hi data found for element {i[1][0]}.')
					False_list.append(i[1][0])
				else:
					LET1dict[f'LET1_{i[1][0]}time']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['x'] #Time
					LET1dict[f'LET1_{i[1][0]}rate']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['y'] \
													+ LET1d[f'psp_epihi_B_{i[1][0]}_Rate']['y'] #Count rate array
					LET1dict[f'LET1_{i[1][0]}ergs']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['v'] #Energy bins array
	if EPILO or EPILOFLAG:
		ICv		= pyspedas.psp.epilo(trange=[Start, End], datatype='ic',
				  get_support_data=EPILOFLAG, notplot=True, level='l2', time_clip=True)
		PEv		= pyspedas.psp.epilo(trange=[Start, End], datatype='pe',
				  get_support_data=EPILOFLAG, notplot=True, level='l2', time_clip=True)
		EPILOd		= ICv | PEv #Merge IC and PE dictionaries
		EPILOdict	= {}
		if EPILOFLAG:
			chanPflag_x	= EPILOd['psp_epilo_Quality_Flag_ChanP']['x']
			chanPflag_y	= EPILOd['psp_epilo_Quality_Flag_ChanP']['y']
			chanCflag_x	= EPILOd['psp_epilo_Quality_Flag_ChanC']['x']
			chanCflag_y	= EPILOd['psp_epilo_Quality_Flag_ChanC']['y']
			chanDflag_x	= EPILOd['psp_epilo_Quality_Flag_ChanD']['x']
			chanDflag_y	= EPILOd['psp_epilo_Quality_Flag_ChanD']['y']
			chanEflag_x	= EPILOd['psp_epilo_Quality_Flag_ChanE']['x']
			chanEflag_y	= EPILOd['psp_epilo_Quality_Flag_ChanE']['y']
		for i in ISOISvars:
			if i[0]:
				if i[1][2] == None:
					continue
				if len(set(EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['v2'][0,0])) <= 1: #Invalidate element if ergs array is filled with -1.e+31
					print(ts(), f'No EPI-Lo data found for element {i[1][2]}.')
					False_list.append(i[1][2])
				else:
					EPILOdict[f'EPILO_{i[1][2]}time']	= EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['x'] #Time
					EPILOdict[f'EPILO_{i[1][2]}ergs']	= EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['v2'][0,0] #Energy bins array
					for j in reversed(range(len(EPILOdict[f'EPILO_{i[1][2]}ergs']))): #Removing high-end energy bins which might be unused (indicated with values of -9.9999998e+30)
						if EPILOdict[f'EPILO_{i[1][2]}ergs'][j] < 0:
							EPILOdict[f'EPILO_{i[1][2]}ergs']	= np.delete(EPILOdict[f'EPILO_{i[1][2]}ergs'], j)
						else:
							continue
						EPILOdict[f'EPILO_{i[1][2]}rate']	= np.nan_to_num(EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['y'], nan=0) # Ion count rate array
						EPILOdict[f'EPILO_{i[1][2]}rate']	= np.mean(EPILOdict[f'EPILO_{i[1][2]}rate'], axis=1) #Mean from the 80 apertures (look directions), or 8 wedges in the case of electrons
						EPILOdict[f'EPILO_{i[1][2]}rate'] 	= EPILOdict[f'EPILO_{i[1][2]}rate'][:, :len(EPILOdict[f'EPILO_{i[1][2]}ergs'])] #Truncating energy axis to match the energy bins in ICprotergs:

print(ts(), 'Data loaded. Plotting...')

########################################Calculations########################################

#Which (nearest) PAD electron energy to plot. Valid indices 0-31.
if PAD:
	def find_nearest(array, value):
		idx = (np.abs(array - np.float32(value))).argmin()
		return array[idx]
	n = list(PAD_v2).index(np.float32(find_nearest(PAD_v2, enumber)))
 
if NORM_BRTN or MAGB or AVEL or AMACH:
	absB			= np.linalg.norm(BRTN_y, axis=1) #Absolute B
	if NORM_BRTN or AVEL or AMACH:
		#Conversion of distance resolution to B resolution by interpolation
		f1 = scipy.interpolate.interp1d(np.float64(heliodist_x), heliodist_y, fill_value='extrapolate')
		ip_heliodist	= f1(np.float64(BRTN_x))

x = 1
if NORM_BRTN: x = (ip_heliodist/149597871)**2

#Alfvén speed and Alfvén MACH number
if AVEL or AMACH:
	#Conversion of density resolution to B resolution by interpolation
	f2p = scipy.interpolate.interp1d(np.float64(protdens_x), protdens_y, fill_value='extrapolate')
	ip_protdens = np.array(f2p(np.float64(BRTN_x)))
	Av_p = 10E-9*absB/(1000*np.sqrt(4*np.pi*10E-7*constants.proton_mass*10E6*ip_protdens)) #Alfvén speed from proton density
	if ELECDENS:
		f2e = scipy.interpolate.interp1d(np.float64(elecdens_x), elecdens_y, fill_value='extrapolate')
		ip_elecdens = np.array(f2e(np.float64(BRTN_x)))
		Av_e = 10E-9*absB/(1000*np.sqrt(4*np.pi*10E-7*constants.proton_mass*10E6*ip_elecdens)) #Alfvén speed from electron density
	if AMACH:
		#Conversion of vR resolution to B resolution by interpolation
		f4 = scipy.interpolate.interp1d(np.float64(VRTN_x), VRTN_y[:,0], fill_value='extrapolate')
		ip_vR = f4(np.float64(BRTN_x))
		Mach_p = ip_vR/Av_p #(radial) Alfvén MACH number from proton density
		if ELECDENS:
			Mach_e = ip_vR/Av_e #(radial) Alfvén MACH number from electron density

########################################Plotting functions########################################

#Function for conventional x,y plots
def xy_plot(x, y, label, color, tag=None, log=False, twinx=False, previousplot=False):
	global index_count
	if previousplot:
		index_count    -=1
	if twinx:
		index_count	   -= 1
		twinx	 		= ax[index_count].twinx()
		twinx.plot(x, y, color=color, label=tag, alpha=0.8)
		twinx.set_ylabel(label, fontsize=labelsize, color=color)
		twinx.tick_params(axis='both', which='major', labelsize=ticklabelsize)
		if log: twinx.set_yscale('log')
	else:
		ax[index_count].plot(x, y, color=color, label=tag)
		ax[index_count].set_ylabel(label, fontsize=labelsize, color=color)
		ax[index_count].tick_params(axis='both', which='major', labelsize=ticklabelsize)
		if tag: ax[index_count].legend(ncol=2, frameon=False)
		if log: ax[index_count].set_yscale('log')
	index_count += 1

#Function for plotting spectrograms
def specgram_plot(time, ergs, rate, label, element, log=True, intensity=False):
	global index_count
	if intensity:
		cbarlabel 	= f'{element}\neV/cm²/s/sr/eV'
		ylabel 		= f'{label}'
	else:
		cbarlabel 	= f'{element}\ncounts/s'
		ylabel 		= f'{element}\n{label}'
	c = ax[index_count].pcolormesh(time, ergs, rate.T, shading='auto', cmap='turbo', norm=LogNorm())
	cbar = plt.colorbar(c, ax=ax[index_count])
	cbar.set_label(cbarlabel, fontsize=labelsize)
	ax[index_count].set_ylabel(ylabel, fontsize=labelsize)
	ax[index_count].tick_params(axis='both', which='major', labelsize=ticklabelsize)
	if log: ax[index_count].set_yscale('log')
	index_count += 1

########################################Plotting########################################
Vars 		= [PAD, EED, PED, BR, BT, BN, MAGB, DENS, AVEL, AMACH, SPEFLAG, SPIFLAG, SPCFLAG, FIELDSFLAG, HETFLAG, LET1FLAG, 4*EPILOFLAG] \
			+ ISOIS*sum((HET, LET1, EPILO))
plots_count = sum(Vars)

#User variable misuse warning
if plots_count == 0:
	print('No variables are selected for plotting. Please select which variables to plot in the source code.')
	exit()

plt.rcParams['font.family'] = font #Plot font
plt.rcParams['mathtext.fontset'] = 'stix' #Math fontset

if plots_count == 1: #plt.subplots() does not allow subscripting when plotting only one plot
	fig, ax = plt.subplots(nrows = 2, sharex=False, constrained_layout=True, gridspec_kw={'hspace':hspace})
	ax[1].axis('off')
else:
	fig, ax = plt.subplots(nrows = plots_count, sharex=True, constrained_layout=True, gridspec_kw={'hspace':hspace})

index_count = 0

#SWEAP and FIELDS products
				  #(x,		y,				label,									color, **kwargs)
if BR:		xy_plot(BRTN_x, x*BRTN_y[:,0],	r'$\mathit{B}_\text{R}$',				'blue')
if RVEL:	xy_plot(VRTN_x, VRTN_y[:,0],	r'$\mathit{v}_\text{R}$'+'\n'+Vlabel,	'orange', twinx=True) 
if BT: 		xy_plot(BRTN_x, x*BRTN_y[:,1],	r'$\mathit{B}_\text{T}$',				'blue')
if TVEL:	xy_plot(VRTN_x, VRTN_y[:,1],	r'$\mathit{v}_\text{T}$'+'\n'+Vlabel,	'orange', twinx=True)
if BN: 		xy_plot(BRTN_x, x*BRTN_y[:,2],	r'$\mathit{B}_\text{N}$',				'blue')
if NVEL: 	xy_plot(VRTN_x, VRTN_y[:,2],	r'$\mathit{v}_\text{N}$'+'\n'+Vlabel,	'orange', twinx=True)
if MAGB: 	xy_plot(BRTN_x, absB,			r'$\mathit{B}$','blue')
if DENS:
	xy_plot(protdens_x, protdens_y, r'$\mathit{n}$ [1/cm³]'+'\n'+Vlabel, 'black', tag=r'$\mathit{n}_\text{p}$', log=True)
	if ELECDENS: xy_plot(BRTN_x, ip_elecdens, r'$\mathit{n}$ [1/cm³]', 'orange', tag=r'$\mathit{n}_\text{e}$', log=True, previousplot=True)
if AVEL:
	xy_plot(BRTN_x, Av_p, r'$\mathit{v}_\text{A}$', 'black')
	if ELECDENS: xy_plot(BRTN_x, Av_e, r'$\mathit{v}_\text{A}$', 'orange', previousplot=True)
if AMACH:
	xy_plot(BRTN_x, Mach_p, 'Mach', 'black', log=True)
	if ELECDENS: xy_plot(BRTN_x, Mach_e, 'Mach', 'orange', log=True, previousplot=True)
	ax[index_count-1].hlines(y=1, xmin=BRTN_x[0], xmax=BRTN_x[-1], color='red', linestyles='--') #Mach-1 limit
if PAD: specgram_plot(PAD_x, PAD_v1, PAD_y[:,:,n], 'PA','%.2f' % PAD_v2[n], log=False, intensity=True)
if EED: specgram_plot(EED_x, EED_v, EED_y, 'Energy\n[eV]', 'Electron', intensity=True)
if PED: specgram_plot(PED_x, PED_v, PED_y, 'Energy\n[eV]', 'Proton', intensity=True)

#ISOIS products
for i in ISOISvars:
	if i[0]:
		#specgram_plot(time, ergs, intensity, label, element)
		if HET:
			if i[1][0] in False_list:
				print(ts(), f'Unable to plot element {i[1][0]}.')
			else:
				if i[1][0] == 'Electrons':
					label 	= 'HET (MeV)'
				else: label = 'HET (MeV/nuc)'
				specgram_plot(HETdict[f'HET_{i[1][0]}time'], HETdict[f'HET_{i[1][0]}ergs'], HETdict[f'HET_{i[1][0]}rate'], label, i[1][0])	
		if LET1:
			if i[1][0] in False_list:
				print(ts(), f'Unable to plot element {i[1][0]}.')
			else:
				if i[1][0] == 'Electrons':
					label 	= 'LET1 (MeV)'
				else: label = 'LET1 (MeV/nuc)'
				specgram_plot(LET1dict[f'LET1_{i[1][0]}time'], LET1dict[f'LET1_{i[1][0]}ergs'], LET1dict[f'LET1_{i[1][0]}rate'], label, i[1][0])
		if EPILO:
			if (i[1][0] in False_list) or (i[1][2] == None):
				print(ts(), f'Unable to plot element {i[1][0]}.')
			else:
				if i[1][0] == 'Electrons':
					label 	= 'EPILO (keV)'
				else: label = 'EPILO (keV/nuc)'
				specgram_plot(EPILOdict[f'EPILO_{i[1][2]}time'], EPILOdict[f'EPILO_{i[1][2]}ergs'], EPILOdict[f'EPILO_{i[1][2]}rate'], label, i[1][0])
#Quality flags
if SPEFLAG:		xy_plot(speflag_x, speflag_y, 'SPE\nflag', 'black')
if SPIFLAG:		xy_plot(spiflag_x, spiflag_y, 'SPI\nflag', 'black')
if SPCFLAG:		xy_plot(spcflag_x, spcflag_y, 'SPC\nflag', 'black')
if FIELDSFLAG:	xy_plot(fieldsflag_x, fieldsflag_y, 'FIELDS\nflag', 'black')
if HETFLAG:		xy_plot(hetflag_x, hetflag_y, 'HET\nflag', 'black')
if LET1FLAG:	xy_plot(let1flag_x, let1flag_y, 'LET1\nflag', 'black')
if EPILOFLAG:
	xy_plot(chanPflag_x, chanPflag_y, 'Channel P\nflag', 'black')
	xy_plot(chanCflag_x, chanCflag_y, 'Channel C\nflag', 'black')
	xy_plot(chanDflag_x, chanDflag_y, 'Channel D\nflag', 'black')
	xy_plot(chanEflag_x, chanEflag_y, 'Channel E\nflag', 'black')

#Heliocentric distance
if HELIODIST:
	dist = ax[0].twiny()
	dist.set_xlim(ax[0].get_xlim())
	dist.set_xticks(heliodist_x[::dist_tick_freq])
	dist.set_xticklabels(np.round(heliodist_y[::dist_tick_freq]/696340, 1))
	dist.set_xlabel("Heliocentric distance [Solar radii]", fontsize=labelsize)
	dist.tick_params(axis='both', which='major', labelsize=ticklabelsize)

fig.align_labels()
fig.align_titles()

plt.show()