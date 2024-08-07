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
BR = RVEL	= False #Magnetic field and proton speed, radial
BT = TVEL	= False #Magnetic field and proton speed, tangential
BN = NVEL	= False #Magnetic field and proton speed, normal
MAGB		= False #Magnetic field strength
DENS		= False #Solar wind proton and electron densities
AVEL		= False #Alfvén speed
AMACH		= False #Alfvén Mach number
PAD			= False #Electron pitch angle distribution for specific energy bin
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
HET			= True
HETFLAG		= False #EPI-Hi HET quality flag
LET1		= False
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
Na			= False #EPI-HI only
Mg			= False
Al			= True
Si			= False
S			= False #EPI-HI only
Ar			= True #EPI-HI only
Ca			= False #EPI-HI only
Cr			= False #EPI-HI only
Fe			= True
Ni			= False #EPI-HI only

#Options
NORM_BRTN	= True #If true, RTN components of B will be normalized to au values (B*au^2)
EPIHI_ENC	= True #(EPI-HI) If True, 1 minute encounter data will be downloaded instead of 1 hour cruise data 
SPI_OPT		= True #Choose between SPAN-I and SPC for plasma data. True = SPI, False = SPC.

#Plot cosmetic options
hspace			= 0.0 #Vertical space between subplots. Plots cannot touch as long as 'constrained_layout' is opted 
labelsize		= 15 #Plot label sizes
ticklabelsize 	= 10 #Plot tick label sizes
font			= 'Times New Roman' #Plot font style

#Plotting variables
SWEAP		= [PAD, EED, PED, HELIODIST, (RVEL, TVEL, NVEL), DENS]
FIELDS		= [(BR, BT, BN, MAGB), DENS]
ISOIS		= [H, E, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Cr, Fe, Ni]
FLAGS		= [SPEFLAG, SPIFLAG, SPCFLAG, FIELDSFLAG, HETFLAG, LET1FLAG, EPILOFLAG, EPILOFLAG, EPILOFLAG, EPILOFLAG]

Vars 		= [PAD, EED, PED, BR, BT, BN, MAGB, DENS, AVEL, AMACH, SPEFLAG, SPIFLAG, SPCFLAG, FIELDSFLAG, HETFLAG, LET1FLAG, EPILOFLAG] \
			+ ISOIS*sum((HET, LET1, EPILO))
plots_count = sum(Vars)
			  #['instrument', 'data specifier']
SWEAPnames	= [['spe','EFLUX_VS_PA_E'], ['spe','EFLUX_VS_ENERGY'], ['spi','EFLUX_VS_ENERGY'], ['spi','SUN_DIST'],
			   ['spi','VEL_RTN_SUN'], ['spi','DENS'], ['spc','vp_moment_RTN'], ['spc','np_moment']]
if not(SPI_OPT):
	SWEAPnames.remove(['spi','VEL_RTN_SUN'])
	SWEAPnames.remove(['spi','DENS'])
SWEAPvars	= list(zip(SWEAP, SWEAPnames))
			  #['variable','data specifier']
FIELDSnames	= [['BRTN','psp_fld_l2_mag_RTN_4_Sa_per_Cyc'], ['elecdens','electron_density']]
FIELDSvars	= list(zip(FIELDS, FIELDSnames))
			  #['HET/LET1 name specifier', 'EPILO channel specifier', 'EPILO name specifier']
ISOISnames	= [['H','ChanP','H'], ['Electrons','ChanE','Electron'], ['He','ChanC','He4'],
			   ['C','ChanD','C'], ['N','ChanD','N'], ['O','ChanC','O'], ['Ne','ChanD','Ne'],
			   ['Na',None,None], ['Mg','ChanD','Mg'], ['Al',None,None], ['Si','ChanD','Si'],
			   ['S',None,None], ['Ar',None,None], ['Ca',None,None], ['Cr',None,None],
			   ['Fe','ChanC','Fe'], ['Ni',None,None]]
ISOISvars	= list(zip(ISOIS, ISOISnames))

SWEAPd = FIELDSd = HETd = LET1d = EPILOd = {} #Initializing data loading dictionaries
False_list = [] #List for appending "bad" variables to. This might be used for skipping over said variables 

#User variable misuse warning
if plots_count == 0:
	print('No variables are selected for plotting. Please select which variables to plot in the source code.')
	exit()

#User input
print('\nPlease ensure data-availability at', url, 'before plotting.')
print('Give start date, press enter, and give end date. Use format "YYYY-MM-DD": \n')
Start	= '2022-06-05'#input()
End		= '2022-06-07'#input()
print('\nSelected time period: ', Start, '-', End)
if PAD:
	print('Give approximate electron energy to be studied in the PAD. Valid energies 0-4000 eV.\n')
	enumber = 300#input()
print(ts(), 'Downloading data from CDAWeb:', '\n')

#Data products from SWEAP suite
if any(SWEAP) or AVEL or AMACH or NORM_BRTN:
	SWEAPd		= pyspedas.psp.spi(trange=[Start, End], datatype='spi_sf00_l3_mom',
			 	  notplot=True, level='l3', time_clip=True)
	if PAD or EED:
		SPE			= pyspedas.psp.spe(trange=[Start, End], datatype='spe_sf0_pad',
				  	  get_support_data=False, notplot=True, level='l3', time_clip=True)
		SWEAPd		= SWEAPd | SPE
	if not(SPI_OPT):
		SPC		= pyspedas.psp.spc(trange=[Start, End], datatype='l3i',
				  get_support_data=SPCFLAG, notplot=True, level='l3', time_clip=True)
		SWEAPd		= SWEAPd | SPC
	SWEAPdict	= {}
	for i in SWEAPvars:
		if i[0] or i[1][1] == 'DENS' or i[1][1] == 'np_moment' or i[1][1] == 'SUN_DIST':
			SWEAPdict[f'{i[1][0]}_{i[1][1]}_x']		= SWEAPd[f'psp_{i[1][0]}_{i[1][1]}']['x'] #x (time) arrays
			SWEAPdict[f'{i[1][0]}_{i[1][1]}_y']		= SWEAPd[f'psp_{i[1][0]}_{i[1][1]}']['y'] #y arrays
			if i[1][1] == 'EFLUX_VS_PA_E': #PAD
				SWEAPdict[f'{i[1][0]}_{i[1][1]}_v1']	= SWEAPd[f'psp_{i[1][0]}_{i[1][1]}']['v1'][0] #Pitch angles
				SWEAPdict[f'{i[1][0]}_{i[1][1]}_v2']	= SWEAPd[f'psp_{i[1][0]}_{i[1][1]}']['v2'][0] #Electron energies
				#Which (nearest) PAD electron energy to plot. Valid indices 0-31.
				def find_nearest(array, value):
					idx = (np.abs(array - np.float32(value))).argmin()
					return array[idx]
				n = list(SWEAPdict[f'{i[1][0]}_{i[1][1]}_v2']).index(np.float32(find_nearest(SWEAPdict[f'{i[1][0]}_{i[1][1]}_v2'], enumber)))
			if i[1][1] == 'EFLUX_VS_ENERGY': #EED or PED
				SWEAPdict[f'{i[1][0]}_{i[1][1]}_v']	= SWEAPd[f'psp_{i[1][0]}_{i[1][1]}']['v'][0]

#Data products from FIELDS suite
if any(FIELDS) or AVEL or AMACH:
	FIELDSd		= pyspedas.psp.fields(trange=[Start, End], datatype='mag_RTN_4_per_cycle',
			  	  get_support_data=FIELDSFLAG, notplot=True, level='l2', time_clip=True)
	FIELDS_QTN	= pyspedas.psp.fields(trange=[Start, End], datatype='sqtn_rfs_v1v2', level='l3',
				  varnames=['electron_density'], notplot=True, time_clip=True)
	FIELDSdict	= FIELDSd | FIELDS_QTN
	ELECDENS	= True
	if FIELDS_QTN == {}:
		FIELDS_QTN	= pyspedas.psp.fields(trange=[Start, End], datatype='rfs_lfr_qtn',
						varnames=['N_elec'], notplot=True, level='l3', time_clip=True)
		FIELDSd	= FIELDSd | FIELDS_QTN
		FIELDSvars[1][1][1] = 'N_elec'
		if FIELDS_QTN == {}:
			print(ts, 'No QTN data found. Unable to get electron density.')
			ELECDENS = False
	for i in FIELDSvars:
		FIELDSdict[f'{i[1][0]}_x']	= FIELDSdict[f'{i[1][1]}']['x']
		FIELDSdict[f'{i[1][0]}_y']	= FIELDSdict[f'{i[1][1]}']['y']

#Data products from ISOIS suite
if any(ISOIS):
	if HET:
		hetdtype = 'het_rates1h'
		if EPIHI_ENC: hetdtype = 'het_rates1min'
		HETd		= pyspedas.psp.epihi(trange=[Start, End], datatype=hetdtype,
					  get_support_data=True, notplot=True, level='l2', time_clip=True)
		HETdict	= {}
		for i in ISOISvars:
			if i[0]:
				if np.count_nonzero(HETd[f'psp_epihi_A_{i[1][0]}_Rate']['y']) == 0: #Invalidate element if rate array is filled with zeros
					print(ts(), f'No EPI-Hi data found for element {i[1][0]}.')
					False_list.append(i[1][0])
				else:
					HETdict[f'HET_{i[1][0]}time']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['x']
					HETdict[f'HET_{i[1][0]}rate']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['y'] \
													+ HETd[f'psp_epihi_B_{i[1][0]}_Rate']['y']
					HETdict[f'HET_{i[1][0]}ergs']	= HETd[f'psp_epihi_A_{i[1][0]}_Rate']['v']
	if LET1:
		let1dtype = 'let1_rates1h'
		if EPIHI_ENC: let1dtype = 'let1_rates1min'
		LET1d		= pyspedas.psp.epihi(trange=[Start, End], datatype=let1dtype, 
				      get_support_data=True, notplot=True, level='l2', time_clip=True)
		LET1dict	= {}
		for i in ISOISvars:
			if i[0]:
				if np.count_nonzero(HETd[f'psp_epihi_A_{i[1][0]}_Rate']['y']) == 0: #Invalidate element if rate array is filled with zeros
					print(ts(), f'No EPI-Hi data found for element {i[1][0]}.')
					False_list.append(i[1][0])
				else:
					LET1dict[f'LET1_{i[1][0]}time']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['x']
					LET1dict[f'LET1_{i[1][0]}rate']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['y'] \
													+ LET1d[f'psp_epihi_B_{i[1][0]}_Rate']['y']
					LET1dict[f'LET1_{i[1][0]}ergs']	= LET1d[f'psp_epihi_A_{i[1][0]}_Rate']['v']
	if EPILO:
		ICv		= pyspedas.psp.epilo(trange=[Start, End], datatype='ic',
				  get_support_data=EPILOFLAG, notplot=True, level='l2', time_clip=True)
		PEv		= pyspedas.psp.epilo(trange=[Start, End], datatype='pe',
				  get_support_data=EPILOFLAG, notplot=True, level='l2', time_clip=True)
		EPILOd		= ICv | PEv #Merge IC and PE dictionaries
		EPILOdict	= {}
		for i in ISOISvars:
			if i[0]:
				if i[1][2] == None:
					continue
				if len(set(EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['v2'][0,0])) <= 1: #Invalidate element if ergs array is filled with -1.e+31
					print(ts(), f'No EPI-Lo data found for element {i[1][2]}.')
					False_list.append(i[1][2])
				else:
					EPILOdict[f'EPILO_{i[1][2]}time']	= EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['x'] #Time array
					EPILOdict[f'EPILO_{i[1][2]}ergs']	= EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['v2'][0,0] #Energy bins array
					for j in reversed(range(len(EPILOdict[f'EPILO_{i[1][2]}ergs']))): #Removing high-end energy bins which might be unused (indicated with values of -9.9999998e+30)
						if EPILOdict[f'EPILO_{i[1][2]}ergs'][j] < 0:
							EPILOdict[f'EPILO_{i[1][2]}ergs']	= np.delete(EPILOdict[f'EPILO_{i[1][2]}ergs'], j)
						else:
							continue
						EPILOdict[f'EPILO_{i[1][2]}rate']	= np.nan_to_num(EPILOd[f'psp_epilo_{i[1][2]}_CountRate_{i[1][1]}']['y'], nan=0) # Time*bins ion rate array
						EPILOdict[f'EPILO_{i[1][2]}rate']	= np.mean(EPILOdict[f'EPILO_{i[1][2]}rate'], axis=1) #Mean from the 80 apertures (look directions)
						EPILOdict[f'EPILO_{i[1][2]}rate'] 	= EPILOdict[f'EPILO_{i[1][2]}rate'][:, :len(EPILOdict[f'EPILO_{i[1][2]}ergs'])] #Truncating energy axis to match the energy bins in ICprotergs:

#Quality flags
		  	  #['instrument', 'flag specifier', 'instrument specifier']
FLAGnames	= [['spe', 'spe_QUALITY_FLAG',SWEAPd], ['spi','spi_QUALITY_FLAG',SWEAPd], ['spc','spc_general_flag',SWEAPd],
		       ['fields', 'fld_l2_quality_flags',FIELDSd], ['het', 'epihi_Quality_Flag',HETd], ['let1', 'epihi_Quality_Flag',LET1d],
		       ['epilo', 'epilo_Quality_Flag_ChanP',EPILOd], ['epilo', 'epilo_Quality_Flag_ChanD',EPILOd],
		   	   ['epilo', 'epilo_Quality_Flag_ChanC',EPILOd], ['epilo', 'epilo_Quality_Flag_ChanE',EPILOd]]
FLAGvars	= list(zip(FLAGS, FLAGnames))
if any(FLAGS):
	FLAGSdict	= {}
	for i in FLAGvars:
		if i[0]:
			FLAGSdict[f'{i[1][0]}_flag_x']	= i[1][2][f'psp_{i[1][1]}']['x']
			FLAGSdict[f'{i[1][0]}_flag_y']	= i[1][2][f'psp_{i[1][1]}']['y']

print(ts(), 'Data loaded. Plotting...')

#Calculations
if NORM_BRTN or MAGB or AVEL or AMACH:
	#Absolute B
	absB			= np.linalg.norm(FIELDSdict['BRTN_y'], axis=1)
	if NORM_BRTN or AVEL or AMACH:
		#Converting distance data dimension to the same dimension as magnetic field data by interpolation
		SPIepoch		= np.float64(SWEAPdict['spi_SUN_DIST_x'])
		FIELDSepoch		= np.float64(FIELDSdict['BRTN_x'])
		f1 = scipy.interpolate.interp1d(SPIepoch, SWEAPdict['spi_SUN_DIST_y'], fill_value='extrapolate')
		ip_heliodist	= f1(FIELDSepoch)

x = 1
if NORM_BRTN: x = (ip_heliodist/149597871)**2

#Alfvén speed and Alfvén MACH number
if AVEL or AMACH:
	#Conversion of density resolution to B resolution by interpolation
	if SPI_OPT:
		f2p = scipy.interpolate.interp1d(SPIepoch, SWEAPdict['spi_DENS_y'], fill_value='extrapolate')
	else:
		SPCepoch = np.float64(SWEAPdict['spc_np_moment_x'])
		f2p = scipy.interpolate.interp1d(SPCepoch, SWEAPdict['spc_np_moment_y'], fill_value='extrapolate')
	ip_protdens = np.array(f2p(FIELDSepoch))
	Av_p = 10E-9*absB/(1000*np.sqrt(4*np.pi*10E-7*constants.proton_mass*10E6*ip_protdens)) #Alfvén speed from proton density
	if ELECDENS:
		FIELDSQTNepoch = np.float64(FIELDSdict['elecdens_x'])
		f2e = scipy.interpolate.interp1d(FIELDSQTNepoch, FIELDSdict['elecdens_y'], fill_value='extrapolate')
		ip_elecdens = np.array(f2e(FIELDSepoch))
		Av_e = 10E-9*absB/(1000*np.sqrt(4*np.pi*10E-7*constants.proton_mass*10E6*ip_elecdens)) #Alfvén speed from electron density
	if AMACH:
		#Conversion of vR resolution to B resolution by interpolation
		if SPI_OPT:
			f4 = scipy.interpolate.interp1d(SPIepoch, SWEAPdict['spi_VEL_RTN_SUN_y'][:,0], fill_value='extrapolate')
		else:
			f4 = scipy.interpolate.interp1d(SPCepoch, SWEAPdict['spc_vp_moment_RTN_y'][:,0], fill_value='extrapolate')
		ip_vR = f4(FIELDSepoch)
		Mach_p = ip_vR/Av_p #(radial) MACH number from proton density
		if ELECDENS:
			Mach_e = ip_vR/Av_e #(radial) MACH number from electron density

#Plotting
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
		cbarlabel 	= f'{element}\neV/cm^2/s/sr/eV'
		ylabel 		= f'{label}'
	else:
		cbarlabel 	= f'{element}\ncounts/s'
		ylabel 		= f'{element}\n{label}'
	c = ax[index_count].pcolormesh(time, ergs, rate.T, shading='auto', cmap='turbo', norm=LogNorm())
	cbar = plt.colorbar(c, ax=ax[index_count])
	cbar.set_label(cbarlabel, fontsize=labelsize)
	ax[index_count].set_ylabel(ylabel, fontsize=labelsize)
	if log: ax[index_count].set_yscale('log')
	index_count += 1

#Function for plotting quality flags
def flags_plot(time, flag, label):
	global index_count
	ax[index_count].plot(time, flag, color='black', alpha=0.6)
	ax[index_count].set_ylabel(label, fontsize=labelsize)
	ax[index_count].tick_params(axis='both', which='major', labelsize=ticklabelsize)		
	index_count += 1

plt.rcParams['font.family'] = font #Plot font
plt.rcParams['mathtext.fontset'] = 'stix' #Math fontset

if plots_count == 1: #plt.subplots() does not allow subscripting when plotting only one plot
	fig, ax = plt.subplots(nrows = 2, sharex=False, constrained_layout=True)#, gridspec_kw={'hspace':hspace})
	ax[1].axis('off')
else:
	fig, ax = plt.subplots(nrows = plots_count, sharex=True, constrained_layout=True, gridspec_kw={'hspace':hspace})

index_count = 0

#SWEAP and FIELDS products
# 				(x,								y,										label,								color,		**kwargs)
if BR: xy_plot(FIELDSdict['BRTN_x'], 			x*FIELDSdict['BRTN_y'][:,0], 			r'$\mathit{B}_\text{R}$', 			'blue')
if RVEL:
	xy_plot(SWEAPdict['spi_VEL_RTN_SUN_x'], 	SWEAPdict['spi_VEL_RTN_SUN_y'][:,0], 	r'$\mathit{v}_\text{R}$'+'\nSPI',	'orange',	twinx=True) if SPI_OPT else \
	xy_plot(SWEAPdict['spc_vp_moment_RTN_x'], 	SWEAPdict['spc_vp_moment_RTN_y'][:,0], 	r'$\mathit{v}_\text{R}$'+'\nSPC',	'orange',	twinx=True)
if BT: xy_plot(FIELDSdict['BRTN_x'], 			x*FIELDSdict['BRTN_y'][:,1], 			r'$\mathit{B}_\text{T}$'		,	'blue')
if TVEL:
	xy_plot(SWEAPdict['spi_VEL_RTN_SUN_x'], 	SWEAPdict['spi_VEL_RTN_SUN_y'][:,1], 	r'$\mathit{v}_\text{T}$'+'\nSPI', 	'orange', 	twinx=True) if SPI_OPT else \
	xy_plot(SWEAPdict['spc_vp_moment_RTN_x'], 	SWEAPdict['spc_vp_moment_RTN_y'][:,1], 	r'$\mathit{v}_\text{T}$'+'\nSPC',	'orange', 	twinx=True)
if BN: xy_plot(FIELDSdict['BRTN_x'], 			x*FIELDSdict['BRTN_y'][:,2], 			r'$\mathit{B}_\text{N}$', 			'blue')
if NVEL:
	xy_plot(SWEAPdict['spi_VEL_RTN_SUN_x'], 	SWEAPdict['spi_VEL_RTN_SUN_y'][:,2], 	r'$\mathit{v}_\text{N}$'+'\nSPI',	'orange', 	twinx=True) if SPI_OPT else \
	xy_plot(SWEAPdict['spc_vp_moment_RTN_x'], 	SWEAPdict['spc_vp_moment_RTN_y'][:,2], 	r'$\mathit{v}_\text{N}$'+'\nSPC',	'orange', 	twinx=True)
if MAGB: xy_plot(FIELDSdict['BRTN_x'], 			absB, 									r'$\mathit{B}$',					'blue')
if DENS:
	xy_plot(SWEAPdict['spi_DENS_x'],			SWEAPdict['spi_DENS_y'], 				r'$\mathit{n}$ [1/cm³]'+'\nSPI',	'black',	tag=r'$\mathit{n}_\text{p}$', log=True) if SPI_OPT else \
	xy_plot(SWEAPdict['spc_np_moment_x'], 		SWEAPdict['spc_np_moment_y'], 			r'$\mathit{n}$ [1/cm³]'+'\nSPC',	'black',	tag=r'$\mathit{n}_\text{p}$', log=True)
	if ELECDENS: xy_plot(FIELDSdict['BRTN_x'], 	ip_elecdens, 							r'$\mathit{n}$ [1/cm³]',			'orange',	tag=r'$\mathit{n}_\text{e}$', log=True, previousplot=True)
if AVEL:
	xy_plot(FIELDSdict['BRTN_x'], 				Av_p, 									r'$\mathit{v}_\text{A}$', 			'black')
	if ELECDENS: xy_plot(FIELDSdict['BRTN_x'], 	Av_e, 									r'$\mathit{v}_\text{A}$', 			'orange', 	previousplot=True)
if AMACH:
	xy_plot(FIELDSdict['BRTN_x'], 				Mach_p, 								'Mach', 							'black', 	log=True)
	if ELECDENS: xy_plot(FIELDSdict['BRTN_x'],	Mach_e, 								'Mach', 							'orange', 	log=True, previousplot=True)
	ax[index_count-1].hlines(y=1,xmin=FIELDSdict['BRTN_x'][0], xmax=FIELDSdict['BRTN_x'][-1], color='red', linestyles='--') #Mach-1 limit

if PAD: specgram_plot(SWEAPdict['spe_EFLUX_VS_PA_E_x'], 	SWEAPdict['spe_EFLUX_VS_PA_E_v1'], 	SWEAPdict['spe_EFLUX_VS_PA_E_y'][:,:,n],	'PA', 			'%.2f' % SWEAPdict['spe_EFLUX_VS_PA_E_v2'][n], log=False, intensity=True)
if EED: specgram_plot(SWEAPdict['spe_EFLUX_VS_ENERGY_x'], 	SWEAPdict['spe_EFLUX_VS_ENERGY_v'], SWEAPdict['spe_EFLUX_VS_ENERGY_y'], 		'Energy [eV]', 	'Electron',	intensity=True)
if PED: specgram_plot(SWEAPdict['spi_EFLUX_VS_ENERGY_x'], 	SWEAPdict['spi_EFLUX_VS_ENERGY_v'], SWEAPdict['spi_EFLUX_VS_ENERGY_y'], 		'Energy [eV]', 	'Proton',	intensity=True)

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
if any(FLAGS):
	for i in FLAGvars:
		if i[0]:
			flags_plot(FLAGSdict[f'{i[1][0]}_flag_x'], FLAGSdict[f'{i[1][0]}_flag_y'], f'{i[1][0]}\nflag')

#Heliocentric distance
if HELIODIST:
	dist = ax[0].twiny()
	dist.set_xlim(ax[0].get_xlim())
	dist.set_xticks(SWEAPdict['spi_SUN_DIST_x'][::4000])
	dist.set_xticklabels(np.round(SWEAPdict['spi_SUN_DIST_y'][::4000]/696340, 1))
	dist.set_xlabel("Heliocentric distance [Solar radii]", fontsize=labelsize)
	dist.tick_params(axis='both', which='major', labelsize=ticklabelsize)

fig.align_labels()
fig.align_titles()

plt.show()