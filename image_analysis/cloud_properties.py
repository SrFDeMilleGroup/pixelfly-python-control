import numpy as np


number = 2.0e9
number_err = 0
axial_temp = 14 # in uK
axial_temp_err = 0
radial_temp = 9 # in uK
radial_temp_err = 0
axial_width = 1.60 # in mm
axial_width_err = 0
radial_width = 1.60 # in mm
radial_width_err = 0

kB = 1.38064852e-23 # J/K
mass = 86.9*1.66053873e-27 # kg
hbar = 1.0545718e-34 # J*s

peak_number_density = number/(radial_width/10)**2/(axial_width/10)/(2*np.pi)**(3/2) # in 1/cm^3
peak_number_density_err = peak_number_density*np.sqrt((number_err/number)**2
                                                        +(2*radial_width_err/radial_width)**2
                                                        +(axial_width_err/axial_width)**2) # in 1/cm^3

radial_thermal_wavelength = hbar*np.sqrt(2*np.pi/mass/kB/(radial_temp/1e6))*100 # in cm
radial_thermal_wavelength_err = radial_thermal_wavelength*radial_temp_err/2/radial_temp # in cm
axial_thermal_wavelength = hbar*np.sqrt(2*np.pi/mass/kB/(axial_temp/1e6))*100 # in cm
axial_thermal_wavelength_err = axial_thermal_wavelength*axial_temp_err/2/axial_temp # in cm

peak_psd = peak_number_density*axial_thermal_wavelength*radial_thermal_wavelength**2 # phase-space density
peak_psd_err = peak_psd*np.sqrt((peak_number_density_err/peak_number_density)**2
                                +(2*radial_thermal_wavelength_err/radial_thermal_wavelength)**2
                                +(axial_thermal_wavelength_err/axial_thermal_wavelength)**2)


print("peak_number_density: "+np.format_float_scientific(peak_number_density, precision=2)
        +" ("+np.format_float_scientific(peak_number_density_err, precision=2)+") cm^-3")

print("peak phase space density: "+np.format_float_scientific(peak_psd, precision=2)
        +" ("+np.format_float_scientific(peak_number_density_err, precision=2)+")")
