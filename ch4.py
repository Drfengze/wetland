import numpy as np
import pandas as pd

def peprmt_ch4_final(theta, data, wetland_type):
    """
    PEPRMT-Tidal Methane Module v1.0
    
    Originally developed by Patty Oikawa (patty.oikawa@gmail.com)
    Published in November 2023
    Translated to Python
    
    Parameters:
    -----------
    theta : array-like
        Vector of 7 parameter values determined via MCMC Bayesian fitting
    data : pandas.DataFrame 
        DataFrame containing required input variables
    wetland_type : int
        1 for freshwater peatland, 2 for tidal wetland
        
    Returns:
    --------
    pandas.DataFrame
        Contains the following columns:
        - pulse_emission_total: total methane emitted (g C methane m^-2 day^-1)
        - Plant_flux_net: net methane released from plants (g C methane m^-2 day^-1) 
        - Hydro_flux: net methane transfer from water to atm (g C methane m^-2 day^-1)
        - M1: methane from soil carbon pool 1 (g C methane m^-3)
        - M2: methane from soil carbon pool 2 (g C methane m^-3)
        - trans2: fraction of CH4 released via plant-mediated transport (unitless)
    """
    
    # Constants
    R = 8.314  # J K-1 mol-1

    # CH4 parameters
    # SOC pool
    M_alpha1 = 6.2e13  # gC m-3 d-1
    M_ea1 = (theta[0] + 67.1) * 1000  # J mol-1
    M_km1 = theta[1] + 17  # g C m-3
    
    # Labile C pool 
    M_alpha2 = 6.2e14  # gC m-3 d-1
    M_ea2 = (theta[2] + 71.1) * 1000  # J mol-1
    M_km2 = theta[3] + 23  # g C m-3
    
    # CH4 oxidation parameters
    M_alpha3 = 6.2e13  # gC m-3 d-1
    M_ea3 = (theta[4] + 75.4) * 1000  # J mol-1
    M_km3 = theta[5] + 23  # g C m-3
    
    # Salinity sulfate parameters
    kiSO4 = theta[6]  # mg L^-1
    kiNO3 = theta[7]  # mg L^-1
    
    # Parameters for hydrodynamic flux
    k = 0.04  # gas transfer velocity (m day-1)
    
    # Parameters for plant-mediated transport
    Vtrans = 0.24  # gas transfer velocity through plants(m d-1)
    Oxi_factor = 0.35  # percent oxidized during transport
    
    # Empirical factors for inhibition when WT drops
    beta1 = 0.48
    beta2 = -0.18
    beta3 = 0.0042
    
    # Empirical factors for decaying inhibition following first flooding
    zeta1 = 5.1e-6
    zeta2 = 0.00058
    zeta3 = 0.11

    def process_site_data(site_data):
        """Process data for a single site"""
        # Extract variables
        Time_2 = site_data.iloc[:, 0].values  # day of year
        DOY_disc_2 = site_data.iloc[:, 1].values  # discontinuous day of year
        Year_2 = site_data.iloc[:, 2].values  # year
        TA_2 = site_data.iloc[:, 3].values  # Air temperature (C)
        WT_2 = site_data.iloc[:, 4].values  # water table depth (cm)
        PAR_2 = site_data.iloc[:, 5].values  # photosynthetically active radiation
        LAI_2 = site_data.iloc[:, 6].values  # Leaf area index
        GI_2 = site_data.iloc[:, 7].values  # greenness index
        FPAR = site_data.iloc[:, 8].values  # FPAR variable
        LUE = site_data.iloc[:, 9].values  # growing season LUE
        wetland_age_2 = site_data.iloc[:, 10].values  # Age of wetland in years
        Sal = site_data.iloc[:, 11].values  # Salinity (ppt)
        NO3 = site_data.iloc[:, 12].values  # Dissolved NO3 (mg/L)
        SOM_2 = site_data.iloc[:, 13].values  # Decomposed Organic matter
        site_2 = site_data.iloc[:, 14].values  # Site
        GPP_2 = site_data.iloc[:, 15].values  # Modeled GPP
        S1_2 = site_data.iloc[:, 16].values  # Modeled SOC pool
        S2_2 = site_data.iloc[:, 17].values  # Modeled labile C pool

        n = len(Time_2)
        WT_2_adj = (WT_2/100) + 1
        GPPmax = np.max(-GPP_2)

        # Time Invariant calculations
        RT = R * (TA_2 + 274.15)  # T in Kelvin
        M_Vmax1 = M_alpha1 * np.exp(-M_ea1/RT)  # g C m-2 d-1
        M_Vmax2 = M_alpha2 * np.exp(-M_ea2/RT)  # gC m-2 d-1
        M_Vmax3 = M_alpha3 * np.exp(-M_ea3/RT)  # gC m-2 d-1

        # Initialize arrays
        S1sol = np.zeros(n)
        S2sol = np.zeros(n)
        M1 = np.zeros(n)
        M2 = np.zeros(n)
        M1_full = np.zeros(n)
        M2_full = np.zeros(n)
        M_full = np.zeros(n)
        M_percent_reduction = np.zeros(n)
        M_percent_reduction_2 = np.zeros(n)
        CH4water = np.zeros(n)
        Hydro_flux = np.zeros(n)
        Plant_flux = np.zeros(n)
        Plant_flux_net = np.zeros(n)
        CH4water_store = np.zeros(n)
        CH4water_0 = np.zeros(n)
        CH4water_0_2 = np.zeros(n)
        Oxi_full = np.zeros(n)
        R_Oxi = np.zeros(n)
        trans2 = np.zeros(n)
        S1 = np.zeros(n)
        S2 = np.zeros(n)
        conc_so4AV = np.zeros(n)

        # Main time loop
        for t in range(n):
            # Calculate plant-mediated transport parameter
            trans2[t] = (GPP_2[t] + GPPmax) / GPPmax
            trans2[t] = np.clip(trans2[t], 0, 1)

            # Calculate sulfate concentration
            conc_so4AV[t] = 7.4e-8 * (Sal[t] * 1e6) * 1000  # ppm or mg L-1

            # Calculate methane production
            M1[t] = (M_Vmax1[t] * (S1_2[t] / (M_km1 + S1_2[t]))) * \
                    (kiSO4 / (kiSO4 + conc_so4AV[t])) * \
                    (kiNO3 / (kiNO3 + NO3[t]))
                    
            M2[t] = (M_Vmax2[t] * (S2_2[t] / (M_km2 + S2_2[t]))) * \
                    (kiSO4 / (kiSO4 + conc_so4AV[t])) * \
                    (kiNO3 / (kiNO3 + NO3[t]))

            M1[t] = max(0, M1[t])
            M2[t] = max(0, M2[t])

            # For Freshwater peatlands only
            if wetland_type == 1:
                if t <= 19 and WT_2_adj[t] < 0.9:
                    M_percent_reduction[t] = (beta1 * WT_2_adj[t]**2) + \
                                          (beta2 * WT_2_adj[t]) + beta3
                else:
                    M_percent_reduction[t] = 1

                if t > 19:
                    Sel = WT_2_adj[max(0, t-19):t+1]
                    if np.min(Sel) < 0.9:
                        M_percent_reduction[t] = (beta1 * WT_2_adj[t]**2) + \
                                              (beta2 * WT_2_adj[t]) + beta3
                    else:
                        M_percent_reduction[t] = 1

                if WT_2_adj[t] < 0:
                    M_percent_reduction[t] = 0

                M_percent_reduction[t] = np.clip(M_percent_reduction[t], 0, 1)

                # CH4 inhibition for 1 yr following restoration
                if wetland_age_2[t] < 2:
                    M_percent_reduction_2[t] = (zeta1 * DOY_disc_2[t]**2) + \
                                             (zeta2 * DOY_disc_2[t]) + zeta3
                else:
                    M_percent_reduction_2[t] = 1

                M_percent_reduction_2[t] = min(1, M_percent_reduction_2[t])
                M_percent_reduction[t] = max(0.75, M_percent_reduction[t])

                M1[t] *= M_percent_reduction[t] * M_percent_reduction_2[t]
                M2[t] *= M_percent_reduction[t] * M_percent_reduction_2[t]

            # Calculate new SOC and labile pools
            S1sol[t] = max(0, S1[t] - M1[t])
            S2sol[t] = max(0, S2[t] - M2[t])

            M_full[t] = M1[t] + M2[t]
            WT_2_adj[t] = max(0, WT_2_adj[t])

            # CH4 transport calculations
            if t == 0:
                if WT_2_adj[t] > 1:
                    R_Oxi[t] = 0
                    Oxi_full[t] = 0
                    CH4water_0[t] = 0
                    CH4water_0_2[t] = 0
                    CH4water[t] = ((M_full[t] * 1) + (CH4water_0[t] * WT_2_adj[t])) / WT_2_adj[t]
                else:
                    CH4water_0[t] = (M_full[t] * 1 + 0.00001 * WT_2_adj[t]) / WT_2_adj[t]
                    R_Oxi[t] = M_Vmax3[t] * CH4water_0[t] / (M_km3 + CH4water_0[t])
                    Oxi_full[t] = R_Oxi[t]
                    CH4water[t] = max(0, CH4water_0[t] - Oxi_full[t])
                    CH4water_0_2[t] = 0

                Hydro_flux[t] = k * CH4water[t]
                Plant_flux[t] = (Vtrans * CH4water[t]) * trans2[t]
                Plant_flux_net[t] = Plant_flux[t] * Oxi_factor
                CH4water_store[t] = CH4water[t] - Plant_flux[t] - Hydro_flux[t]

            else:
                if WT_2_adj[t] > 1:
                    CH4water_0[t] = (CH4water_store[t-1] * WT_2_adj[t-1]) / WT_2_adj[t]
                    R_Oxi[t] = 0
                    Oxi_full[t] = 0
                    CH4water_0_2[t] = 0
                    CH4water[t] = ((M_full[t] * 1) + (CH4water_0[t] * WT_2_adj[t])) / WT_2_adj[t]
                else:
                    CH4water_0[t] = (CH4water_store[t-1] * WT_2_adj[t-1]) / WT_2_adj[t]
                    CH4water_0_2[t] = ((M_full[t] * 1) + (CH4water_0[t] * WT_2_adj[t])) / WT_2_adj[t]
                    R_Oxi[t] = M_Vmax3[t] * CH4water_0_2[t] / (M_km3 + CH4water_0_2[t])
                    Oxi_full[t] = R_Oxi[t]
                    CH4water[t] = max(0, CH4water_0_2[t] - Oxi_full[t])

                Hydro_flux[t] = k * CH4water[t]
                Plant_flux[t] = (Vtrans * CH4water[t]) * trans2[t]
                Plant_flux_net[t] = Plant_flux[t] * Oxi_factor
                CH4water_store[t] = CH4water[t] - Plant_flux[t] - Hydro_flux[t]

        pulse_emission_total = Plant_flux_net + Hydro_flux

        return pd.DataFrame({
            'pulse_emission_total': pulse_emission_total,
            'Plant_flux_net': Plant_flux_net,
            'Hydro_flux': Hydro_flux,
            'M1': M1,
            'M2': M2,
            'trans2': trans2
        })

    # Process each site
    sites = data['site'].unique()
    results = []
    
    for site in sites:
        site_data = data[data['site'] == site]
        site_results = process_site_data(site_data)
        results.append(site_results)
    
    return pd.concat(results, ignore_index=True)