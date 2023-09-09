import numpy as np

def GI_reflex(b, d, t):
    PP_raw = t / np.power(b, 2) * np.exp(-np.power(t, 2) / (2*np.power(b, 2)))
    PP_max = b / np.power(b, 2) * np.exp(-np.power(b, 2) / (2*np.power(b, 2)))

    PP = 1 + d * PP_raw / PP_max
    return PP

def BA_ode(t, state, param):
    '''
    Define the bile acid metabolism extended with PSC pathophysiology using a system of ordinary differential equations (ODEs)
    A summary of the biochemical and physical processes being modeled and the complete mathematical descriptions can be found in the appendix
    '''
    ##### Defining States #####
    
    x_li_cCA    = state[0]
    x_li_cCDCA  = state[1]
    x_li_cSBA   = state[2]
    x_li_uCA    = 0
    x_li_uCDCA  = 0
    x_li_uSBA   = 0

    x_bd_cCA    = state[3]
    x_bd_cCDCA  = state[4]
    x_bd_cSBA   = state[5]
    x_bd_uCA    = 0
    x_bd_uCDCA  = 0
    x_bd_uSBA   = 0

    x_dsi_cCA   = state[6]
    x_dsi_cCDCA = state[7]
    x_dsi_cSBA  = state[8]
    x_dsi_uCA   = state[9]
    x_dsi_uCDCA = state[10]
    x_dsi_uSBA  = state[11]

    x_co_cCA    = state[12]
    x_co_cCDCA  = state[13]
    x_co_cSBA   = state[14]
    x_co_uCA    = state[15]
    x_co_uCDCA  = state[16]
    x_co_uSBA   = state[17]

    x_pl_cCA    = state[18]
    x_pl_cCDCA  = state[19]
    x_pl_cSBA   = state[20]
    x_pl_uCA    = state[21]
    x_pl_uCDCA  = state[22]
    x_pl_uSBA   = state[23]

    # x_fe_cCA    = state[24]
    # x_fe_cCDCA  = state[25]
    # x_fe_cSBA   = state[26]
    # x_fe_uCA    = state[27]
    # x_fe_uCDCA  = state[28]
    # x_fe_uSBA   = state[29]



    ##### Calculate Time (Minute) Since Meal #####
    t_in_day = (t % (60*24)) / 60
    if t_in_day <= 6:
        t_since_meal = t_in_day
    elif t_in_day <= 12:
        t_since_meal = t_in_day - 6
    else:
        t_since_meal = t_in_day - 12
    t_since_meal = t_since_meal * 60



    ##### Defining Parameters #####

    p = dict()

    p["synthesis"]                        = param[0] 
    p["syn_frac_CA"]                      = param[1] 
    p["li_to_bd_freq"]                    = param[2] 
    p["bd_to_dsi_freq"]                   = param[3] 
    p["dsi_to_co_freq"]                   = param[4] 
    p["co_to_fe_freq"]                    = param[5] 
    p["pl_volume"]                        = param[6] 
    p["pl_flow_thru_li"]                  = param[7] 
    p["hep_extract_ratio_conj_tri"]       = param[8] 
    p["hep_extract_ratio_conj_di"]        = param[9] 
    p["hep_extract_ratio_unconj_to_conj"] = param[10]
    p["gut_deconj_freq_co"]               = param[11]
    p["gut_deconj_freq_dsi_to_co"]        = param[12]
    p["gut_biotr_freq_CA"]                = param[13]
    p["gut_biotr_freq_CDCA_to_CA"]        = param[14]
    p["max_asbt_rate"]                    = param[15]
    p["gut_pu_freq_co"]                   = param[16]
    p["gut_pu_freq_dsi_to_co"]            = param[17]
    p["meal_reflex_loc_bd"]               = param[18]
    p["meal_reflex_loc_dsi"]              = param[19]
    p["meal_reflex_loc_co"]               = param[20]
    p["meal_reflex_peak_bd"]              = param[21]
    p["meal_reflex_peak_dsi"]             = param[22]
    p["meal_reflex_peak_co"]              = param[23]
    p["co_sulfate_freq"]                  = param[24]
    p["bd_max_ba"]                        = param[25]
    p["bd_max_flow"]                      = param[26]

    p["hep_extract_ratio_unconj_tri"]     = p["hep_extract_ratio_conj_tri"] * p["hep_extract_ratio_unconj_to_conj"]
    p["hep_extract_ratio_unconj_di"]      = p["hep_extract_ratio_conj_di"]  * p["hep_extract_ratio_unconj_to_conj"]
    p["gut_deconj_freq_dsi"]              = p["gut_deconj_freq_dsi_to_co"]  * p["gut_deconj_freq_co"]
    p["gut_biotr_freq_CDCA"]              = p["gut_biotr_freq_CDCA_to_CA"]  * p["gut_biotr_freq_CA"]
    p["gut_pu_freq_dsi"]                  = p["gut_pu_freq_dsi_to_co"]      * p["gut_pu_freq_co"]

    p["bd_transit_coef"]                  = max(GI_reflex(p["meal_reflex_loc_bd"],  p["meal_reflex_peak_bd"],  t_since_meal), 1)
    p["dsi_transit_coef"]                 = max(GI_reflex(p["meal_reflex_loc_dsi"], p["meal_reflex_peak_dsi"], t_since_meal), 1)
    p["co_transit_coef"]                  = max(GI_reflex(p["meal_reflex_loc_co"], p["meal_reflex_peak_co"],  t_since_meal), 1)

    ##### Defining Rates Of Change #####

    li_in_synthesis_cCA   = p["synthesis"] * p["syn_frac_CA"]
    li_in_synthesis_cCDCA = p["synthesis"] * (1 - p["syn_frac_CA"])

    li_out_to_bd_cCA      = p["li_to_bd_freq"] * x_li_cCA
    li_out_to_bd_cCDCA    = p["li_to_bd_freq"] * x_li_cCDCA
    li_out_to_bd_cSBA     = p["li_to_bd_freq"] * x_li_cSBA
    bd_in_from_li_cCA     = li_out_to_bd_cCA
    bd_in_from_li_cCDCA   = li_out_to_bd_cCDCA
    bd_in_from_li_cSBA    = li_out_to_bd_cSBA

    dsi_out_to_co_cCA     = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_cCA
    dsi_out_to_co_cCDCA   = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_cCDCA
    dsi_out_to_co_cSBA    = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_cSBA
    dsi_out_to_co_uCA     = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_uCA
    dsi_out_to_co_uCDCA   = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_uCDCA
    dsi_out_to_co_uSBA    = p["dsi_to_co_freq"] * p["dsi_transit_coef"] * x_dsi_uSBA
    co_in_from_dsi_cCA    = dsi_out_to_co_cCA
    co_in_from_dsi_cCDCA  = dsi_out_to_co_cCDCA
    co_in_from_dsi_cSBA   = dsi_out_to_co_cSBA
    co_in_from_dsi_uCA    = dsi_out_to_co_uCA
    co_in_from_dsi_uCDCA  = dsi_out_to_co_uCDCA
    co_in_from_dsi_uSBA   = dsi_out_to_co_uSBA

    co_out_to_fe_cCA     = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_cCA
    co_out_to_fe_cCDCA   = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_cCDCA
    co_out_to_fe_cSBA    = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_cSBA
    co_out_to_fe_uCA     = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_uCA
    co_out_to_fe_uCDCA   = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_uCDCA
    co_out_to_fe_uSBA    = p["co_to_fe_freq"] * p["co_transit_coef"] * x_co_uSBA
    fe_in_from_co_cCA    = co_out_to_fe_cCA
    fe_in_from_co_cCDCA  = co_out_to_fe_cCDCA
    fe_in_from_co_cSBA   = co_out_to_fe_cSBA
    fe_in_from_co_uCA    = co_out_to_fe_uCA
    fe_in_from_co_uCDCA  = co_out_to_fe_uCDCA
    fe_in_from_co_uSBA   = co_out_to_fe_uSBA

    pl_out_to_li_cCA      = p["hep_extract_ratio_conj_tri"]   * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_cCA
    pl_out_to_li_cCDCA    = p["hep_extract_ratio_conj_di"]    * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_cCDCA
    pl_out_to_li_cSBA     = p["hep_extract_ratio_conj_di"]    * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_cSBA
    pl_out_to_li_uCA      = p["hep_extract_ratio_unconj_tri"] * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_uCA
    pl_out_to_li_uCDCA    = p["hep_extract_ratio_unconj_di"]  * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_uCDCA
    pl_out_to_li_uSBA     = p["hep_extract_ratio_unconj_di"]  * p["pl_flow_thru_li"] / p["pl_volume"] * x_pl_uSBA
    li_in_from_pl_cCA     = pl_out_to_li_cCA   + pl_out_to_li_uCA
    li_in_from_pl_cCDCA   = pl_out_to_li_cCDCA + pl_out_to_li_uCDCA
    li_in_from_pl_cSBA    = pl_out_to_li_cSBA  + pl_out_to_li_uSBA

    dsi_out_deconj_cCA    = p["gut_deconj_freq_dsi"] * x_dsi_cCA
    dsi_out_deconj_cCDCA  = p["gut_deconj_freq_dsi"] * x_dsi_cCDCA
    dsi_out_deconj_cSBA   = p["gut_deconj_freq_dsi"] * x_dsi_cSBA
    co_out_deconj_cCA     = p["gut_deconj_freq_co"]  * x_co_cCA
    co_out_deconj_cCDCA   = p["gut_deconj_freq_co"]  * x_co_cCDCA
    co_out_deconj_cSBA    = p["gut_deconj_freq_co"]  * x_co_cSBA
    dsi_in_deconj_uCA     = dsi_out_deconj_cCA
    dsi_in_deconj_uCDCA   = dsi_out_deconj_cCDCA
    dsi_in_deconj_uSBA    = dsi_out_deconj_cSBA
    co_in_deconj_uCA      = co_out_deconj_cCA
    co_in_deconj_uCDCA    = co_out_deconj_cCDCA
    co_in_deconj_uSBA     = co_out_deconj_cSBA

    co_out_biotr_uCA     = p["gut_biotr_freq_CA"] * x_co_uCA
    co_out_biotr_uCDCA   = p["gut_biotr_freq_CDCA"] * x_co_uCDCA
    co_in_biotr_uSBA     = co_out_biotr_uCA + co_out_biotr_uCDCA

    dsi_out_au_cCA        = p["max_asbt_rate"] * x_dsi_cCA
    dsi_out_au_cCDCA      = p["max_asbt_rate"] * x_dsi_cCDCA
    dsi_out_au_cSBA       = p["max_asbt_rate"] * x_dsi_cSBA
    dsi_out_au_uCA        = p["max_asbt_rate"] * x_dsi_uCA
    dsi_out_au_uCDCA      = p["max_asbt_rate"] * x_dsi_uCDCA
    dsi_out_au_uSBA       = p["max_asbt_rate"] * x_dsi_uSBA

    dsi_out_pu_uCA        = p["gut_pu_freq_dsi"] * x_dsi_uCA
    dsi_out_pu_uCDCA      = p["gut_pu_freq_dsi"] * x_dsi_uCDCA
    dsi_out_pu_uSBA       = p["gut_pu_freq_dsi"] * x_dsi_uSBA
    co_out_pu_uCA         = p["gut_pu_freq_co"]  * x_co_uCA
    co_out_pu_uCDCA       = p["gut_pu_freq_co"]  * x_co_uCDCA
    co_out_pu_uSBA        = p["gut_pu_freq_co"]  * x_co_uSBA
                          
    li_in_from_gut_cCA    = p["hep_extract_ratio_conj_tri"]         * dsi_out_au_cCA   + p["hep_extract_ratio_unconj_tri"]       * (dsi_out_au_uCA   + dsi_out_pu_uCA   + co_out_pu_uCA)
    li_in_from_gut_cCDCA  = p["hep_extract_ratio_conj_di"]          * dsi_out_au_cCDCA + p["hep_extract_ratio_unconj_di"]        * (dsi_out_au_uCDCA + dsi_out_pu_uCDCA + co_out_pu_uCDCA)
    li_in_from_gut_cSBA   = p["hep_extract_ratio_conj_di"]          * dsi_out_au_cSBA  + p["hep_extract_ratio_unconj_di"]        * (dsi_out_au_uSBA  + dsi_out_pu_uSBA  + co_out_pu_uSBA)
    pl_in_from_gut_cCA    = (1 - p["hep_extract_ratio_conj_tri"])   * dsi_out_au_cCA
    pl_in_from_gut_cCDCA  = (1 - p["hep_extract_ratio_conj_di"])    * dsi_out_au_cCDCA
    pl_in_from_gut_cSBA   = (1 - p["hep_extract_ratio_conj_di"])    * dsi_out_au_cSBA
    pl_in_from_gut_uCA    =                                                              (1 - p["hep_extract_ratio_unconj_tri"]) * (dsi_out_au_uCA   + dsi_out_pu_uCA   + co_out_pu_uCA)
    pl_in_from_gut_uCDCA  =                                                              (1 - p["hep_extract_ratio_unconj_di"])  * (dsi_out_au_uCDCA + dsi_out_pu_uCDCA + co_out_pu_uCDCA)
    pl_in_from_gut_uSBA   =                                                              (1 - p["hep_extract_ratio_unconj_di"])  * (dsi_out_au_uSBA  + dsi_out_pu_uSBA  + co_out_pu_uSBA)

    co_out_sulfate_cSBA     = p["co_sulfate_freq"] * p["co_transit_coef"] * x_co_cSBA
    co_out_sulfate_uSBA     = p["co_sulfate_freq"] * p["co_transit_coef"] * x_co_uSBA
    fe_in_sulfate_cSBA     = co_out_sulfate_cSBA
    fe_in_sulfate_uSBA     = co_out_sulfate_uSBA

    p["bdl_discount"]     = min(p["bd_max_flow"] / (p["bd_to_dsi_freq"] * p["bd_transit_coef"] * (x_bd_cCA + x_bd_cCDCA + x_bd_cSBA)), 1)
                          
    bd_out_to_dsi_cCA     = p["bdl_discount"] * p["bd_to_dsi_freq"] * p["bd_transit_coef"] * x_bd_cCA
    bd_out_to_dsi_cCDCA   = p["bdl_discount"] * p["bd_to_dsi_freq"] * p["bd_transit_coef"] * x_bd_cCDCA
    bd_out_to_dsi_cSBA    = p["bdl_discount"] * p["bd_to_dsi_freq"] * p["bd_transit_coef"] * x_bd_cSBA
    dsi_in_from_bd_cCA    = bd_out_to_dsi_cCA
    dsi_in_from_bd_cCDCA  = bd_out_to_dsi_cCDCA
    dsi_in_from_bd_cSBA   = bd_out_to_dsi_cSBA

    p["bd_backflow_coef"] = max((x_bd_cCA + x_bd_cCDCA + x_bd_cSBA + (bd_in_from_li_cCA - bd_out_to_dsi_cCA) + (bd_in_from_li_cCDCA - bd_out_to_dsi_cCDCA) + (bd_in_from_li_cSBA  - bd_out_to_dsi_cSBA) - p["bd_max_ba"]) / (x_bd_cCA + x_bd_cCDCA + x_bd_cSBA), 0)

    bd_out_to_li_cCA      = p["bd_backflow_coef"] * x_bd_cCA
    bd_out_to_li_cCDCA    = p["bd_backflow_coef"] * x_bd_cCDCA
    bd_out_to_li_cSBA     = p["bd_backflow_coef"] * x_bd_cSBA
    li_in_from_bd_cCA     = bd_out_to_li_cCA
    li_in_from_bd_cCDCA   = bd_out_to_li_cCDCA
    li_in_from_bd_cSBA    = bd_out_to_li_cSBA

    ##### Defining ODEs #####

    dx_li_cCA    = + li_in_synthesis_cCA   + li_in_from_gut_cCA   + li_in_from_pl_cCA   - li_out_to_bd_cCA   + li_in_from_bd_cCA
    dx_li_cCDCA  = + li_in_synthesis_cCDCA + li_in_from_gut_cCDCA + li_in_from_pl_cCDCA - li_out_to_bd_cCDCA + li_in_from_bd_cCDCA
    dx_li_cSBA   =                         + li_in_from_gut_cSBA  + li_in_from_pl_cSBA  - li_out_to_bd_cSBA  + li_in_from_bd_cSBA
    dx_li_uCA    = 0
    dx_li_uCDCA  = 0
    dx_li_uSBA   = 0

    dx_bd_cCA    = + bd_in_from_li_cCA   - bd_out_to_dsi_cCA   - bd_out_to_li_cCA
    dx_bd_cCDCA  = + bd_in_from_li_cCDCA - bd_out_to_dsi_cCDCA - bd_out_to_li_cCDCA
    dx_bd_cSBA   = + bd_in_from_li_cSBA  - bd_out_to_dsi_cSBA  - bd_out_to_li_cSBA
    dx_bd_uCA    = 0
    dx_bd_uCDCA  = 0
    dx_bd_uSBA   = 0    
    
    dx_dsi_cCA   = + dsi_in_from_bd_cCA   - dsi_out_deconj_cCA   - dsi_out_au_cCA                      - dsi_out_to_co_cCA   
    dx_dsi_cCDCA = + dsi_in_from_bd_cCDCA - dsi_out_deconj_cCDCA - dsi_out_au_cCDCA                    - dsi_out_to_co_cCDCA 
    dx_dsi_cSBA  = + dsi_in_from_bd_cSBA  - dsi_out_deconj_cSBA  - dsi_out_au_cSBA                     - dsi_out_to_co_cSBA  
    dx_dsi_uCA   =                        + dsi_in_deconj_uCA    - dsi_out_au_uCA   - dsi_out_pu_uCA   - dsi_out_to_co_uCA   
    dx_dsi_uCDCA =                        + dsi_in_deconj_uCDCA  - dsi_out_au_uCDCA - dsi_out_pu_uCDCA - dsi_out_to_co_uCDCA 
    dx_dsi_uSBA  =                        + dsi_in_deconj_uSBA   - dsi_out_au_uSBA  - dsi_out_pu_uSBA  - dsi_out_to_co_uSBA  

    dx_co_cCA   = + co_in_from_dsi_cCA   - co_out_deconj_cCA                                            - co_out_to_fe_cCA   
    dx_co_cCDCA = + co_in_from_dsi_cCDCA - co_out_deconj_cCDCA                                          - co_out_to_fe_cCDCA 
    dx_co_cSBA  = + co_in_from_dsi_cSBA  - co_out_deconj_cSBA                                           - co_out_to_fe_cSBA  - co_out_sulfate_cSBA
    dx_co_uCA   = + co_in_from_dsi_uCA   + co_in_deconj_uCA    - co_out_biotr_uCA   - co_out_pu_uCA     - co_out_to_fe_uCA   
    dx_co_uCDCA = + co_in_from_dsi_uCDCA + co_in_deconj_uCDCA  - co_out_biotr_uCDCA - co_out_pu_uCDCA   - co_out_to_fe_uCDCA 
    dx_co_uSBA  = + co_in_from_dsi_uSBA  + co_in_deconj_uSBA   + co_in_biotr_uSBA   - co_out_pu_uSBA    - co_out_to_fe_uSBA  - co_out_sulfate_uSBA  

    dx_pl_cCA    = + pl_in_from_gut_cCA   - pl_out_to_li_cCA
    dx_pl_cCDCA  = + pl_in_from_gut_cCDCA - pl_out_to_li_cCDCA
    dx_pl_cSBA   = + pl_in_from_gut_cSBA  - pl_out_to_li_cSBA
    dx_pl_uCA    = + pl_in_from_gut_uCA   - pl_out_to_li_uCA
    dx_pl_uCDCA  = + pl_in_from_gut_uCDCA - pl_out_to_li_uCDCA
    dx_pl_uSBA   = + pl_in_from_gut_uSBA  - pl_out_to_li_uSBA

    # dx_fe_cCA    = + fe_in_from_co_cCA
    # dx_fe_cCDCA  = + fe_in_from_co_cCDCA
    # dx_fe_cSBA   = + fe_in_from_co_cSBA + fe_in_sulfate_cSBA
    # dx_fe_uCA    = + fe_in_from_co_uCA
    # dx_fe_uCDCA  = + fe_in_from_co_uCDCA
    # dx_fe_uSBA   = + fe_in_from_co_uSBA + fe_in_sulfate_uSBA

    dx = np.zeros(24)
    dx[0]  = dx_li_cCA
    dx[1]  = dx_li_cCDCA
    dx[2]  = dx_li_cSBA
    dx[3]  = dx_bd_cCA
    dx[4]  = dx_bd_cCDCA
    dx[5]  = dx_bd_cSBA
    dx[6]  = dx_dsi_cCA
    dx[7]  = dx_dsi_cCDCA
    dx[8]  = dx_dsi_cSBA
    dx[9]  = dx_dsi_uCA
    dx[10] = dx_dsi_uCDCA
    dx[11] = dx_dsi_uSBA
    dx[12] = dx_co_cCA
    dx[13] = dx_co_cCDCA
    dx[14] = dx_co_cSBA
    dx[15] = dx_co_uCA
    dx[16] = dx_co_uCDCA
    dx[17] = dx_co_uSBA    
    dx[18] = dx_pl_cCA
    dx[19] = dx_pl_cCDCA
    dx[20] = dx_pl_cSBA
    dx[21] = dx_pl_uCA
    dx[22] = dx_pl_uCDCA
    dx[23] = dx_pl_uSBA
    # dx[24] = dx_fe_cCA
    # dx[25] = dx_fe_cCDCA
    # dx[26] = dx_fe_cSBA
    # dx[27] = dx_fe_uCA
    # dx[28] = dx_fe_uCDCA
    # dx[29] = dx_fe_uSBA

    return dx