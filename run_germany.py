# -*- coding: utf-8 -*-
"""
Configurable Germany Unit Commitment (UC) Model Runner
Edit the configuration values below and run: python run_germany.py
"""

# ========== CONFIGURATION ==========
# Edit these values to configure your simulation

YEAR = 2033             # Options: 2025, 2033
CLIMATIC_YEAR = 2016     # Options: 1982, 1989, 1996, 2003, 2010, 2016
START_MONTH = 1         # Month (1-12)
START_DAY = 1          # Day of month
NUM_DAYS = 2             # Simulation duration in days (Quick test to verify storage plots)

# ===================================

"""
  0) Preliminar functional aspects / technical functions
"""
# (0.a) Generate simulation run ID for organized output
from datetime import datetime, timedelta

# Calculate end date for the folder name
_temp_start = datetime(year=1900, month=START_MONTH, day=START_DAY)
_temp_end = _temp_start + timedelta(days=NUM_DAYS)
_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
_simulation_run_id = f'{YEAR}_cy{CLIMATIC_YEAR}_{START_MONTH:02d}{START_DAY:02d}-{_temp_end.month:02d}{_temp_end.day:02d}_{_timestamp}'

# Set the simulation run ID for output organization
from common.long_term_uc_io import set_simulation_run_id
set_simulation_run_id(_simulation_run_id)

print(f'=== Starting Germany UC Simulation ===')
print(f'Run ID: {_simulation_run_id}')
print(f'Year: {YEAR}, Climatic Year: {CLIMATIC_YEAR}')
print(f'Period: {START_MONTH}/{START_DAY} for {NUM_DAYS} days')
print(f'Output folder: output/long_term_uc/monozone_ger/{_simulation_run_id}/')
print('=' * 50)

# (0.b) deactivate some verbose warnings (mainly in pandas, pypsa)
from common.logger import deactivate_verbose_warnings
deactivate_verbose_warnings(deact_deprecation_warn=True)

# (0.b) Function to get an object describing ERAA dataset (mainly available values)
from common.constants.extract_eraa_data import ERAADatasetDescr


def get_eraa_data_description() -> ERAADatasetDescr:
    from common.long_term_uc_io import get_json_fixed_params_file
    json_fixed_params_file = get_json_fixed_params_file()
    from utils.read import check_and_load_json_file
    json_params_fixed = check_and_load_json_file(json_file=json_fixed_params_file,
                                                 file_descr='JSON fixed params')
    json_available_values_dummy = {'available_climatic_years': None,
                                   'available_countries': None,
                                   'available_aggreg_prod_types': None,
                                   'available_intercos': None,
                                   'available_target_years': None}
    json_params_fixed |= json_available_values_dummy

    return ERAADatasetDescr(**json_params_fixed)


# Function to get a few parameters for plot -> only style, in particular to set fixed colors
# per (aggreg.) production type/country
from common.plot_params import PlotParams


def get_plots_params() -> (PlotParams, PlotParams):
    from utils.read import read_plot_params
    per_dim_plot_params = read_plot_params()
    from utils.read import read_given_phase_plot_params
    fig_style = read_given_phase_plot_params(phase_name=phase_name)
    from utils.basic_utils import print_non_default
    print_non_default(obj=fig_style, obj_name=f'FigureStyle - for phase {phase_name}', log_level='debug')
    return per_dim_plot_params[DataDimensions.agg_prod_type], per_dim_plot_params[DataDimensions.zone]


# name of current "phase" (of the course), the one associated to this script: a 1-zone Unit Commitment model
from common.constants.usage_params_json import EnvPhaseNames
phase_name = EnvPhaseNames.monozone_toy_uc_model

"""
  I) Set global parameters for the case simulated
"""
# Use configuration from top of file
country = 'germany'
year = YEAR
climatic_year = CLIMATIC_YEAR

# Set start and end date from configuration (datetime already imported above)
uc_period_start = datetime(year=1900, month=START_MONTH, day=START_DAY)
uc_period_end = uc_period_start + timedelta(days=NUM_DAYS)

from common.constants.prod_types import ProdTypeNames
agg_prod_types_selec = [ProdTypeNames.wind_onshore, ProdTypeNames.wind_offshore, ProdTypeNames.solar_pv]

"""
  II) Initialize two objects - used hereafter to have clean and 'robust' code operations/functions
"""
# (II.a) UC main run parameters (dictionary gathering main characteristics of the pb simulated)
from common.uc_run_params import UCRunParams
selected_countries = [country]
uc_run_params = UCRunParams(selected_countries=selected_countries, selected_target_year=year,
                            selected_climatic_year=climatic_year,
                            selected_prod_types={country: agg_prod_types_selec},
                            uc_period_start=uc_period_start,
                            uc_period_end=uc_period_end)

# (II.b) Dataset object
from include.dataset import Dataset
eraa_data_descr = get_eraa_data_description()
eraa_dataset = Dataset(agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data)

"""
  III) Get needed data - from ERAA csv files in data\ERAA_2023-2
"""
# Get data for Germany
eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def)
eraa_dataset.complete_data()

# Accessing the data: globally all is made with pandas dataframes (df)
from utils.df_utils import selec_in_df_based_on_list
from common.constants.prod_types import ProdTypeNames
prod_type_col = 'production_type_agg'
solar_pv_cf_data = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col=prod_type_col,
                                       selec_vals=[ProdTypeNames.solar_pv], rm_selec_col=True)
}
wind_on_shore_cf_data = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col=prod_type_col,
                                       selec_vals=[ProdTypeNames.wind_onshore], rm_selec_col=True)
}
wind_off_shore_cf_data = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col=prod_type_col,
                                       selec_vals=[ProdTypeNames.wind_offshore], rm_selec_col=True)
}

# Load hydro time-series data (Phase 2 & 3: Time-series integration + Constraints)
from utils.eraa_data_reader import get_hydro_ror_generation, get_hydro_inflows, get_hydro_level_constraints
from common.long_term_uc_io import INPUT_ERAA_FOLDER

hydro_folder = INPUT_ERAA_FOLDER
# Load run-of-river generation profile
hydro_ror_profile = get_hydro_ror_generation(
    folder=hydro_folder,
    zone=country,
    climatic_year=climatic_year,
    period_start=uc_run_params.uc_period_start,
    period_end=uc_run_params.uc_period_end
)

# Load hydro inflows for reservoir and pump storage open loop
hydro_inflows = get_hydro_inflows(
    folder=hydro_folder,
    zone=country,
    climatic_year=climatic_year,
    period_start=uc_run_params.uc_period_start,
    period_end=uc_run_params.uc_period_end
)

# Load hydro level constraints (Phase 3)
hydro_level_constraints = get_hydro_level_constraints(
    folder=hydro_folder,
    zone=country,
    climatic_year=climatic_year,
    period_start=uc_run_params.uc_period_start,
    period_end=uc_run_params.uc_period_end
)

"""
  IV) Build PyPSA model - with unique country (Germany here)
"""
# (IV.1) Initialize PyPSA Network
print('Initialize PyPSA network')
from include.dataset_builder import set_country_trigram
country_trigram = set_country_trigram(country=country)
from include.dataset_builder import PypsaModel
pypsa_model = PypsaModel(name=f'my 1-zone {country_trigram} toy model')

import pandas as pd
date_idx = eraa_dataset.demand[uc_run_params.selected_countries[0]].index
horizon = pd.date_range(
    start=uc_run_params.uc_period_start.replace(year=uc_run_params.selected_target_year),
    end=uc_run_params.uc_period_end.replace(year=uc_run_params.selected_target_year),
    freq='h'
)
pypsa_model.init_pypsa_network(date_idx=date_idx, date_range=horizon)
print(pypsa_model.network)

# (IV.2) Define Germany parameters
# (IV.2.i) Add bus for considered country
# Import parameters based on year
if year == 2025:
    from toy_model_params.germany_parameters_2025 import gps_coords, get_generators, set_gen_as_list_of_gen_units_data
elif year == 2033:
    from toy_model_params.germany_parameters_2033 import gps_coords, get_generators, set_gen_as_list_of_gen_units_data
else:
    raise ValueError(f"Unsupported year: {year}. Choose 2025 or 2033")

coordinates = {country: gps_coords}
pypsa_model.add_gps_coordinates(countries_gps_coords=coordinates)

# (IV.2.ii) Generators definition
from common.fuel_sources import set_fuel_sources_from_json, DUMMY_FUEL_SOURCES
fuel_sources = set_fuel_sources_from_json()

generators = get_generators(country_trigram=country_trigram, fuel_sources=fuel_sources,
                            wind_on_shore_cf_data=wind_on_shore_cf_data[country],
                            wind_off_shore_cf_data=wind_off_shore_cf_data[country],
                            solar_pv_cf_data=solar_pv_cf_data[country],
                            hydro_ror_profile=hydro_ror_profile,
                            hydro_inflows=hydro_inflows,
                            hydro_level_constraints=hydro_level_constraints)
generation_units_data = set_gen_as_list_of_gen_units_data(generators=generators)
eraa_dataset.set_generation_units_data(gen_units_data={country: generation_units_data})

fuel_sources |= DUMMY_FUEL_SOURCES
pypsa_model.add_energy_carriers(fuel_sources=fuel_sources)
pypsa_model.add_per_bus_energy_carriers(fuel_sources=fuel_sources)
pypsa_model.add_generators(generators_data=eraa_dataset.generation_units_data)

# IV.2.iii) Add load
pypsa_model.add_loads(demand=eraa_dataset.demand)

# IV.2.iv) Check/observe that created PyPSA model be coherent
print(f'PyPSA network main properties: {pypsa_model.network}')
pypsa_model.plot_network(toy_model_output=True, country=country)
print(pypsa_model.network.generators)

"""
  V) 'Optimize network' i.e., solve the associated (1-zone) Unit-Commitment problem
"""
# (V.1) Set solver to be used
# Load solver params from solver_params.json to use Gurobi (or other configured solver)
from utils.read import read_solver_params
solver_params = read_solver_params()
pypsa_model.set_optim_solver(solver_params=solver_params)

# (V.2) Save lp model, then solve optimisation model
result = pypsa_model.optimize_network(year=uc_run_params.selected_target_year, n_countries=1,
                                      period_start=uc_run_params.uc_period_start, save_lp_file=True,
                                      toy_model_output=True, countries=[country])
print(f'PyPSA result: {result}')

"""
  VI) Analyse/plot obtained UC solution
"""
from common.constants.optimisation import OPTIM_RESOL_STATUS
optim_status = result[1]
pypsa_opt_resol_status = OPTIM_RESOL_STATUS.optimal

if optim_status == pypsa_opt_resol_status:
    objective_value = pypsa_model.get_opt_value(pypsa_resol_status=pypsa_opt_resol_status)
    print(f'Total cost at optimum: {objective_value:.2f}')
    
    pypsa_model.get_prod_var_opt()
    pypsa_model.get_storage_vars_opt()
    pypsa_model.get_sde_dual_var_opt()

    print('Plot installed capacities (INPUT parameter), generation and prices (optimisation OUTPUTS) figures')
    pypsa_model.plot_installed_capas(country=country, year=uc_run_params.selected_target_year, toy_model_output=True)
    
    from common.constants.datadims import DataDimensions
    plot_params_agg_pt, plot_params_zone = get_plots_params()
    pypsa_model.plot_opt_prod_var(plot_params_agg_pt=plot_params_agg_pt, country=country,
                                  year=uc_run_params.selected_target_year,
                                  climatic_year=uc_run_params.selected_climatic_year,
                                  start_horizon=uc_run_params.uc_period_start,
                                  toy_model_output=True)
    
    pypsa_model.plot_failure_at_opt(country=country, year=uc_run_params.selected_target_year,
                                    climatic_year=uc_run_params.selected_climatic_year,
                                    start_horizon=uc_run_params.uc_period_start,
                                    toy_model_output=True)
    
    pypsa_model.plot_marginal_price(plot_params_zone=plot_params_zone, year=uc_run_params.selected_target_year,
                                    climatic_year=uc_run_params.selected_climatic_year,
                                    start_horizon=uc_run_params.uc_period_start, toy_model_output=True,
                                    country=country)

    print('Save optimal dispatch decisions to .csv file')
    pypsa_model.save_opt_decisions_to_csv(year=uc_run_params.selected_target_year,
                                          climatic_year=uc_run_params.selected_climatic_year,
                                          start_horizon=uc_run_params.uc_period_start, toy_model_output=True,
                                          country=country)

    pypsa_model.save_marginal_prices_to_csv(year=uc_run_params.selected_target_year,
                                            climatic_year=uc_run_params.selected_climatic_year,
                                            start_horizon=uc_run_params.uc_period_start, toy_model_output=True,
                                            country=country)
else:
    print(f'Optimisation resolution status is not {pypsa_opt_resol_status} '
          f'-> output data (resp. figures) cannot be saved (resp. plotted), excepting installed capas one')
    pypsa_model.plot_installed_capas(country=country, year=uc_run_params.selected_target_year,
                                     toy_model_output=True)

print('=' * 50)
print(f'THE END of ERAA-PyPSA long-term UC toy model of country {country} simulation!')
print(f'Run ID: {_simulation_run_id}')
print(f'Results saved to: output/long_term_uc/monozone_ger/{_simulation_run_id}/')
print('=' * 50)

