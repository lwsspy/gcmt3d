import os
from lwsspy.utils.io import read_yaml_file

scriptdir = os.path.dirname(os.path.abspath(__file__))


class Constants:

    # Locations
    # ---------
    processdict = read_yaml_file(os.path.join(scriptdir, "process.yml"))
    inputfilename = os.path.join(os.path.join(scriptdir, "input.yml"))

    # Parameter lists:
    # ----------------
    # This parameters we know the frechet derivative computation of
    parameter_check_list = ['depth_in_m', "time_shift", 'latitude', 'longitude',
                            "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    # Parameters that do not need simulation
    nosimpars = ["time_shift", "half_duration"]

    # Parameters related to the hypocenter
    hypo_pars = ['depth_in_m', "time_shift", 'latitude', 'longitude']

    # Parameters that are related to the moment tensor
    mt_params = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
