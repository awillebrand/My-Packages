import numpy as np
import pandas as pd
from ASEN_6080.Tools import Integrator, MeasurementMgr

class BatchLLSEstimator:
    def __init__(self, integrator : Integrator, measurement_mgr_list : list):
        """
        Initialize the Batch Least Squares Estimator.

        Parameters:
        integrator : Integrator
            An instance of the Integrator class for orbit propagation.
        measurement_mgr_list : list
            A list of MeasurementMgr instances for different ground stations.
        """
        self.integrator = integrator
        self.measurement_mgrs = measurement_mgr_list



