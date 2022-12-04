from pyoptsparse import OPT, Optimization, History
import numpy as np

hist = History("/home/hschsc/job_dir/LST_opt/code/snopt_LCO_Hist.hst", flag="r")
xdict = hist.getValues(names='xvars', major=True)

filename = "opt_hist_spring.dat"
np.savetxt(filename, xdict['xvars'])