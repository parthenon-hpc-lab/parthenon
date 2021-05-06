import numpy as np

################################################################################
#   Sample error functions
#   
#   Each error function accepts the gold and test data in the format
#   and returns a error metric
################################################################################

def l1_err(gold,test):
    return np.sum(np.abs(gold-test))
def l2_err(gold,test):
    return np.linalg.norm(gold-test)
def linf_err(gold,test):
    return np.max(np.abs(gold-test))

def l1_rel_err(gold,test):
    mask = (gold != 0)
    return np.sum(np.abs((gold[mask]-test[mask])/gold[mask]))
def l2_rel_err(gold,test):
    mask = (gold != 0)
    return np.linalg.norm((gold[mask]-test[mask])/gold[mask])
def linf_rel_err(gold,test):
    mask = (gold != 0)
    return np.max(np.abs((gold[mask]-test[mask])/gold[mask]))

################################################################################
#   Compare_analytic
#   
#   Compares data in filename to a dictionary of analytic functions for some
#   components
################################################################################

def compare_analytic(filename,analytic_components,
        err_func=l2_err,tol=1e-12,quiet=False):
    """Compares data in filename to analytic gold data in analytic_components.

    Arguments:
    filename: Filename of ".phdf" file to compare data
    analytic_components: Dictionary keying component names to analytic functions.
        Each analytic function in the dictionary takes the arguments:

        def analytic_func(X,Y,Z,t)

        where X,Y,Z comprise arrays of x,y,z coords to compute the analytic solution
        for a component at time t

    Keyword Arguments:
    err_func=l2_err: Error function that accepts the analytic solution and data and
                     returns an error metric
    tol=1e-12: Tolerance of error metric. 
    quiet: Set to true to supress printing errors exceeding tol

    Returns True if the error of all components is under the tolerance, otherwise
    returns False.
    """

    try:
        import phdf
    except ModuleNotFoundError:
        print("Couldn't find module to read Parthenon hdf5 files.")
        return False

    datafile = phdf.phdf(filename)

    #Dictionary of component_name:component[grid_idx,k,j,i]
    file_components = {}

    #Get ready to match vector components of variables with component names
    num_componentss = datafile.Info["NumVariables"]
    component_names = datafile.Info["VariableNames"].astype(str)
    idx_component_name = 0

    #Read all data from the file
    for var,num_components in zip(datafile.Info["DatasetNames"].astype(str),
                                  num_componentss):
        dataset = datafile.Get(var,flatten=False)

        if num_components != 1:
            #Assign vector components to file_components with component_name
            for idx_component in range(num_components):
                file_components[component_names[idx_component_name]] = \
                        dataset[:,:,:,:,idx_component]
                idx_component_name+=1
        else:
            #Assign dataset to file_components with component_name
            file_components[component_names[idx_component_name]] = dataset
            idx_component_name+=1

    #Generate location arrays for each grid

    #Location lists
    locations_x = datafile.x
    locations_y = datafile.y
    locations_z = datafile.z

    #loc[grid_idx,k,j,i]
    loc_shape = (locations_x.shape[0],
                 locations_z.shape[1],
                 locations_y.shape[1],
                 locations_x.shape[1])

    X = np.empty(loc_shape)
    Y = np.empty(loc_shape)
    Z = np.empty(loc_shape)
    for grid_idx in range(loc_shape[0]):
        Z[grid_idx],Y[grid_idx],X[grid_idx] = np.meshgrid(
                locations_z[grid_idx],
                locations_y[grid_idx],
                locations_x[grid_idx],
                indexing="ij")

    #Flatten the coordinate arrays, for simplicity
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    #Check all components for which an analytic version exists
    all_ok = True
    for component in analytic_components.keys():

        #Compute the analytic component at X,Y,Z
        analytic_component = analytic_components[component](X,Y,Z,datafile.Time)

        #Compute the error between the file and analytic component
        err = err_func(analytic_component,file_components[component].ravel())

        if err > tol:
            if not quiet:
                print(f"Component {component} in {filename} error {err} exceeds tolerance {tol}")
            all_ok = False

    return all_ok



