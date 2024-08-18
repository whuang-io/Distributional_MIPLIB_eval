import gurobipy as gp
import os
import numpy as np
import pandas as pd
import argparse
import pickle
np.random.seed(0)


def run_bnb_gurobi(model, time_limit):
    '''
    Function to obtain performance metrics for classifying hardness levels: solving time, primal-dual gap, primal-dual integral
    '''
    def data_cb(model, where):
        if ((where == gp.GRB.Callback.MIP)):
            # see gurobi callback codes: https://www.gurobi.com/documentation/current/refman/cb_codes.html

            objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            #gap = abs((objbst - objbnd) / objbst)
            cur_time = model.cbGet(gp.GRB.Callback.RUNTIME)

            L_t.append(cur_time)
            L_objbst.append(objbst)
            L_objbnd.append(objbnd)
            L_gap.append(abs(objbst - objbnd) / abs(objbst))
            if len(L_t)>1:
               delta_t = L_t[-1]-L_t[-2]
            else:
               delta_t = L_t[-1]
            integral[0] += delta_t * L_gap[-1]
                

    L_t = []
    L_objbst = []
    L_objbnd = []
    L_gap = []
    integral=[0]
    
    #model._best_obj = np.inf if (model.ModelSense == 1) else np.NINF   # model.ModelSense == 1: minimization   # -1 means maximization
    model.setParam('TimeLimit', time_limit)
    model.optimize(data_cb)

    return {'t':L_t, 'objbst':L_objbst, 'objbnd':L_objbnd, 'gap': L_gap}, model.status, L_gap[-1], integral[0], model.Runtime


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type = str)
    args = parser.parse_args()
    inst_root = "/project/dilkina_438/weiminhu/miplib/instances"

    with open (args.file, 'rb') as fp:
        insts = pickle.load(fp)

      
    for inst in insts:
        inst_name=os.path.basename(inst)

        if not os.path.exists("/"+os.path.join(*inst.split("/")[:-2],"gurobi_results")):
            os.mkdir("/"+os.path.join(*inst.split("/")[:-2],"gurobi_results"))
            
        model = gp.read(inst)
        res, status, gap, integral, time = run_bnb_gurobi(model,time_limit=3600)

        formats = ["mps","lp","proto.lp","gz","mps.gz"]
        if inst_name.endswith("mps"):
            suffix=".mps"
        if inst_name.endswith("lp"):
            suffix=".lp"
        if inst_name.endswith("proto.lp"):
            suffix=".proto.lp"
        if inst_name.endswith("mps.gz"):
            suffix=".mps.gz"
        if inst_name.endswith("gz"):
            suffix=".gz"
        df_res=pd.DataFrame.from_dict(res)
        df_res.to_csv("/"+os.path.join(*inst.split("/")[:-2],"gurobi_results",inst_name.split(suffix)[0]+"+full_df"+".csv"))
        df_temp = pd.DataFrame([status, gap, integral, time])
        df_temp.to_csv("/"+os.path.join(*inst.split("/")[:-2],"gurobi_results",inst_name.split(suffix)[0]+"_brief"+".csv"), index=False)
        
