import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric_short

ndec = {"direction": 0, "time": 3, "speed": 0}

def transform_pd_file(pd_file):
    pd_file = pd_file.drop(labels = "date", axis = 1)
    pd_file = pd_file.to_numpy()
    return pd_file 

def transform_np_file(np_file):
    np_file = np.array(np_file).squeeze()
    return np_file

def find_match_X(transformed_pd_file, transformed_xs_file, transformed_true_file, transformed_pred_file, var_name, suf1, suf2, t_val, num_val):
    print("X", var_name, suf1)
    dict_xs_pred_true = dict()
    use_ndec = 5
    if var_name in ndec:
        use_ndec = ndec[var_name]

    for ix_xs in range(len(transformed_xs_file)):
        dict_xs_pred_key = np.round(transformed_xs_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        dict_xs_true_val = np.round(transformed_true_file[ix_xs], use_ndec)
        dict_xs_pred_val = np.round(transformed_pred_file[ix_xs], use_ndec)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
            dict_xs_true_val = dict_xs_true_val[-1]
            dict_xs_pred_val = dict_xs_pred_val[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            dict_xs_pred_true[dict_xs_pred_key] = []
        dict_xs_pred_true[dict_xs_pred_key].append([dict_xs_true_val, dict_xs_pred_val])
    
    doubled = []
    for ix_xs in range(len(transformed_xs_file)):
        dict_xs_pred_key = np.round(transformed_xs_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if len(dict_xs_pred_true[dict_xs_pred_key]) > 1:
            doubled.append(len(dict_xs_pred_true[dict_xs_pred_key]))

    #print("Double min max", min(doubled), max(doubled))
    #print("Double", len(doubled), len(transformed_true_file), np.round(len(doubled) / len(transformed_true_file) * 100, 2))

    notfound = 0
    maxgap = -1
    mingap = 1000000
    maxix = -1
    minix = 1000000
    lastix = 0
    newpred = []
    newtrue = []
    newpredmax = []
    newtruemax = []
    used_all = []
    candidates_all = []
    for ix_match in range(len(transformed_pd_file)):
        dict_xs_pred_key = np.round(transformed_pd_file[ix_match], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        else:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            notfound += 1
            gap = ix_match - lastix
            mingap = min(gap, mingap)
            maxgap = max(gap, maxgap)

            if "S" not in suf1:
                current_key = np.array(dict_xs_pred_key)
                key_array = np.array([list(ck) for ck in dict_xs_pred_true.keys()])
            else:
                current_key = np.array([dict_xs_pred_key])
                key_array = np.array([[ck] for ck in dict_xs_pred_true.keys()])
            key_diff = [abs(kd) for kd in list(key_array - current_key)]
            if "S" not in suf1:
                key_diff = [sum(kd) for kd in key_diff]
            min_key_diff = min(key_diff)
            ix_min_key_diff = key_diff.index(min_key_diff)
            min_key = key_array[ix_min_key_diff]
            dict_xs_pred_key = min_key
            if "S" not in suf1:
                dict_xs_pred_key = tuple(min_key)
            else:
                dict_xs_pred_key = min_key[0]
        else:
            lastix = ix_match
            minix = min(lastix, minix)
            maxix = max(lastix, maxix)

        candidates = dict_xs_pred_true[dict_xs_pred_key]
        ix_choose = 0
        min_error = abs(candidates[ix_choose][0] - candidates[ix_choose][1])
        ix_choose_max = 0
        max_error = abs(candidates[ix_choose_max][0] - candidates[ix_choose_max][1])
        if "S" not in suf1:
            min_error = sum(min_error)
            max_error = sum(max_error)
        for ix_c in range(len(candidates)):
            new_error = abs(candidates[ix_choose][0] - candidates[ix_choose][1])
            if "S" not in suf1:
                new_error = sum(new_error)
            if new_error < min_error:
                ix_choose = ix_c
                min_error = new_error
            if new_error > max_error:
                ix_choose_max = ix_c
                max_error = new_error
        newtrue.append(candidates[ix_choose][0])
        newpred.append(candidates[ix_choose][1])
        newtruemax.append(candidates[ix_choose_max][0])
        newpredmax.append(candidates[ix_choose_max][1])
        used_all.append(dict_xs_pred_key)
        candidates_all.append(candidates)
    
    #print("Gap", mingap, maxgap)
    #print(minix, maxix, len(transformed_pd_file))
    #print("Empty", notfound, len(transformed_pd_file), np.round(notfound / len(transformed_pd_file) * 100, 2))
    newpred, newtrue = np.array(newpred), np.array(newtrue)
    mae, mse, rmse = metric_short(newpred, newtrue)
    print(mae, mse, rmse)
    if not os.path.isdir("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/"):
        os.makedirs("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/")
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/min_true_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newtrue, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/min_pred_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newpred, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/max_true_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newtruemax, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/max_pred_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newpredmax, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/dict_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(dict_xs_pred_true, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/used_all_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(used_all, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/X/candidates_all_X_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(candidates_all, file_object)  
        file_object.close()

def find_match_Y(transformed_pd_file, transformed_true_file, transformed_pred_file, var_name, suf1, suf2, t_val, num_val):
    print("Y", var_name, suf1)
    dict_xs_pred_true = dict()
    use_ndec = 5
    if var_name in ndec:
        use_ndec = ndec[var_name]

    for ix_xs in range(len(transformed_true_file)):
        dict_xs_pred_key = np.round(transformed_true_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        dict_xs_pred_val = np.round(transformed_pred_file[ix_xs], use_ndec)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
            dict_xs_pred_val = dict_xs_pred_val[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            dict_xs_pred_true[dict_xs_pred_key] = []
        dict_xs_pred_true[dict_xs_pred_key].append(dict_xs_pred_val)
    
    doubled = []
    for ix_xs in range(len(transformed_true_file)):
        dict_xs_pred_key = np.round(transformed_true_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if len(dict_xs_pred_true[dict_xs_pred_key]) > 1:
            doubled.append(len(dict_xs_pred_true[dict_xs_pred_key]))

    #print("Double min max", min(doubled), max(doubled))
    #print("Double", len(doubled), len(transformed_true_file), np.round(len(doubled) / len(transformed_true_file) * 100, 2))

    notfound = 0
    maxgap = -1
    mingap = 1000000
    maxix = -1
    minix = 1000000
    lastix = 0
    newpred = []
    newtrue = []
    newpredmax = []
    newtruemax = []
    used_all = []
    candidates_all = []
    for ix_match in range(len(transformed_pd_file)):
        dict_xs_pred_key = np.round(transformed_pd_file[ix_match], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        else:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            notfound += 1
            gap = ix_match - lastix
            mingap = min(gap, mingap)
            maxgap = max(gap, maxgap)

            if "S" not in suf1:
                current_key = np.array(dict_xs_pred_key)
                key_array = np.array([list(ck) for ck in dict_xs_pred_true.keys()])
            else:
                current_key = np.array([dict_xs_pred_key])
                key_array = np.array([[ck] for ck in dict_xs_pred_true.keys()])
            key_diff = [abs(kd) for kd in list(key_array - current_key)]
            if "S" not in suf1:
                key_diff = [sum(kd) for kd in key_diff]
            min_key_diff = min(key_diff)
            ix_min_key_diff = key_diff.index(min_key_diff)
            min_key = key_array[ix_min_key_diff]
            dict_xs_pred_key = min_key
            if "S" not in suf1:
                dict_xs_pred_key = tuple(min_key)
            else:
                dict_xs_pred_key = min_key[0]
        else:
            lastix = ix_match
            minix = min(lastix, minix)
            maxix = max(lastix, maxix)

        candidates = dict_xs_pred_true[dict_xs_pred_key]
        ix_choose = 0
        min_error = abs(candidates[ix_choose] - dict_xs_pred_key)
        ix_choose_max = 0
        max_error = abs(candidates[ix_choose_max] - dict_xs_pred_key)
        if "S" not in suf1:
            min_error = sum(min_error)
            max_error = sum(max_error)
        for ix_c in range(len(candidates)):
            new_error = abs(candidates[ix_c] - dict_xs_pred_key)
            if "S" not in suf1:
                new_error = sum(new_error)
            if new_error < min_error:
                ix_choose = ix_c
                min_error = new_error
            if new_error > max_error:
                ix_choose_max = ix_c
                max_error = new_error
        newtrue.append(dict_xs_pred_key)
        newpred.append(candidates[ix_choose])
        newtruemax.append(dict_xs_pred_key)
        newpredmax.append(candidates[ix_choose_max])
        used_all.append(dict_xs_pred_key)
        candidates_all.append(candidates)
    
    #print("Gap", mingap, maxgap)
    #print(minix, maxix, len(transformed_pd_file))
    #print("Empty", notfound, len(transformed_pd_file), np.round(notfound / len(transformed_pd_file) * 100, 2))
    newpred, newtrue = np.array(newpred), np.array(newtrue)
    mae, mse, rmse = metric_short(newpred, newtrue)
    print(mae, mse, rmse)
    if not os.path.isdir("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/"):
        os.makedirs("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/")
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/min_true_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newtrue, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/min_pred_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newpred, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/max_true_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newtruemax, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/max_pred_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(newpredmax, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/dict_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(dict_xs_pred_true, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/used_all_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(used_all, file_object)  
        file_object.close()
    with open("results_eval/" + varname + "/" + suf1 + suf2 + str(num_val) + "_" + t_val + "/Y/candidates_all_Y_" + suf1 + suf2 + str(num_val) + "_" + t_val, 'wb') as file_object:
        pickle.dump(candidates_all, file_object)  
        file_object.close()
    
varnames = ["direction", "speed", "time", "longitude_no_abs", "latitude_no_abs"]
first_sufix = ["S_", "MS_"]
second_sufix = ["all_"]
types_used = ["test"]

second_sufix = [""]
types_used = ["val"]

for num in range(6, 11): 

    for varname in varnames: 

        print(varname, num)

        for s1 in first_sufix:
            for s2 in second_sufix:
                for t in types_used:
                    print(s1, s2, t)
 
                    pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
                    pd_file_val_transformed = transform_pd_file(pd_file_val)  

                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "xs_transformed_" + varname, 'rb') as file_object:
                        xs_val = pickle.load(file_object) 
                        file_object.close()
                    xs_val = transform_np_file(xs_val)

                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "trues_transformed_" + varname, 'rb') as file_object:
                        trues_val = pickle.load(file_object) 
                        file_object.close()
                    trues_val = transform_np_file(trues_val) 
            
                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "preds_transformed_" + varname, 'rb') as file_object:
                        preds_val = pickle.load(file_object) 
                        file_object.close()
                    preds_val = transform_np_file(preds_val)

                    find_match_X(pd_file_val_transformed, xs_val, trues_val, preds_val, varname, s1, s2, t, num)
                    find_match_Y(pd_file_val_transformed, trues_val, preds_val, varname, s1, s2, t, num)