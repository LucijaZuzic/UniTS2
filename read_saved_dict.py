import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric_short
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
 
mode = "min"
ndec = {"direction": 0, "time": 3, "speed": 0}

def transform_pd_file(pd_file):
    pd_file = np.array(pd_file["OT"])
    return pd_file 

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

varnames = ["direction", "speed", "time", "longitude_no_abs", "latitude_no_abs"]
first_sufix = ["S_", "MS_"]
t = "val"
s2 = ""

chose_vals = dict()
for varname in varnames:

    with open("actual/actual_" + varname, 'rb') as file_object:
        file_object_test = pickle.load(file_object)

    all_mine_skip_flat = []
    for filename in file_object_test: 
        for val in file_object_test[filename]:
            all_mine_skip_flat.append(val)

    min_combo = (varname, -1, -1, -1, -1)
    min_combo_val = 1000000
    for num in range(6, 11):
        for s1 in first_sufix:
                #print(varname, num, s1, s2, t)

                pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
                pd_file_val_transformed = transform_pd_file(pd_file_val)  

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/candidates_all_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    preds_val = pickle.load(file_object)  
                    file_object.close()

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/used_all_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    trues_val = pickle.load(file_object)  
                    file_object.close()
                    
                preds_val = [np.average(val) for val in preds_val]

                #mae, mse, rmse = metric_short(preds_val, trues_val)
                #print(mae, mse, rmse)
                mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
                #print(mae, mse, rmse)

                final_train_MAE = mean_absolute_error(preds_val, pd_file_val_transformed)
                final_train_R2 = r2_score(preds_val, pd_file_val_transformed)
                final_train_RMSE = math.sqrt(mean_squared_error(preds_val, pd_file_val_transformed) / (max(all_mine_skip_flat) - min(all_mine_skip_flat)))
                #print(final_train_MAE, final_train_R2, final_train_RMSE)

                if final_train_RMSE < min_combo_val:
                    min_combo_val = final_train_RMSE
                    min_combo = (varname, num, s1, s2, t)

    print(varname, min_combo, min_combo_val)
    chose_vals[varname] = (min_combo[1], min_combo[2])

t = "test"
s2 = "all_"
predicted_all = dict()
y_test_all = dict()
ws_all = dict()
model_name = "UNITS"
for varname in chose_vals:

    with open("actual/actual_" + varname, 'rb') as file_object:
        file_object_test = pickle.load(file_object)

    all_mine_skip_flat = []
    for filename in file_object_test: 
        for val in file_object_test[filename]:
            all_mine_skip_flat.append(val)

    predicted_all[varname] = dict()
    predicted_all[varname][model_name] = dict()
    y_test_all[varname] = dict()
    y_test_all[varname][model_name] = dict()
    ws_all[varname] = dict()
    ws_all[varname][model_name] = chose_vals[varname]
    num, s1 = chose_vals[varname]
    
    pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
    pd_file_val_transformed = transform_pd_file(pd_file_val)  

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/candidates_all_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        preds_val = pickle.load(file_object)  
        file_object.close()

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/used_all_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        trues_val = pickle.load(file_object)  
        file_object.close()

    preds_val = [np.average(val) for val in preds_val]
 
    with open("actual/actual_" + varname, 'rb') as file_object:
        file_object_test = pickle.load(file_object)

    lens = []
        
    for k in file_object_test:

        x_test_part, y_test_part = get_XY(file_object_test[k], num, 1, 1)
        
        predicted_all[varname][model_name][k] = preds_val[sum(lens):sum(lens) + len(x_test_part)]
        y_test_all[varname][model_name][k] = trues_val[sum(lens):sum(lens) + len(x_test_part)]
        lens.append(len(x_test_part))

    #mae, mse, rmse = metric_short(preds_val, trues_val)
    #print(varname, mae, mse, rmse)
    mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
    #print(varname, mae, mse, rmse)

    final_train_MAE = mean_absolute_error(preds_val, pd_file_val_transformed)
    final_train_R2 = r2_score(preds_val, pd_file_val_transformed)
    final_train_RMSE = math.sqrt(mean_squared_error(preds_val, pd_file_val_transformed) / (max(all_mine_skip_flat) - min(all_mine_skip_flat)))
    print(varname, final_train_MAE, np.round(final_train_R2 * 100, 2), np.round(final_train_RMSE * 100, 2))
    
if not os.path.isdir("UNITS_result/"):
    os.makedirs("UNITS_result/")

with open("UNITS_result/predicted_all", 'wb') as file_object:
    pickle.dump(predicted_all, file_object)  
    file_object.close()

with open("UNITS_result/y_test_all", 'wb') as file_object:
    pickle.dump(y_test_all, file_object)  
    file_object.close()

with open("UNITS_result/ws_all", 'wb') as file_object:
    pickle.dump(ws_all, file_object)  
    file_object.close()