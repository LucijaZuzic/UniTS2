import pickle
import pandas as pd
import numpy as np

missing_size = 30

username = "G:/MarkovOtoTrak2/"
username = "C:/Users/lzuzi/Documents/GitHub/MarkovOtoTrak2/"

def change_angle(angle, name_file):
    
    file_with_ride = pd.read_csv(username + name_file) 
    
    x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
    y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

def get_sides_from_angle(longest, angle):
    return longest * np.cos(angle / 180 * np.pi), longest * np.sin(angle / 180 * np.pi)

with open("UNITS_result/predicted_all", 'rb') as file_object:
    predicted_all = pickle.load(file_object)  
    file_object.close()

with open("UNITS_result/y_test_all", 'rb') as file_object:
    y_test_all = pickle.load(file_object)  
    file_object.close()

with open("UNITS_result/ws_all", 'rb') as file_object:
    ws_all = pickle.load(file_object)  
    file_object.close()

predicted_long = dict()
predicted_lat = dict()

actual_long = dict()
actual_lat = dict()
 
for model_name in predicted_all["speed"]:
    
    print(model_name)
    print(ws_all["longitude_no_abs"][model_name], ws_all["latitude_no_abs"][model_name])
 
    actual_long[model_name] = dict()
    actual_lat[model_name] = dict()

    for k in y_test_all["longitude_no_abs"][model_name]:
        print(model_name, k, "actual")
        actual_long[model_name][k] = [0]
        actual_lat[model_name][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][0], ws_all["latitude_no_abs"][model_name][0])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][0]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][0]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            actual_long[model_name][k].append(actual_long[model_name][k][-1] + y_test_all["longitude_no_abs"][model_name][k][ix + long_offset])
            actual_lat[model_name][k].append(actual_lat[model_name][k][-1] + y_test_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name] = dict()
    predicted_lat[model_name] = dict()
        
    predicted_long[model_name]["long no abs"] = dict()
    predicted_lat[model_name]["lat no abs"] = dict()

    for k in predicted_all["longitude_no_abs"][model_name]:
        print(model_name, k, "long no abs")
        predicted_long[model_name]["long no abs"][k] = [0]
        predicted_lat[model_name]["lat no abs"][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][0], ws_all["latitude_no_abs"][model_name][0])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][0]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][0]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            predicted_long[model_name]["long no abs"][k].append(predicted_long[model_name]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][k][ix + long_offset])
            predicted_lat[model_name]["lat no abs"][k].append(predicted_lat[model_name]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name]["long speed dir"] = dict()
    predicted_lat[model_name]["lat speed dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed dir")
        predicted_long[model_name]["long speed dir"][k] = [0]
        predicted_lat[model_name]["lat speed dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][0], ws_all["direction"][model_name][0]), ws_all["time"][model_name][0])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][0]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][0]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][0]
        range_speed_time = len(y_test_all["speed"][model_name][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][k][ix + dir_offset_time], k))
            predicted_long[model_name]["long speed dir"][k].append(predicted_long[model_name]["long speed dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed dir"][k].append(predicted_lat[model_name]["lat speed dir"][k][-1] + new_lat)
            
    predicted_long[model_name]["long speed ones dir"] = dict()
    predicted_lat[model_name]["lat speed ones dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed ones dir")
        predicted_long[model_name]["long speed ones dir"][k] = [0]
        predicted_lat[model_name]["lat speed ones dir"][k] = [0]
    
        max_offset_speed_dir = max(ws_all["speed"][model_name][0], ws_all["direction"][model_name][0])
        speed_offset = max_offset_speed_dir - ws_all["speed"][model_name][0]
        dir_offset = max_offset_speed_dir - ws_all["direction"][model_name][0]
        range_speed = len(y_test_all["speed"][model_name][k]) - speed_offset
        range_dir = len(y_test_all["direction"][model_name][k]) - dir_offset
        min_range_speed_dir = min(range_speed, range_dir)

        for ix in range(min_range_speed_dir):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset] / 111 / 0.1 / 3600 * missing_size, change_angle(predicted_all["direction"][model_name][k][ix + dir_offset], k))
            predicted_long[model_name]["long speed ones dir"][k].append(predicted_long[model_name]["long speed ones dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed ones dir"][k].append(predicted_lat[model_name]["lat speed ones dir"][k][-1] + new_lat)
            
with open("UNITS_result/actual_long", 'wb') as file_object:
    pickle.dump(actual_long, file_object)  
    file_object.close()

with open("UNITS_result/actual_lat", 'wb') as file_object:
    pickle.dump(actual_lat, file_object)  
    file_object.close()

with open("UNITS_result/predicted_long", 'wb') as file_object:
    pickle.dump(predicted_long, file_object)  
    file_object.close()

with open("UNITS_result/predicted_lat", 'wb') as file_object:
    pickle.dump(predicted_lat, file_object)  
    file_object.close()