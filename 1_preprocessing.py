import pickle
import glob
import time
import random
random.seed(10)
import hickle as hkl
import argparse


from utils import *
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="5G DDoS")

parser.add_argument("--flow_length", type=int, default=10, required=True)

parser.add_argument("--time_window", type=int, default=10, required=True)
args = parser.parse_args()

preprocessed_flows = []

filelist = glob.glob('data_file_'+str(args.flow_length)+'/*.data')

for file in filelist:
    with open(file, 'rb') as filehandle:
        # read the data as binary data stream
        preprocessed_flows = preprocessed_flows + pickle.load(filehandle)

def count_flows(preprocessed_flows):
    ddos_flows = 0
    total_flows = len(preprocessed_flows)
    ddos_fragments = 0
    total_fragments = 0
        
    for flow in preprocessed_flows:
        flow_fragments = len(flow[1]) - 1
        total_fragments += flow_fragments
        if flow[1]['label'] > 0:
            ddos_flows += 1
            ddos_fragments += flow_fragments  # the label does not count

    return (total_flows, ddos_flows, total_flows - ddos_flows), (total_fragments, ddos_fragments, total_fragments-ddos_fragments)

(total_flows, ddos_flows, benign_flows),  (total_fragments, ddos_fragments, benign_fragments) = count_flows(preprocessed_flows)

log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                ") | fragments (tot,ben,ddos):(" + str(total_fragments) + "," + str(benign_fragments) + "," + str(ddos_fragments) + \
                ")  |\n"
print(log_string)

# balance the dataset based on the number of benign and malicious fragments of flows
def balance_dataset(flows,total_fragments=float('inf')):
    new_flow_list = []

    _,(_, ddos_fragments, benign_fragments) = count_flows(flows)

    if ddos_fragments == 0 or benign_fragments == 0:
        min_fragments = total_fragments
    else:
        min_fragments = min(total_fragments/2,ddos_fragments,benign_fragments)

    random.shuffle(flows)
    new_benign_fragments = 0
    new_ddos_fragments = 0

    for flow in flows:
        if flow[1]['label'] == 0 and (new_benign_fragments < min_fragments ):
            new_benign_fragments += len(flow[1]) - 1
            new_flow_list.append(flow)
        elif flow[1]['label'] > 0 and (new_ddos_fragments < min_fragments):
            new_ddos_fragments += len(flow[1]) - 1
            new_flow_list.append(flow)

    return new_flow_list, new_benign_fragments, new_ddos_fragments


'''
preprocessed_flows, _, _ = balance_dataset(preprocessed_flows)

(total_flows, ddos_flows, benign_flows),  (total_fragments, ddos_fragments, benign_fragments) = count_flows(preprocessed_flows)

log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                ") | fragments (tot,ben,ddos):(" + str(total_fragments) + "," + str(benign_fragments) + "," + str(ddos_fragments) + \
                ")  |\n"
print (log_string)
'''


# dataset to list of fragment
keys = []
X = []
y = []

for flow in preprocessed_flows:
    tuple = flow[0]
    flow_data = flow[1]
    label = flow_data['label']
    for key, fragment in flow_data.items():
        if key != 'label':
            X.append(fragment)
            y.append(label)
            keys.append(tuple)

#print(np.unique(y,return_counts=True))


# normalization and padding
mins,maxs = static_min_max(time_window=args.time_window)

mins,maxs = find_min_max(X,time_window=args.time_window)
print(mins,maxs)

print("Before normalization")
print(X[0])

# print(X[0].shape)

X = normalize_and_padding(X, mins, maxs, args.flow_length)
print("After normalization")
# print("Length of dataset: {}".format(len(X)))
print(X[0])


#packets = count_packets_in_dataset(norm_X_full)
#log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | total_flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
#                ") | Packets :(" + str(packets) + ") |\n"

#print(log_string)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


print("Train size: {}".format(np.unique(y_train, return_counts=True)))

print("Test size: {}".format(np.unique(y_test, return_counts=True)))



# save to file
data = {'xtrain': X_train, 'ytrain': y_train, 'xtest':X_test,'ytest':y_test}
hkl.dump(data,'data_'+str(args.flow_length)+'.hkl')
