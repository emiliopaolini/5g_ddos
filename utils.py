import numpy as np

from collections import OrderedDict

SEED = 1
FILENAME = "Slowloris_BS1_nogtp.pcapng"
TIME_WINDOW = 10
MAX_FLOW_LEN = 10
protocols = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
powers_of_two = np.array([2**i for i in range(len(protocols))])

# feature list with min and max values
feature_list = OrderedDict([
    ('timestamp', [0,10]),
    ('packet_length',[0,1<<16]),
    ('highest_layer',[0,1<<32]),
    ('IP_flags',[0,1<<16]),
    ('protocols',[0,1<<len(protocols)]),
    ('TCP_length',[0,1<<16]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_flags',[0,1<<16]),
    ('TCP_window_size',[0,1<<16]),
    ('UDP_length',[0,1<<16]),
    ('ICMP_type',[0,1<<8])]
)

# min/max values of features based on the nominal min/max values of the single features (as defined in the feature_list dict)
def static_min_max(time_window=10):
    feature_list['timestamp'][1] = time_window

    min_array = np.zeros(len(feature_list))
    max_array = np.zeros(len(feature_list))

    i=0
    for feature, value in feature_list.items():
        min_array[i] = value[0]
        max_array[i] = value[1]
        i+=1

    return min_array,max_array

def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):
    rng = maxs - mins
    print("in scaling")
    print(high - (((high - low) * (maxs - rawpoints)) / rng))
    return high - (((high - low) * (maxs - rawpoints)) / rng)
    

def normalize_and_padding(X,mins,maxs,max_flow_len,padding=True):
    norm_X = []
    for sample in X:
        if sample.shape[0] > max_flow_len: # if the sample is bigger than expected, we cut the sample
            sample = sample[:max_flow_len,...]
        packet_nr = sample.shape[0] # number of packets in one sample

        print(sample)
        norm_sample = scale_linear_bycolumn(sample, mins, maxs, high=1.0, low=0.0)
        np.nan_to_num(norm_sample, copy=False)  # remove NaN from the array
        if padding == True:
            norm_sample = np.pad(norm_sample, ((0, max_flow_len - packet_nr), (0, 0)), 'constant',constant_values=(0, 0))  # padding
        norm_X.append(norm_sample)
        exit(1)
    return norm_X

def count_packets_in_dataset(X_list):
    packet_counters = []
    for X in X_list:
        TOT = X.sum(axis=1)
        packet_counters.append(np.count_nonzero(TOT))

    return packet_counters