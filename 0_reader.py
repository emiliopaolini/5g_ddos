import sys
import time
import pyshark
import socket
import pickle
import random
import hashlib
import argparse
import ipaddress
import os
import glob

from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Manager, Value, Queue
from utils import *
from collections import OrderedDict


vector_proto = CountVectorizer()
vector_proto.fit_transform(protocols).todense()

mapping = {"Goldeneye_BS1_nogtp.pcapng": ["10.155.15.3","10.41.150.68",1],
           "Slowloris_BS1_nogtp.pcapng": ["10.155.15.0","10.41.150.68",2],          
           "SYNflood_BS1_nogtp.pcapng": ["10.155.15.4","10.41.150.68",3],
           "SYNScan_BS1_nogtp.pcapng": ["10.155.15.1","10.41.150.68",4],
           #"ICMPflood_BS1_nogtp.pcapng": ["10.155.15.9","10.41.150.68",5],
           "TCPConnect_BS1_nogtp.pcapng": ["10.155.15.1","10.41.150.68",5],
           "Torshammer_BS1_nogtp.pcapng": ["10.155.15.4","10.41.150.68",6],   
           "UDPflood_BS1_nogtp.pcapng": ["10.155.15.4","10.41.150.68",7],
           "UDPScan_BS1_nogtp.pcapng": ["10.155.15.9","10.41.150.68",8]}

class packet_features:
    def __init__(self):
        self.id_fwd = (0,0,0,0,0) # 5-tuple: src ip, src port, dst ip, dst port, protocol
        self.id_bwd = (0,0,0,0,0) # 5-tuple: src ip, src port, dst ip, dst port, protocol
        self.features_list = []
    
    def __str__(self):
        return "{} -> {}".format(self.id_fwd,self.features_list)

def process_pcap(pcap_file, max_flow_len, time_window, ip_attacker, ip_victim, label, filename, max_flows=0):
    temp_dict = OrderedDict()

    start_time_window = -1
    
    cap = pyshark.FileCapture(pcap_file)
    labelled_flows = []
    for i, pkt in enumerate(cap):
        if i % 1000 == 0:
            print(pcap_file + " packet #", i)
        if start_time_window == -1 or float(pkt.sniff_timestamp) > start_time_window + time_window:
            start_time_window = float(pkt.sniff_timestamp)

        # parse packet
        try:
            pf = packet_features()
            tmp_id = [0,0,0,0,0]
            tmp_id[0] = str(pkt.ip.src)  # int(ipaddress.IPv4Address(pkt.ip.src))
            tmp_id[2] = str(pkt.ip.dst)  # int(ipaddress.IPv4Address(pkt.ip.dst))
            

            
            pf.features_list.append(float(pkt.sniff_timestamp)) # timestamp
            pf.features_list.append(int(pkt.ip.len)) # len packet
            pf.features_list.append(int(pkt.ip.ttl)) # len packet
            pf.features_list.append(int(hashlib.sha256(str(pkt.highest_layer).encode('utf-8')).hexdigest(),16) % 10 ** 8) # highest layer encoded as number 
            pf.features_list.append(int(int(pkt.ip.flags, 16))) # base 16, ip flags
            
            
            protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray().tolist()[0] # dense vector containing 1 when protocol is present
            
            protocols = [1 if i >= 1 else 0 for i in protocols]  # we do not want the protocols counted more than once (sometimes they are listed twice in pkt.frame_info.protocols)
            #print("Features list: {}".format(protocols))
            protocols_value = int(np.dot(np.array(protocols), powers_of_two)) # from binary vector to integer representation
            #print("Features list: {}".format(protocols_value))
            pf.features_list.append(protocols_value)

            protocol = int(pkt.ip.proto)
            tmp_id[4] = protocol
            if pkt.transport_layer != None:
                if protocol == socket.IPPROTO_TCP:
                    tmp_id[1] = int(pkt.tcp.srcport)
                    tmp_id[3] = int(pkt.tcp.dstport)
                    pf.features_list.append(int(pkt.tcp.len))  # TCP length
                    pf.features_list.append(int(pkt.tcp.ack))  # TCP ack
                    pf.features_list.append(int(pkt.tcp.flags, 16))  # TCP flags
                    pf.features_list.append(int(pkt.tcp.window_size_value))  # TCP window size
                    pf.features_list = pf.features_list + [0, 0]  # UDP + ICMP positions
                elif protocol == socket.IPPROTO_UDP:
                    pf.features_list = pf.features_list + [0, 0, 0, 0]  # TCP positions
                    tmp_id[1] = int(pkt.udp.srcport)
                    pf.features_list.append(int(pkt.udp.length))  # UDP length
                    tmp_id[3] = int(pkt.udp.dstport)
                    pf.features_list = pf.features_list + [0]  # ICMP position
            elif protocol == socket.IPPROTO_ICMP:
                pf.features_list = pf.features_list + [0, 0, 0, 0, 0]  # TCP and UDP positions
                pf.features_list.append(int(pkt.icmp.type))  # ICMP type
            else:
                pf.features_list = pf.features_list + [0, 0, 0, 0, 0, 0]  # padding for layer3-only packets
                tmp_id[4] = 0
            pf.id_fwd = (tmp_id[0], tmp_id[1], tmp_id[2], tmp_id[3], tmp_id[4])
            pf.id_bwd = (tmp_id[2], tmp_id[3], tmp_id[0], tmp_id[1], tmp_id[4])
        except AttributeError as e:
            print("Error in parsing packet")
            continue
        #print(pf)


        # store packet
        # temp_dict is structured as: temp_dict[pf.id_fwd] -> {start_time_window: features list, label: 0/1}
        
        if pf.id_fwd in temp_dict and start_time_window in temp_dict[pf.id_fwd] and \
                temp_dict[pf.id_fwd][start_time_window].shape[0] < max_flow_len:
            
            temp_dict[pf.id_fwd][start_time_window] = np.vstack([temp_dict[pf.id_fwd][start_time_window], pf.features_list])
        
        elif pf.id_bwd in temp_dict and start_time_window in temp_dict[pf.id_bwd] and \
                temp_dict[pf.id_bwd][start_time_window].shape[0] < max_flow_len:
            
            temp_dict[pf.id_bwd][start_time_window] = np.vstack(
                [temp_dict[pf.id_bwd][start_time_window], pf.features_list])
        else:
            if pf.id_fwd not in temp_dict and pf.id_bwd not in temp_dict:
                temp_dict[pf.id_fwd] = {start_time_window: np.array([pf.features_list]), 'label': 0}
            elif pf.id_fwd in temp_dict and start_time_window not in temp_dict[pf.id_fwd]:
                temp_dict[pf.id_fwd][start_time_window] = np.array([pf.features_list])
            elif pf.id_bwd in temp_dict and start_time_window not in temp_dict[pf.id_bwd]:
                temp_dict[pf.id_bwd][start_time_window] = np.array([pf.features_list])
        if max_flows > 0 and len(temp_dict) >= max_flows:
            break

    # apply label
    for five_tuple, flow in temp_dict.items():
    
        if five_tuple[0]==ip_attacker and five_tuple[2]==ip_victim:
            flow['label'] = label
        elif five_tuple[2]==ip_attacker and five_tuple[0]==ip_victim:
            flow['label'] = label
        else: 
            flow['label'] = 0
        labelled_flows.append((five_tuple, flow))


    count_good = 0
    count_bad = 0

    for i,j in labelled_flows:
  
        if j['label']==0:
            count_good += 1
            #print(i) 
            #print(j)
        else:
            count_bad += 1
    print("total good: "+str(count_good))
    print("total bad: "+str(count_bad))
    with open(filename+'.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(labelled_flows, filehandle)
    return labelled_flows


            
def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="5G DDoS")
    
    parser.add_argument("--flow_length", type=int, default=10, required=True)
    
    parser.add_argument("--time_window", type=int, default=10, required=True)
    
    # parser.add_argument("--ip_attacker", required=True)

    #  parser.add_argument("--ip_victim", required=True)

    
    args = parser.parse_args()
    flows = []
    
    manager = Manager()
    process_list = []
    flows_list = []
    
    filelist = glob.glob('pcap_files/*.pcapng')
    
    for file in filelist:
        
            try:
                filename = file.split("/")[1]
                if filename=='ICMPflood_BS1_nogtp.pcapng':
                    continue
                print(filename)
                print("Processing file: ",filename)
                print("Ip attacker:", mapping[filename][0])
                print("Ip victim:", mapping[filename][1])
                print("Label: ",mapping[filename][2])
                flows = manager.list()
                
                p = Process(target=process_pcap,args=(file, args.flow_length, args.time_window, mapping[filename][0], mapping[filename][1], mapping[filename][2], filename))
                process_list.append(p)
                flows_list.append(flows)
            except FileNotFoundError as e:
                continue
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    
if __name__=="__main__":
    main()