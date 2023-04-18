import sys
import time
import pyshark
import socket
import pickle
import random
import hashlib
import argparse
import ipaddress

from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Manager, Value, Queue
from utils import *
from collections import OrderedDict


vector_proto = CountVectorizer()
vector_proto.fit_transform(protocols).todense()

class packet_features:
    def __init__(self):
        self.id_fwd = (0,0,0,0,0) # 5-tuple: src ip, src port, dst ip, dst port, protocol
        self.id_bwd = (0,0,0,0,0) # 5-tuple: src ip, src port, dst ip, dst port, protocol
        self.features_list = []
    
    def __str__(self):
        return "{} -> {}".format(self.id_fwd,self.features_list)

def process_pcap(pcap_file, in_labels, max_flow_len, labelled_flows, max_flows=0, traffic_type='all', time_window=TIME_WINDOW):
    start_time = time.time()
    temp_dict = OrderedDict()

    start_time_window = -1
    
    cap = pyshark.FileCapture(pcap_file)

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
            

            pf.features_list.append(pkt.sniff_timestamp) # timestamp
            pf.features_list.append(pkt.ip.len) # len packet
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
        print(pf)
        # store packet

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
        if i == 2:
            print(temp_dict)
            exit(1)

process_pcap(FILENAME, None, None, None)