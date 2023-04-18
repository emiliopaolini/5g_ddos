import numpy as np

SEED = 1
FILENAME = "SYNflood_BS1_nogtp.pcapng"
TIME_WINDOW = 10
protocols = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
powers_of_two = np.array([2**i for i in range(len(protocols))])