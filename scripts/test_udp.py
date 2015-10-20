#!/usr/bin/env python

import socket
import struct
import time

def main(argv):
	#with open("out2.dat", 'w') as f:
	#	pass
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	#sock.connect(("127.0.0.1",8321))
	payload_size = 16
	for i in xrange(5):#8):#32):
		header  = struct.pack('>Q', i*payload_size)
		payload = chr(i%256)*payload_size
		#with open("out2.dat", 'a') as f:
		#	f.write(header+payload)
		sock.sendto(header+payload, ("127.0.0.1",8321))
		#time.sleep(10e-3)
		#time.sleep(1e-3)
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
