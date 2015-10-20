#!/usr/bin/env python

import zmq
import json

def recv_broadcast(sock):
	msg = sock.recv()
	msg = msg.split(None, 1)[1] # Crop off topic name
	return json.loads(msg)

def get_rates(stats, new_stats):
	try:
		p = stats['__period__']
	except KeyError:
		p = 1000000000
	dt = (new_stats['__time__'] - stats['__time__']) / float(p)
	return {k: (new_stats[k]-v)/dt for k,v in stats.items()
	        if not (k.startswith('__') and k.endswith('__'))}

def main(argv):
	ctx  = zmq.Context()
	sock = zmq.Socket(ctx, zmq.SUB)
	addr = "tcp://"+argv[1] if len(argv) >= 1 else "tcp://ledaovro1:7777"
	sock.connect(addr)
	#sock.setsockopt(zmq.SUBSCRIBE, 'LWAOV-Pipeline.Depacketize.stats')
	sock.setsockopt(zmq.SUBSCRIBE, 'ADP-Pipeline.Depacketize.stats')
	print ("%7s %7s %7s %7s %7s %7s "
	       "%8s %8s %8s" %
	       ("RECV", "GOOD", "MISSING", "NRECV", "NIGNORE", "NLATE",
	        "OVRWRTN", "PENDING", "MISSING"))
	print ("%7s %7s %7s %7s %7s %7s "
	       "%8s %8s %8s" %
	       ("Gb/s", "Gb/s", "Bytes/s", "pkt/s", "pkt/s",   "pkt/s",
	        "Bytes",   "Bytes",   "Bytes"))
	stats = recv_broadcast(sock)
	while True:
		new_stats = recv_broadcast(sock)
		rates = get_rates(stats, new_stats)
		print ("%7.3f %7.3f %8.2e %7.0f %7.0f %7.0f "
		       "%8.2e %8.2e %8.2e" %
		       (rates["nrecv_bytes"]/1e9*8,
		        rates["ngood_bytes"]/1e9*8,
		        rates["nmissing_bytes"],
		        rates["nrecv"],
		        rates["nignored"],
		        rates["nlate"],
				float(new_stats["noverwritten_bytes"]),
		        float(new_stats["npending_bytes"]),
		        float(new_stats["nmissing_bytes"])))
		stats = new_stats
		new_stats = None
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
