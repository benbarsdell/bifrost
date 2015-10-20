


// TODO: Change this to accept src_ip, dst_ip and len instead of headers
unsigned short udp_checksum(iphdr  ip_header,
                            udphdr udp_header,
                            const uint8_t* data,
                            int data_len) {
	udp_pseudo_hdr header;// = {0};
	/*
	// HACK TESTING
	header.saddr = ((152<<24)+( 1<<16)+(51<<8)+27);
	header.daddr = ((152<<24)+(14<<16)+(94<<8)+75);
	header.zero  = 0;
	header.protocol = 0x11;
	header.len   = htons(10);
	printf("saddr = %x\n", ntohl(header.saddr));
	printf("daddr = %x\n", ntohl(header.daddr));
	printf("proto = %x\n", header.protocol);
	//printf("sum16bit = %u\n", ntohl(sum16bit((uint8_t*)&header, sizeof(header))));
	//printf("sum16bit = %x\n", ntohl(sum16bit((uint8_t*)&header, sizeof(header))));
	printf("sum16bit = %x\n", ntohl(sum16bit((uint8_t*)&header, 8)));
	*/
	header.saddr    = ip_header.saddr;
	header.daddr    = ip_header.daddr;
	//header.zero     = 0;
	header.protocol = htons(ip_header.protocol);
	printf("PROTOCOL: %x\n", header.protocol);
	//header.len      = htons((unsigned short)udp_header.len);
	//header.len      = htons(udp_header.len);
	header.len      = (udp_header.len);
	printf("LEN: %hu\n", ntohs(header.len));
	printf("CRC: %x\n", ntohs(udp_header.check));
	
	printf("saddr = %x\n", ntohl(header.saddr));
	printf("daddr = %x\n", ntohl(header.daddr));
	printf("proto = %hx\n", header.protocol);
	
	unsigned int sum = 0;
	////sum += sum16bit((uint8_t*)&header, sizeof(header));
	sum += sum16bit(&header);
	printf("DATA_LEN: %i\n", data_len);
	//*sum += sum16bit(data, data_len); // HACK TESTING disabled
	printf("sum = %x\n:", sum);
	while( sum >> 16 ) {
		sum = (sum & 0xffff) + (sum >> 16);
	}
	printf("sum folded = %x\n:", sum);
	//*unsigned short checksum = (unsigned short)~sum;
	unsigned short checksum = (unsigned short)sum;
	return checksum;
}

//// TODO: Check this when passed the address of a stack variable; printf's failing under valgrind
// TODO: This doesn't work on stack-allocated pseudo-header?
unsigned int sum16bit(const void* data,
                      int         len) {
	const uint16_t* buf = (uint16_t*)data;
	unsigned int sum = 0;
	//printf("sum16bit len = %i\n", len);
	while( len > 1 ) {
		//for( int i=0; i<len/(int)sizeof(unsigned short); ++i ) {
		//sum += ((unsigned short*)data)[i];
		sum += *buf++;
		len -= 2;
	}
	if( len & 1 ) {
		sum += *((uint8_t*)buf);
	}
	/*
	typedef ushort2 word;
	int nword = (len-1) / (int)sizeof(word) + 1;
	for( int i=0; i<nword; ++i ) {
		word w = ((word*)data)[i];
		sum += (unsigned int)w.x + w.y;
	}
	*/
	/*
	printf("**** %hx\n", ((unsigned short*)data)[0]);
	printf("**** %hx\n", ((unsigned short*)data)[1]);
	printf("**** %hx\n", ((unsigned short*)data)[2]);
	printf("**** %hx\n", ((unsigned short*)data)[3]);
	printf("**** %hx\n", ((unsigned short*)data)[4]);
	*/
	//printf("**** %hx\n", ((unsigned short*)data)[5]);
	return sum;
	//printf("len = %i\n", len);
	//unsigned int sum = 0;
	/*
	for( int i=0; i<4; ++i ) {
		unsigned short h = ((unsigned short*)data)[i];
		printf("**** %hx\n", h);
	}
	*/
	/*
	unsigned int sum = 0;
	for( int i=0; i<len/(int)sizeof(unsigned short); ++i ) {
		sum += ((unsigned short*)data)[i];
		//printf("%hx ", ((unsigned short*)data)[i]);
		//printf("**** %hx\n", ((unsigned short*)data)[0]);
		//printf("%i\n", i);
		//printf("**** %hx\n", ((unsigned short*)data)[i]);
	}
	//printf("**** %hx\n", ((unsigned short*)data)[2]);
	//printf("\n");
	return sum;
	*/
}

unsigned int sum16bit(const udp_pseudo_hdr* hdr) {
	unsigned int sum = 0;
	sum += ((unsigned short*)hdr)[0];
	sum += ((unsigned short*)hdr)[1];
	sum += ((unsigned short*)hdr)[2];
	sum += ((unsigned short*)hdr)[3];
	sum += ((unsigned short*)hdr)[4];
	sum += ((unsigned short*)hdr)[5];
	return sum;
}


struct udp_pseudo_hdr {
	uint32_t saddr;
	uint32_t daddr;
	//uint8_t  zero;
	//uint8_t  protocol;
	uint16_t protocol;
	uint16_t len;
};
/*
struct __attribute__((aligned(32))) ushort2 {
	unsigned short x, y;
};
*/

/*
struct udp_header {
	unsigned short src_port;
	unsigned short dst_port;
	unsigned short length;
	unsigned short checksum;
};
*/


			offset += (check_for_ip_header_options ?
			           ip_header->ihl*4 :
			           sizeof(iphdr));
			const udphdr*  udp_header = (const udphdr*)ptr;
			offset += sizeof(udphdr);
			
			
			
			offset += sizeof(ethhdr);
			if( check_for_ip_header_options ) {
				iphdr* ip_header = (iphdr*)&block[offset];
				offset += ip_header->ihl * 4;
			}
			else {
				offset += sizeof(iphdr);
			}
			offset += sizeof(udphdr);
			// Update mapping of (seq,idx) pairs to memory pointers
			uint64_t seq, idx;
			//*offset += decode_seq_idx(&block[offset], &seq, &idx);

//unsigned int offset    = frame_offsets[p];


		enum { MASK = (~0ULL) >> (sizeof(0ULL)*8 - NIDX) };
		for( uint64_t seq=seq_beg; seq!=seq_end; ++seq ) {
			uint64_t seq_set = _packet_set[seq % NSEQ_BUF];
			if( (~seq_set & MASK) ) {
				return false;
			}
		}

//self->_new_seqs.push_back(seq);
				//self->_new_idxs.push_back(idx);
				//self->_new_ptrs.push_back();

//bool seq_complete = self->_packet_set[seq % NSEQ_BUF].flip().none();
			//if( seq_complete ) {
			//}

////process_sequences(cur_seq, cur_seq+nseq_proc);
				////self->zero_sequences(cur_seq, cur_seq + nseq_proc); // TODO: Do this in process
				////ndropped += self->count_sequence_packets(cur_seq, cur_seq + nseq_proc); //   and this.
				//cur_seq += nseq_proc;

		//while( all_sequences_complete(cur_seq, cur_seq + nseq_proc) ) {


		// for each sorted packet:
		//   if seq >= self->_cur_seq + NSEQ_BUF:
		//     close and process cur_seq:cur_seq+
			
			else if( seq >= self->_cur_seq + NSEQ_BUF ) {
				
			}
			else {
				self->_packet_map[seq % NSEQ_BUF][idx] = d_block + (ptr - block);
				self->_packet_set[seq % NSEQ_BUF].set(idx);
				max_seq = std::max(max_seq, seq);
			}
		}
		self->process(max_seq);
		
		while( !all_sequences_complete(cur_seq, cur_seq + nseq_proc) ||
		       max_seq - cur_seq > NSEQ_BUF ) {
			ndropped += self->count_sequence_packets(cur_seq, cur_seq + nseq_proc);
			//process_sequences(cur_seq, cur_seq+nseq_proc);
			self->zero_sequences(cur_seq, cur_seq + nseq_proc);
			cur_seq += nseq_proc;
		}
		
		// TODO: Keep track of: ndropped = nlate (arrived after seq processed) + nmissing (never arrived)
		
		/*
		// Process sequences of contiguous data as they are completed or grow too old
		while( all(packet_set[cur_seq:cur_seq+nseq_proc]) ||
		       max_seq - cur_seq > nseq_buf) {
			size_t ndropped = nseq_proc*NIDX - count(packet_set[cur_seq:cur_seq+nseq_proc]);
			self->_ndroped_tot += ndropped;
			// Execute packet-gather kernel on GPU
			process(cur_seq, cur_seq+nseq_proc);
			packet_set[cur_seq:cur_seq+nseq_proc] = 0;
			cur_seq += nseq_proc;
		}
		*/

	//uint64_t _packet_set[NSEQ_BUF];// = {0};

//max_seq = std::max(max_seq, seq);

//uint64_t max_seq = 0;

// Loop through packets still in this buffer that will be overwritten
		for( PacketIdSet::const_iterator it=self->_block_pkts[cur_block % NBLOCK_BUF].begin();
		     it!=self->_block_pkts[cur_block % NBLOCK_BUF].end();
		     ++it ) {
			uint64_t seq = it->first;
			uint64_t idx = it->second;
			// Unmark packet
			self->_packet_set[seq].reset(idx);
			++noverwritten;
		}

// TODO: Take care of existing remaining data in this block
		//         For each (seq,idx) still in the block, remove (seq,idx) from _packet_set
		//           and increment noverwritten.


	bool all_sequences_complete(uint64_t seq_beg, uint64_t seq_end) {
		for( uint64_t seq=seq_beg; seq!=seq_end; ++seq ) {
			if( _packet_set[seq % NSEQ_BUF].flip().any() ) {
				return false;
			}
		}
		return true;
	}
	int count_sequence_packets(uint64_t seq_beg, uint64_t seq_end) {
		int count = 0;
		for( uint64_t seq=seq_beg; seq!=seq_end; ++seq ) {
			count += _packet_set[seq % NSEQ_BUF].count();
		}
		return count;
	}
	void zero_sequences(uint64_t seq_beg, uint64_t seq_end) {
		for( uint64_t seq=seq_beg; seq!=seq_end; ++seq ) {
			_packet_set[seq % NSEQ_BUF].reset();
		}
	}

//const uint8_t*    _packet_ptrs[NSEQ_BUF][NIDX];// = {0}; // GPU mem location of each packet
	//std::bitset<NIDX> _packet_set[NSEQ_BUF]; // Bitset of received packet idx's for each sequence number

//uint8_t*   _d_block_bufs[NBLOCK_BUF]; // Buffers for blocks of frames
	//PacketList _block_pkts[NBLOCK_BUF];   // Which packets are in each block of frames

//printf("Cur seq:                 %lu\n", self->_cur_seq);
		//printf("No. late packets:        %i\n", nlate);
		//printf("No. overwritten packets: %i\n", noverwritten);

void test_udp_checksum() {
	udp_pseudo_hdr pheader;// = {0};
	pheader.saddr = htonl((152<<24)+( 1<<16)+(51<<8)+27);
	//pheader.saddr = htonl((127<<24)+( 0<<16)+( 0<<8)+ 1);
	pheader.daddr = htonl((152<<24)+(14<<16)+(94<<8)+75);
	//pheader.zero  = 0;
	pheader.protocol = htons(0x11);
	pheader.len   = htons(10);
	//printf("saddr = %x\n", ntohl(pheader.saddr));
	//printf("daddr = %x\n", ntohl(pheader.daddr));
	//printf("proto = %hhx\n", pheader.protocol);
	//printf("sum16bit = %u\n", ntohl(sum16bit((uint8_t*)&header, sizeof(header))));
	//printf("sum16bit = %x\n", ntohl(sum16bit((uint8_t*)&header, sizeof(header))));
	//printf("sum16bit = %x\n", ntohl(sum16bit((uint8_t*)&header, 8)));
	//printf("**** %x\n", ((int*)&pheader)[0]);
	//uint8_t* ppheader = (uint8_t*)malloc(sizeof(pheader));
	//memcpy((void*)ppheader, (void*)&pheader, sizeof(pheader));
	/*
	printf("sum16bit = %x\n", sum16bit((uint8_t*)&pheader, sizeof(pheader)));
	//printf("sum16bit = %x\n", (sum16bit((uint8_t*)ppheader, 8)));
	printf("sum16bit = %x\n", sum16bit(&pheader));
	unsigned int sum = sum16bit(&pheader);
	while( sum >> 16 ) {
		sum = (sum & 0xffff) + (sum >> 16);
	}
	printf("sum folded = %x\n", sum);
	unsigned short checksum = ~(unsigned short)sum;
	//printf("checksum  = %hx\n", checksum);
	printf("checksum  = %hx\n", ntohs(checksum));
	*/
	char pkt[10] = {0};
	uint16_t checksum2 = udp_checksum(pkt, ntohs(pheader.len),
	                                  pheader.saddr, pheader.daddr);
	printf("checksum2 = %hx\n", ntohs(checksum2));
}

/*
	virtual void init(const char* addr,
	                  uint16_t    port) {
		this->del();
		check_error(_fd = socket(AF_INET, SOCK_DGRAM, 0),
		            "create socket");
		_msg_size_max = DEFAULT_PACKET_SIZE_MAX;
		_block_size   = DEFAULT_BLOCK_SIZE;
		this->allocate();
	}
	virtual void del() {
		if( _fd >= 0 ) {
			close(_fd);
			_fd = -1;
			_msg_size_max = 0;
			_block_size   = 0;
		}
	}
	*/

/*
	virtual void init(const char* addr,
	                  uint16_t    port) = 0;
	virtual void del() = 0;
	*/


class Capturer2 {
	typedef uint64_t seq_type;
	typedef std::map<seq_type, uint8_t const*> PacketMap;
	PacketMap _packet_map;
	seq_type  _current_seq;
	seq_type sequence_population(seq_type nseq) const {
		seq_type seq_beg = _packet_map.begin()->first;
		seq_type seq_end = seq_beg + nseq;
		return _packet_map.lower_bound(seq_end) - _packet_map.begin();
	}
public:
	Capturer2() : _current_seq(0) {}
	void   set_timeout(float secs) { _timeout_secs = secs; }
	size_t read(const void* ptr, size_t size) {
		float deadline = ;
		while( time() < deadline ) {
			_rx->recv_block(..., time() - deadline);
			process_block(...);
			// ** TODO: decode_packet should return byte offset, not sequence number
		}
		
		
	}
	long   tell() const { return _current_seq; }
	
	static void block_callback(const uint8_t*  block,
	                           uint64_t        size,
	                           uint64_t        npacket,
	                           const uint64_t* packet_offsets,
	                           const uint64_t* packet_sizes,
	                           void*           userdata) {
		Capturer* self = (Capturer*)userdata;
		
		for( uint64_t p=0; p<npacket; ++p ) {
			const uint8_t* ptr  = block + packet_offsets[p];
			size_t         size = packet_sizes[p];
			seq_type seq;
			ptr = _decode_packet(ptr, size, &seq, _decode_packet_userdata);
			if( ptr ) {
				_packet_map[seq] = std::make_pair(ptr, size);
			}
		}
		while( self->sequence_population(nseq) == nseq ||
		       _packet_map.size() > _npacket_buf ) {
			size_t ngood = self->sequence_population(nseq);
			size_t nmiss = nseq - ngood;
			
		}
	}
};

seq_type sequence_population(seq_type nseq) const {
		seq_type seq_beg = _packet_map.begin()->first;
		seq_type seq_end = seq_beg + nseq;
		return _packet_map.lower_bound(seq_end) - _packet_map.begin();
	}


			//uint8_t const* src_ptr  = pkt_ptr + (std::max(pkt_byte, _current_byte) - pkt_byte);
			uint8_t const* dst_ptr  = ptr + (pkt_byte - _current_byte);
			
			size_t head_cut = std::max(pkt_byte,            _current_byte) - pkt_byte;
			size_t tail_cut = std::min(pkt_byte + pkt_size, _current_byte + pkt_size);
			
			size_t         cpy_size = std::min(pkt_size, ptr + size - dst_ptr);
			dst_ptr =     ptr + std::max(pkt_byte - _current_byte, 0);
			src_ptr = pkt_ptr + std::min(_current_byte - pkt_byte, pkt_size);
			// dst_ptr <--> _current_byte
			// dst_ptr+size <--> _current_byte+size
			if( pkt_byte < _current_byte ) {
				src_ptr  += _current_byte - pkt_byte;
				cpy_size -= _current_byte - pkt_byte;
			}
			if( pkt_byte + pkt_size > _current_byte + size ) {
				cpy_size -= (pkt_byte + pkt_size) - (_current_byte + size);
			}
			::memcpy(dst_ptr, src_ptr, cpy_size);

//_packet_map.insert(std::make_pair(pkt_byte,
							//                                  std::make_pair(pkt_ptr,
							//                                                 pkt_size)));


	static void block_callback(const uint8_t*  block,
	                           uint64_t        size,
	                           uint64_t        npacket,
	                           const uint64_t* packet_offsets,
	                           const uint64_t* packet_sizes,
	                           void*           userdata) {
		Capturer* self = (Capturer*)userdata;
		
		for( uint64_t p=0; p<npacket; ++p ) {
			const uint8_t* ptr0 = block + packet_offsets[p];
			size_t         size = packet_sizes[p];
			seq_type seq;
			const uint8_t* ptr = _decode_packet(ptr0, size, &seq, _decode_packet_userdata);
			size -= ptr - ptr0;
			if( ptr ) {
				_packet_map[seq] = std::make_pair(ptr, size);
			}
		}
		while( self->sequence_population(nseq) == nseq ||
		       _packet_map.size() > _npacket_buf ) {
			size_t ngood = self->sequence_population(nseq);
			size_t nmiss = nseq - ngood;
			
		}
	}

//_tail = (_tail == 0 ? _bufs.size() : _tail) - 1;


	// ** TODO: Seriously consider not using a callback (just return data by ref)
	// Note: timeout_secs  < 0 => wait for complete block
	//       timeout_secs == 0 => return immediately
	virtual void get_next_block(rx_callback_type process_block,
	                            void*            userdata=NULL,
	                            uint64_t*        npacket=NULL,
	                            float            timeout_secs=-1) {
		/*
		timespec timeout;
		timeout.tv_sec  = (int)timeout_secs;
		timeout.tv_nsec = (int)((timeout_secs - timeout.tv_sec)*1e9);
		printf("%li\t%li\n", timeout.tv_sec, timeout.tv_nsec);
		const timespec* timeout_ptr = (timeout_secs < 0) ? 0 : &timeout;
		*/
		// WAR for BUG in recvmmsg timeout behaviour
		if( timeout_secs > 0 ) {
			timeval timeout;
			timeout.tv_sec  = (int)timeout_secs;
			timeout.tv_usec = (int)((timeout_secs - timeout.tv_sec)*1e6);
			setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
		}
		int flags = (timeout_secs == 0) ? MSG_DONTWAIT : 0;
		ssize_t nmsg = recvmmsg(_fd, &_msgs[0], _msgs.size(), flags, 0);//timeout_ptr);
		if( nmsg < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ) ) {
			nmsg = 0;
		}
		else {
			check_error(nmsg, "receive messages");
		}
		if( nmsg > 0 ) {
			_frame_starts.resize(nmsg);
			_frame_sizes.resize(nmsg);
			for( int m=0; m<nmsg; ++m ) {
				_frame_starts[m] = (uint8_t*)_msgs[m].msg_hdr.msg_iov->iov_base - &_msgbuf[0];
				_frame_sizes[m]  = _msgs[m].msg_len;
			}
			if( process_block ) {
				process_block((uint8_t*)&_msgbuf[0], _msgbuf.size(),
				              nmsg, &_frame_starts[0], &_frame_sizes[0],
				              userdata);
			}
		}
		if( npacket ) {
			*npacket = nmsg;
		}
	}

virtual void get_memory(void**    ptr,
	                        uint64_t* size) const {
		*ptr  = (void*)&_msgbuf[0];
		*size = _msgbuf.size();
	}


			}
		}
		if( _packet_map.empty() ) {
			return;
		}
		size_t pkt_byte = _packet_map.begin()->first;
		size_t pkt_size = _packet_map.begin()->second.second;
		while( pkt_byte + pkt_size <= _current_byte ) {
			_packet_map.erase(_packet_map.begin());
			if( _packet_map.empty() ) {
				return;
			}
			pkt_byte = _packet_map.begin()->first;
			pkt_size = _packet_map.begin()->second.second;
		}

// ** TODO: Add ring buffer to store blocks until they're read
//            Could try using a queue to begin with?
// TODO: decode_packet should return byte offset, not sequence number

//_rx->recv_block(&block, &block_size, &npacket,
			//                &packet_offsets, &packet_sizes);
			// ** TODO: This will be necessary for the GPU, but on the CPU it is undesirable
			//            ** See just above^^
			//::memcpy(&_blocks[??], block, block_size);

//const uint8_t*  block;
			//uint64_t        block_size;
//const uint64_t* packet_offsets;
			//const uint64_t* packet_sizes;
			// Receive block of packets and copy to block buffer
			// *** TODO: This should instead accept a block buffer into which the packets
			//             are directly received.
			//             The stuff inside Rx is just trivial boilerplate, which can
			//               be done on the fly instead.

				//const uint8_t* pkt_ptr0 = block + packet_offsets[pkt];

std::vector<uint8_t>  _msgbuf;
	std::vector<uint64_t> _frame_starts;
	std::vector<uint64_t> _frame_sizes;

// Note: Slight hack to workaround no operator< for BlockId
	//typedef std::map<BlockType*, std::set<size_t> > BlockMap;

// TODO: Should allow some form of reset() that calls _packet_map.clear();

//std::atomic<ssize_t>   _tail;
	//std::atomic<bool>      _guarantees_exist;

/*
			// Copy the frames in the ghost region to the front of the buffer
			_buf.get_allocator().copy(&_buf[0],           // dst = start of buffer
			                          &_buf[_buf.size()], // src = start of ghost region
			                          buf_frame_end * _frame_size);
			*/

/*
			// Copy the frames at the front of the buffer to the ghost region
			_buf.get_allocator().copy(&_buf[_buf.size()], // src = start of ghost region
			                          &_buf[0],           // dst = start of buffer
			                          buf_frame_end * _frame_size);
			*/


class ClientSocket : public SocketBase {
public:
	
	
};

class ServerSocket : public SocketBase {
public:
	
};

/*
	void set_timeout(double timeout_secs) {
		if( timeout_secs > 0 ) {
			timeval timeout;
			timeout.tv_sec  = (int)timeout_secs;
			timeout.tv_usec = (int)((timeout_secs - timeout.tv_sec)*1e6);
			this->set_option(SO_RCVTIMEO, timeout);
			this->set_option(SO_SNDTIMEO, timeout);
		}
		else if( timeout_secs == 0 ) {
			// TODO: Could set O_NONBLOCK (via fctl) here, but it would cause connect()
			//         to return immediately too (requiring poll()'ing for completion).
		}
	}
	*/

/*
	  if( npacket ) {
	  *npacket = nmsg;
	  }
	*/

/*&npacket, */

//size_t npacket;
//return npacket;

template<typename SockAddrType> struct sockaddr_traits {};
	template<> struct sockaddr_traits<sockaddr_in>  { enum { FAMILY = AF_INET }; };
	template<> struct sockaddr_traits<sockaddr_in6> { enum { FAMILY = AF_INET6 }; };
	template<> struct sockaddr_traits<sockaddr_un>  { enum { FAMILY = AF_UNIX }; };

//this->open(Socket::sockaddr_traits<sockaddr_storage>::FAMILY, protocol);


	struct sockaddr_in* h;
	const char* ip = NULL;
	for( p = servinfo; p!=NULL; p=p->ai_next ) {
		h = (struct sockaddr_in*)p->ai_addr;
		ip = ::inet_ntoa(h->sin_addr);
		break; // Return first address
	}

static const char* ip_from_hostname(const char* hostname);

static sockaddr_storage addr_from_hostname(const char* hostname,
	                                           int         family);


static sockaddr_in Socket::ip4(const char*    addr,
                               unsigned short port) {
	sockaddr_storage sas;
	sockaddr_in* sa = static_cast<sockaddr_in*>(&sas);
	memset(sa, 0, sizeof(*sa));
	//sa->sin_family = AF_INET;
	//sa->sin_port   = htons(port);
	if( ::inet_pton(AF_INET, addr, &(sa->sin_addr)) != 1 ) {
		//throw Socket::Error("Invalid address");
		Socket::addr_from_hostname(addr, (sockaddr*)sa
	}
	return sas;
	
	
	if( !Socket::is_valid_ip(addr, AF_INET) ) {
		addr = ip_from_hostname(addr);
	}
}

static bool Socket::is_valid_ip(const char* ipstr, int family) {
	switch( family ) {
	case AF_INET: {
		struct in_addr addr;
		return ::inet_pton(family, ipstr, &addr) == 1;
	}
	case AF_INET6: {
		struct in6_addr addr;
		return ::inet_pton(family, ipstr, &addr) == 1;
	}
	default: throw Socket::Error("Invalid IP address family");
	}
}
// Similar to pton(), copies first found address into *address and returns 1
//   on success, else 0.
static int Socket::addr_from_hostname(const char* hostname,
                                      sockaddr*   address,
                                      int         family,
                                      int         socktype) {
	struct addrinfo hints;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family   = family;
    hints.ai_socktype = socktype;
    hints.ai_flags    = 0; // Any
    hints.ai_protocol = 0; // Any
    struct addrinfo* servinfo;
	if( ::getaddrinfo(hostname, 0, &hints, &servinfo) != 0 ) {
		return 0;
	}
	for( struct addrinfo* it=servinfo; it!=NULL; it=it->ai_next ) {
		::memcpy(address, it->ai_addr, it->ai_addrlen);
		break; // Return first address
	}
	::freeaddrinfo(servinfo);
	return 1;
}

static bool             is_valid_ip(const char* ip, int family);

switch( protocol ) {
	case SOCK_DGRAM: break;
	case SOCK_STREAM: {
		check_error(::listen(_fd, max_conn_queue),
		            "set socket to listen");
		// TODO: Work out where/how to do this
		//struct sockaddr_in remote_addr;
		//check_error(::accept4(_fd, &remote_addr, sizeof(remote_addr), SOCK_NONBLOCK),
		//            "accept incoming connection");
		break;
	}
	default: throw Socket::Error("Unsupported/invalid protocol");
	}

static std::string get_error_string(int errno) {
		switch( errno ) {
		case EACCES:     return "EACCES: Permission denied";
		case EADDRINUSE: return "EADDRINUSE: Address already in use";
		case EADDRNOTAVAIL: return "EADDRNOTAVAIL: Address not available";
		case EAFNOSUPPORT:  return "EAFNOSUPPORT: Address family not supported";
		case EAGAIN:        return "EAGAIN: Resource temporarily unavailable";
		case EALREADY:      return "EALREADY: Connection already in progress";
		case EBADF:         return "EBADF: Bad file descriptor";
		case EBADFD:        return "EBADFD: File descriptor in bad state";
		case EBUSY:         return "EBUSY: Device or resource busy";
		case ECOMM:         return "ECOMM: Communication error on send";
		case ECONNABORTED:  return "ECONNABORTED: Connection aborted";
			
		}
	}

struct ReadSpan {
		frame_idx_type frame0;
		size_t         nframe;
		bool           guaranteed;
		const_pointer  data;
	};


	// Writer interface
	// ----------------
	typedef std::pair<pointer,size_t> WriteSpan;
	// Note: Currently only a single writer is supported
	WriteSpan reserve(size_t nframe,
	                  double timeout_secs=-1) {
		unique_lock_type lock(_mutex);
		return _open_write(nframe, timeout_secs, lock);
	}
	WriteSpan commit(size_t nframe,
	                 size_t nframe_rereserve=0,
	                 double timeout_secs=-1) {
		unique_lock_type lock(_mutex);
		_close_write(nframe);
		if( nframe_rereserve ) {
			return _open_write(nframe_rereserve, timeout_secs, lock);
		}
		else {
			return WriteSpan(0, 0);
		}
	}
	// ----------------
	
	// Reader interface
	// ----------------


  TODO: Consider extending write support to allow push-pop style
          opening and closing of whole blocks (i.e., under assumption of ncommit=nreserve).
          Use-case 1: Packet receiver could open multiple small blocks instead of one large window,
                        which would avoid needing a large ghost region.
          Use-case 2: Multiple writers opening and filling whole blocks

//void  advance_write(WriteBlock* w, size_t nframe_reopen,
	//                    double timeout_secs=-1);

/*
	RingBuffer*    _ring;
	frame_idx_type _frame0;
	size_t         _nframe;
	bool           _guaranteed;
	const_pointer  _data;
	*/

/*
	frame_idx_type frame0()     const { return _frame0; }
	size_t         nframe()     const { return _nframe; }
	bool           guaranteed() const { return _guaranteed; }
	size_t         frame_size() const { assert(_ring); return _ring->frame_size(); }
	size_t         size()       const { return _nframe*this->frame_size(); }
	frame_idx_type head()       const { assert(_ring); return _ring->head(); }
	frame_idx_type tail()       const { assert(_ring); return _ring->tail(); }
	const_pointer  data()       const { return _data; }
	const_reference operator[](size_t n) const {
		return _data[n];
	}
	*/

inline void advance(ssize_t nframe_advance, size_t nframe_reopen=0) {
		_ring->advance_read(this, nframe_advance, nframe_reopen);
	}


// TODO: Implement some form of Subscriber class and give it friend access to
//         reader methods of RingBuffer, which are otherwise private.
/*
template<typename T>
class RingReader {
public:
	typedef RingBuffer<T> ring_type;
	typedef typename ring_type::frame_idx_type frame_idx_type;
	RingReader() {}
	
	const ReadSpan& open(size_t nframe) {
		assert( !_is_open );
		_is_open = true;
		_span = _ring->open_read(_frame0, nframe);
		return _span;
	}
	const ReadSpan& advance(size_t nframe) {
		assert( _is_open );
		_span = _ring->advance(_span, nframe);
		return _span;
	}
	void close() {
		assert( _is_open );
		_is_open = false;
		return _ring->close_read(_span);
	}
	void seek(long offset, int origin) {
		switch( origin ) {
		case SEEK_SET: _frame0  = offset; break;
		case SEEK_CUR: _frame0 += offset; break;
		default: throw std::runtime_error("Invalid origin for seek");
		}
	}
	bool still_valid() const { return _ring->still_valid(_frame0); }
private:
	ring_type*     _ring;
	frame_idx_type _frame0;
	bool           _is_open;
	ReadSpan       _span;
};
*/

		operator const pthread_mutexattr_t&() const { return _attr; }
		operator       pthread_mutexattr_t&()       { return _attr; }

Mutex(Attrs attrs=Attrs()) {
		check_error( pthread_mutex_init(&_mutex, &(pthread_mutexattr_t&)attrs) );
	}

ret = pthread_cond_timedwait(&_cond, &(pthread_mutex_t&)mutex,
			                                 &abstime);
				if( ret == ETIMEDOUT ) {
					return pred();
				}


template<typename Mutex>
class LockGuard {
	Mutex& _mutex
	bool   _locked;
public:
	explicit LockGuard(Mutex& m, bool locked=true)
		: _mutex(m), _locked(locked) { if( locked ) { _mutex.lock(); } }
	~LockGuard() { this->unlock(); }
	void unlock()     {
		if( !_locked ) {
			return;
		}
		_locked = false;
		_mutex.unlock();
	}
	void lock() {
		if( _locked ) {
			return; 
		}
		_mutex.lock();
		_locked = true;
	}
	operator       Mutex&()       { return _mutex; }
	operator const Mutex&() const { return _mutex; }
private: // Prevent copying or assignment
	void operator=(const LockGuard&);
	LockGuard(const LockGuard&);
};

/*
// An exception-safe scoped lock-keeper
template<typename Mutex>
class unique_lock {
	std::auto_ptr<Mutex> _mutex;
	bool                 _locked;
public:
	explicit unique_lock(Mutex& m, bool locked=true)
		: _mutex(m), _locked(locked) { if( locked ) { _mutex.lock(); } }
	unique_lock(LockGuard& other)
		: _mutex(other._mutex),
		  _locked(other._locked) {
		other._mutex
	}
	~unique_lock() { this->unlock(); }
	unique_lock& operator=(unique_lock& other) {
		
		return *this;
	}
	
	void unlock()     {
		if( !_locked ) {
			return;
		}
		_locked = false;
		_mutex.unlock();
	}
	void lock() {
		if( _locked ) {
			return; 
		}
		_mutex.lock();
		_locked = true;
	}
};
*/

operator       mutex_type&()       { return _mutex; }
	operator const mutex_type&() const { return _mutex; }

bool wait(UniqueLock<Mutex>& lock, double timeout_secs=-1) {
		if( timeout >= 0 ) {
			struct timespec abstime;
			check_error( clock_gettime(CLOCK_REALTIME, &abstime) );
			time_t secs = (time_t)(std::min(timeout_secs, double(INT_MAX))+0.5);
			abstime.tv_sec  += secs;
			abstime.tv_nsec += (long)((timeout_secs - secs)*1e9 + 0.5);
			int ret = pthread_cond_timedwait(&_cond, &(pthread_mutex_t&)lock.mutex(),
			                                 &abstime);
			if( ret == ETIMEDOUT ) {
				return false;
			}
			else {
				check_error( ret );
				return true;
			}
		}
		else {
			check_error( pthread_cond_wait(&_cond, &(pthread_mutex_t&)lock.mutex()) );
			return true;
		}
	}



	// Waits for the range [frame0:frame0+nframe] to be ready and returns data
	// If guarantee==true, the data span will not be overwritten until close_read
	// If guarantee==false, the user must check still_valid() after reading data
	// The returned nframe may be smaller than requested if the timeout is hit
	ReadSpan open_read(frame_idx_type frame0,
	                   size_t         nframe,
	                   bool           guarantee=false,
	                   double         timeout_secs=-1) {
		unique_lock_type lock(_mutex);
		++_nread_open;
		assert( nframe <= (_ghost_nframe+1) );
		frame0 = std::max(frame0, _tail);
		// Note: Important to insert this before waiting so that writers will
		//         not immediately overwrite before the read is done.
		if( guarantee ) {
			_guarantees.insert(frame0);
		}
		auto predicate = [&](){ return _head >= (std::max(frame0, _tail) +
		                                         (frame_idx_type)nframe); };
		bool timed_out = false;
		if( timeout_secs >= 0 ) {
			//std::chrono::nanoseconds timeout(uint64_t(timeout_secs*1e9));
			//timed_out = !_read_condition.wait_for(lock, timeout, predicate);
			timed_out = !_read_condition.wait_for(lock, timeout_secs, predicate);
		}
		else {
			_read_condition.wait(lock, predicate);
		}
		frame0 = std::max(frame0, _tail);
		if( timed_out ) {
			nframe = _head - frame0;
		}
		// Check for overlap with dirty ghost region and copy if necessary
		frame_idx_type buf_frame_beg = buf_frame(frame0);
		frame_idx_type buf_frame_end = buf_frame(frame0 + nframe);
		if( buf_frame_end < buf_frame_beg &&
		    buf_frame_end > _dirty_beg &&
		       _dirty_end > _dirty_beg) {
			copy_to_ghost(_dirty_beg, buf_frame_end - _dirty_beg);
			_dirty_beg = buf_frame_end;
		}
		pointer ptr = buf_pointer(frame0);
		//return (ReadSpan){ frame0, nframe, guarantee, ptr };
		return Read(frame0, nframe, guarantee, ptr);
	}
	ReadSpan advance_read(ReadSpan span,
	                      size_t   nframe_advance=0,
	                      double   timeout_secs=-1) {
		close_read(span);
		if( nframe_advance == 0 ) {
			nframe_advance = span.nframe;
		}
		return open_read(span.frame0 + nframe_advance,
		                 span.nframe,
		                 span.guarantee,
		                 timeout_secs);
	}
	// on write into ghost region: copy ghost region to beginning
	// on write into beginning:    mark ghost region as dirty
	// on read from ghost region:  if dirty: copy beginning to ghost region, mark as clean
	void close_read(ReadSpan span) {
		lock_guard_type lock(_mutex);
		assert( _nread_open > 0 );
		--_nread_open;
		if( span.guaranteed ) {
			_guarantees.erase(_guarantees.find(span.frame0));
			_write_condition.notify_all();
		}
	}
	bool still_valid(frame_idx_type frame0) const {
		lock_guard_type lock(_mutex);
		return frame0 >= _tail;
	}
	// ----------------

/*
			// One or both conditions were not satisfied
			// (One or zero conditions were satisfied)
			if( _realloc_pending ) {
				nframe = 0;
			}
	  open_write(nr):
	    if( _head + nr > _tail + buf_nframe ) {
	      _tail = _head + nr - buf_nframe;
	      // also deal with read guarantees
	    }
	    return buf_ptr(_head);
	    
	  close_write(nc):
	    _head += nc;
	  
	  open_write(n): (push reserve, pull tail)
	    if( _reserve_head + nr > _tail + buf_nframe ) {
	      _tail = _reserve_head + nr - buf_nframe;
	      // also deal with read guarantees
	    }
	    ptr = buf_ptr(_reserve_head);
	    _reserve_head += n;
	    return ptr;
	    
	  close_write(start, n): (push head)
	    wait until _head == start on cond1
	    _head += n;
	    notify cond1
	  
	  w1 = ring.open_write_exactly(n); // Cannot be called if there is an open write_upto
	  w1.advance(n_new);
	  w1.close();
	  w2 = ring.open_write_upto(nr); // Only one allowed at a time
	  w2.advance(nc, nr_new);
	  w2.close(nc);
	  r = ring.open_read(f0, nr);
	  r.advance(nc, nr_new);
	  r.close();
	  
	  open_read(n):
	    
	 */
	WriteSpan _open_write(size_t nframe_reserve,
	                      double timeout_secs,
	                      unique_lock_type& lock) {
		assert( _nwrite_open == 0 );
		++_nwrite_open;
		//std::cout << nframe_reserve << " vs. " << _ghost_nframe << std::endl;
		assert( nframe_reserve <= (_ghost_nframe+1) );
		frame_idx_type new_head = _head + nframe_reserve;
		frame_idx_type buf_nframe = _buf.size() / _frame_size;
		if( new_head - _tail > buf_nframe ) {
			frame_idx_type new_tail = new_head - buf_nframe;
			if( !_guarantees.empty() ) {
				auto predicate = [&](){ return *_guarantees.begin() >= new_tail; };
				bool timed_out = false;
				if( timeout_secs >= 0 ) {
					//std::chrono::nanoseconds timeout(uint64_t(timeout_secs*1e9));
					//timed_out = !_write_condition.wait_for(lock, timeout, predicate);
					timed_out = !_write_condition.wait_for(lock, timeout_secs, predicate);
				}
				else {
					_write_condition.wait(lock, predicate);
				}
				if( timed_out ) {
					//std::cout << nframe_reserve << ": " << _tail << " --> " << new_tail << " vs. " << *_guarantees.begin() << std::endl;
					size_t ncrop = std::min(new_tail - *_guarantees.begin(),
					                        (frame_idx_type)nframe_reserve);
					nframe_reserve -= ncrop;
					new_head       -= ncrop;
					new_tail = *_guarantees.begin();
				}
			}
			_tail = new_tail;
		}
		pointer ptr = buf_pointer(_head);
		//_head = new_head;
		return WriteSpan(ptr, nframe_reserve);
	}
	void _close_write(size_t nframe_commit) {
		assert( _nwrite_open == 1 );
		--_nwrite_open;
		frame_idx_type buf_frame_beg = buf_frame(_head);
		frame_idx_type buf_frame_end = buf_frame(_head + nframe_commit);
		if( buf_frame_end < buf_frame_beg ) {
			// The write went into the ghost region, so copy to the front of the buffer
			copy_from_ghost(0, buf_frame_end);
			_dirty_beg = buf_frame_end;
		}
		else if( buf_frame_beg < (frame_idx_type)_ghost_nframe ) {
			frame_idx_type ghost_frame_end = std::min(buf_frame_end,
			                                          (frame_idx_type)_ghost_nframe);
			//copy_to_ghost(buf_frame_beg, ghost_frame_end - buf_frame_beg);
			_dirty_beg = std::min(_dirty_beg, buf_frame_beg);
			_dirty_end = std::max(_dirty_end, ghost_frame_end);
		}
		_head += nframe_commit;
		_read_condition.notify_all();
	}


	// ** TODO: Implement these
	void close_write(WriteBlock* w);
	void advance_write(WriteStream* w, size_t nframe_commit, size_t nframe_reopen,
	                   double timeout_secs=-1);
	void close_write(WriteStream* w);
	void advance_read(Read* r, ssize_t nframe_advance, size_t nframe_reopen,
	                  double timeout_secs=-1);
	void close_read(Read* r);

//typedef std::mutex                   mutex_type;
	////typedef std::recursive_mutex         mutex_type;
	//typedef std::lock_guard<mutex_type>  lock_guard_type;
	//typedef std::unique_lock<mutex_type> unique_lock_type;
	//typedef std::condition_variable      condition_type;


class IRingBuffer {
	virtual frame_idx_type tail() const = 0;
	virtual void request_size(size_t nframe_span,
	                          size_t nframe_buffer=0) = 0;
};
class IReadableRingBuffer<T> : public virtual IRingBuffer {
	virtual Read open_read(frame_idx_type frame0,
	                       size_t         nframe,
	                       bool           guarantee=false,
	                       double         timeout_secs=-1) = 0;
};
class IWritableRingBuffer<T> : public virtual IRingBuffer {
	virtual WriteBlock open_write_block(size_t nframe,
	                                    double timeout_secs=-1) = 0;
};
template<typename T, class Allocator>
class RingBuffer : public IReadableRingBuffer<T>, public IWritableRingBuffer<T> {
	
};
template<typename T>
class RingReader {
	IReadableRingBuffer* _ring;
};
template<typename T>
class RingWriter {
	IWritableRingBuffer* _ring;
};
RingBuffer<T, SystemAllocator<T> > ring;
RingReader<T>                      input_ring(&ring);
RingWriter<T>                      output_ring(&ring);

// Copy orig[_buf_frame(_tail):_buf_frame(_head)] to new[:_head-_tail]
			// Set _buf_frame0 = _tail;

// Copy orig[:_buf_frame(_head)] to new[:_buf_frame(_head)]
			// Copy orig[_buf_frame(_tail):] to new[_buf_frame(_tail)+(newsize-oldsize):]
			// Set _buf_frame0 = _head - _buf_frame(_head);


		// ** TODO: Need to insert new space between _buf_frame(_head) and _buf_frame(_tail)
		//            And also extend the ghost region
		//_buf.resize(req_buf_size);
		
		// ** TODO: Should allocate amortizing space to main buffer or ghost region?
		//          Should use Buffer object or just manage manually?
		//            Reasons to manage manually: Need to use copy()
		//                                        Control over amortizing
		//                                        Only minimal functionality of Buffer required

if( _buf_frame(_tail) < _buf_frame(_head) ) {
			//// Copy orig[:_buf_frame(_tail)] to new[:_buf_frame(_tail)]
			//// Copy orig[_buf_frame(_head)
			// Copy middle to beginning
			// Copy orig[_buf_frame(_tail):_buf_frame(_head)] to new[:_head-_tail]
			// Set _buf_frame0 = _tail;
		}
		else {
			// Copy beginning to beginning and end to end, with larger gap between
			// Copy orig[:_buf_frame(_head)] to new[:_buf_frame(_head)]
			// Copy orig[_buf_frame(_tail):] to new[_buf_frame(_tail)+(newsize-oldsize):]
			// Set _buf_frame0 = _head - _buf_frame(_head);
		}


		             
		_dirty_beg = std::min(_dirty_beg, std::max(_ghost_nframe,
		                                           _buf_frame(_tail)));
		                                           
		                                           buf_frame_beg);
		_dirty_end = std::max(_dirty_end, ghost_frame_end);


		// Allocate new, larger buffer
		// TODO: Nut out how to deal with ghost region
		//         If it grows, just need to expand dirty region to cover any overlap between
		//           the new part of the ghost region and any corresponding beginning of the
		//           buffer that's been written-to.

// TODO: How to handle connecting new task processes after pipeline is running
	//         requiring re-allocation of a ringbuffer?
	//         Set reallocation_pending and have all new opens wait until it's done?
	//           This should be doable!
	//           Reallocate sets reallocation_pending=true and waits on _read_write_condition
	//             for nwrite_open and nread_open to be 0 before performing the reallocation.
	//           open_write/read operations wait on _write_realloc_condition/_read_realloc_condition
	//             for write/read to be possible and reallocation_pending to be false
	//             before performing the open_write/read.


//#include <set>
//#include <mutex>
//#include <condition_variable>
//#include <chrono>

//typedef IAllocator<value_type>      alloc_iface;
	//typedef SystemAllocator<value_type> default_allocator;
	
	//RingBuffer(const alloc_iface& allocator=default_allocator(),

_copier(// Src: start of buffer
		        _buf,
		        _buf + nframe*_frame_size,
		        // Dst: start of ghost region
		        _buf + _buf_size);
		        
		
		_copier(// dst = buf_frame0 frames into ghost region
		                          &_buf[_buf_size + buf_frame0*_frame_size],
		                          // src = buf_frame0 frames from start of buffer
		                          &_buf[              buf_frame0*_frame_size],
		                          nframe * _frame_size);
		//_buf.get_allocator().copy(&_buf[_buf.size()], // dst = start of ghost region
		_copier(&_buf[_buf_size], // dst = start of ghost region
		                          &_buf[0],           // src = start of buffer
		                          nframe * _frame_size);


		
		//_buf.get_allocator().copy(// dst = buf_frame0 frames from start of buffer
		_copier(// dst = buf_frame0 frames from start of buffer
		                          &_buf[              buf_frame0*_frame_size],
		                          // src = buf_frame0 frames into ghost region
		                          &_buf[_buf_size + buf_frame0*_frame_size],
		                          nframe * _frame_size);

//inline const alloc_iface& get_allocator() const { return _buf.get_allocator(); }

/*
void RingBuffer::request_buffer_size(size_t nframe) {
	lock_guard_type lock(_mutex);
	assert( _nwrite_open == 0 &&
	        _nread_open == 0 );
	size_t size = nframe * _frame_size;
	if( size > _buf.size() ) {
		_buf.reserve(size + _ghost_nframe*_frame_size);
		_buf.resize(size);
	}
}
void RingBuffer::request_span_size(size_t nframe) {
	lock_guard_type lock(_mutex);
	assert( _nwrite_open == 0 &&
	        _nread_open == 0 );
	// Note: -1 because write/read will always start before ghost region
	_ghost_nframe = std::max(_ghost_nframe, nframe) - 1;
	request_buffer_size(nframe);
	_buf.reserve(_buf.size() + _ghost_nframe*_frame_size);
}
*/
/*
void RingBuffer::clear() {
	lock_guard_type lock(_mutex);
	assert( _nwrite_open == 0 &&
	        _nread_open == 0 );
	_ghost_nframe = 0;
	_tail = 0;
	_head = 0;
	_dirty_beg = 0;
	_dirty_end = 0;
	_buf.clear();
	_guarantees.clear();
	_nwrite_open = 0;
	_nread_open  = 0;
}
*/

/*
template<typename T, class A, class C, size_t MR>
void RingBuffer<T,A,C,MR>::set_frame_size(size_t n) {
	lock_guard_type lock(_mutex);
	assert( _nwrite_open == 0 &&
	        _nread_open == 0 );
	_frame_size = n;
}
*/

/*
template<typename T>
class reference_wrapper {
	T const& _ref;
	reference_wrapper(T const& t) : _ref(t) {}
	T const& get() { return _ref; }
};
template<typename T>
reference_wrapper<T> ref(T& t) { return reference_wrapper<T>(t); }
*/

/*
// Helper class for member predicate callbacks (unnecessary with C++11 lambdas)
template<class Class, class Method>
struct MemberPredicate {
	Method       _method;
	Class const* _self;
	MemberPredicate(Method method, Class const* self, )
		: _method(method), _self(self) {}
	bool operator()() const {
		return (_self.*_method)();
	}
};
template<class Class, class Method>
MemberPredicate<Class,Method> make_member_predicate(Method method, Class const* self) {
	return MemberPredicate<Class,Method>(method, self);
}
*/

		//std::copy(src, src + n, dst);
	//inline void operator()(pointer dst, const_pointer src, size_type n) const {


	//typedef bool (RingBuffer::*predicate_method_type)();
	//typedef MemberPredicate<RingBuffer,predicate_method_type> predicate_type;
	//friend predicate_type;
	//friend

/*
	size_t frame_size = 123;
	RingBuffer<uint8_t> ring(frame_size);
	ring.request_buffer_size(15);
	ring.request_span_size(10);
	//ring.open_write(10);
	//ring.close_write(10);
	ring.reserve(10);
	ring.commit(10);
	RingBuffer<uint8_t>::ReadSpan rs = ring.open_read(0, 5, true);
	cout << rs.frame0 << endl;
	*/
	//ring.close_read(rs);
	//*RingBuffer<uint8_t>::WriteSpan ws = ring.reserve(6, 2.);
	//*cout << (ws.first - ring.data()) / frame_size << ", " << ws.second << endl;
	/*
	Subscriber sub(ring);
	?? = sub.read(blocksize);
	rx->recv_block(??.data, ??.nframe*sub.frame_size(), packet_size, ...);
	//sub.seek(-overlap, SEEK_CUR);
	sub.advance(blocksize - overlap);
	*/
	/*
	size_t frame_size = 123;
	RingBuffer<uint8_t> ring(frame_size);
	ring.request_size(10, 15);
	RingBuffer<uint8_t>::WriteStream w = ring.open_write_stream(10);
	w.advance_by(3);
	*/
	//std::cout << "AFTER" << std::endl;
	//RingBuffer<uint8_t>::Read r = ring.open_read(0, 5);
	
	//ring.reset_write();

/*
	w = ring.open_write_stream(7);
	assert( &w[0] == data + 0 );
	for( size_t i=0; i<7; ++i ) {
		w[i] = i;
	}
	for( size_t i=0; i<1*7; ++i ) {
		assert( data[i] == i );
	}
	w.advance_by(7);
	assert( &w[0] == data + 7 );
	for( size_t i=0; i<7; ++i ) {
		w[i] = 7 + i;
	}
	for( size_t i=0; i<2*7; ++i ) {
		assert( data[i] == i );
	}
	cout << ring.capacity() << endl;
	w.advance_by(nframe_commit);
	assert( &w[0] == data + 2*7 );
	for( size_t i=0; i<7; ++i ) {
		w[i] = 2*7 + i;
	}
	for( size_t i=0; i<3*7; ++i ) {
		assert( data[i] == i );
	}
	print_ring(ring);
	w.close();
	for( size_t i=0; i<3*7; ++i ) {
		if( i < 7 - (ring.size() % 7) ) {
			assert( data[i] == ring.size() + i );
		}
		else {
			assert( data[i] == i );
		}
	}
	print_ring(ring);
	*/
	
	//std::cout << "HERE" << std::endl;
	/*
	assert( &w[0] == data + 2*nframe_commit );
	for( size_t i=0; i<nframe_commit; ++i ) {
		w[i] = 2*nframe_commit + i;
	}
	for( size_t i=0; i<3*nframe_commit; ++i ) {
		assert( data[i] == i );
	}
	w.close();
	*/

_payload_ring.request_span_size(   chunk_nframe);
		_payload_ring.request_buffer_size(buffer_nframe);


		_nscatter_chunk = 3;
		size_t buffer_factor  = 3;
		size_t buffer_size_min = chunk_size_min * (nscatter_chunk + buffer_factor-1);
		size_t buffer_nframe = ((buffer_size_min-1)/frame_size+1) / sizeof(value_type);
		_chunk_nframe = (( chunk_size_min-1)/frame_size+1) / sizeof(value_type);


					
				while( pkt_begin >= cur_byte + _output_queue.size()*chunk_size_out ) {
					if( _output_queue.size() == _nscatter_chunk ) {
						
					}
					_output_queue.push_front(_output_ring.open_write(_chunk_nframe_out));
				}
					
				size_t queue_idx = (pkt_byte_offset - cur_byte) / chunk_size_out;
				while( queue_idx >= _output_queue.size() ) {
					if( _output_queue.size() == _nscatter_chunk ) {
						// TODO: Record missing packets and fill with fill_char
						// TODO: Probably want to maintain a mask output ring,
						//         which can be used to memset missing packets
						//         and by downstream tasks.
						_output_queue.pop_back();
						--queue_idx;
						cur_byte += chunk_size_out;
					}
					_output_queue.push_front(_output_ring.open_write(_chunk_nframe_out));
				}
				ssize_t buf_start = cur_byte + queue_idx*chunk_size_out;
				//::memcpy(&_output_queue[queue_idx][frame_offset - buf_start],
				//         frame_ptr, frame_size);
				_input_ring->get_copier()(frame_ptr, frame_ptr + frame_size,
				                          &_output_queue[queue_idx][frame_byte_offset - buf_start]);
				if( !in.still_valid() ) {
					// Blank-out bad data that were just copied
					::memset(frame_ptr, _fill_char, frame_size);
					// Skip ahead to latest data
					in.advance_to(in.head() - _chunk_nframe);
				}

//size_t  frame_size_in = _input_ring->frame_size();
		//size_t  chunk_size    = _chunk_nframe_in * frame_size * sizeof(value_type);


template<class Ringbuf>
class ReorderPackets {
public:
	ReorderPackets() {
		this->add_input("packet_data");
		this->add_input("packet_sizes");
		this->add_input("packet_sources");
		this->add_output();
	}
	virtual void init( ) {
		this->input("packet_data") = ...;
		this->input("packet_data")->request_size(...);
		// ...
		this->output()->set_frame_size(...);
		// ...
	}
	virtual void main() {
		// ...
		while( !this->stop_requested() ) {
			// ...
		}
	}
};

int main() {
	ReorderPackets reorder_task;
	reorder_task.input("packet_data") = recv_task.output("packet_data");
}

void start(bool async=true) {
		_running_flag.test_and_set();
		if( async ) {
			_thread = std::thread([&]() { this->main(); });
		}
		else {
			this->main();
		}
	}

bool running() {
		// Note: This would be simpler with atomic<bool>, but atomic_flag works
		//         and is guaranteed to be lock-free.
		bool was_true = _running_flag.test_and_set();
		if( !was_true ) {
			_running_flag.clear();
		}
		return was_true;
	}

std::thread spawn() {
		return std::thread(&ReorderPackets::main, this);
	}
	void quit() {
		_running_flag.clear();
	}

bool running() {
		// Note: This would be simpler with atomic<bool>, but atomic_flag works
		//         and is guarantee_readsd to be lock-free.
		bool was_true = _running_flag.test_and_set();
		if( !was_true ) {
			_running_flag.clear();
		}
		return was_true;
	}

//size_t nrecv_bytes = _stats.nrecv_bytes.exchange(0);
			//size_t nrecv       = _stats.nrecv.exchange(0);
			//size_t ndrop       = _stats.ndrop.exchange(0);

//nrecv_bytes/1e9*8/dt,
			       //nrecv/dt,
			       //ndrop/dt,

//(double)_stats_tot.nrecv_bytes,
			       //(size_t)_stats_tot.nrecv,
			       //(size_t)_stats_tot.ndrop,

_stats_tot.nrecv_bytes += _socket.get_recv_size();
			_stats_tot.nrecv       += nframe;
			_stats_tot.ndrop       += _socket.get_drop_count();

// TODO: Consider replacing manage() with a named constructor
	//         Will (probably) need move semantics
	//Socket(int fd)   : _fd(-1), _type(type), _family(AF_UNSPEC),
	//                   _mode(Socket::MODE_CLOSED)

//// Manage an existing socket (usually one returned by a call to accept())
	//void manage(int fd);

void Socket::manage(int fd) {
	if( _mode != Socket::MODE_CLOSED ) {
		throw Socket::Error("Socket is already open");
	}
	_fd = fd;
	if( this->get_option<int>(SO_TYPE) != _type ) {
		throw Socket::Error("Socket type does not match");
	}
	_family = this->get_option<int>(SO_DOMAIN);
	if( this->get_option<int>(SO_ACCEPTCONN) ) {
		_mode = Socket::MODE_LISTENING;
	}
	else {
		// Not listening
		try {
			_mode = Socket::MODE_CONNECTED;
			this->get_remote_address();
		}
		catch( Socket::Error const& ) {
			// Not connected
			try {
				_mode = Socket::MODE_BOUND;
				this->get_local_address();
			}
			catch( Socket::Error const& ) {
				// Not bound
				_mode = Socket::MODE_CLOSED;
			}
		}
	}
	this->set_default_options();
}

/*
	  int fd;
	  struct ifreq ifr;
	  //fd = ::socket(family, socktype, 0);
	  fd = ::socket(AF_INET, socktype, 0);
	  ifr.ifr_addr.sa_family = family;
	  std::strncpy(ifr.ifr_name, ifname, IFNAMSIZ-1);
	  if( ::ioctl(fd, SIOCGIFADDR, &ifr) == -1 ) {
	  ::close(fd);
	  return 0;
	  }
	  ::close(fd);
	  ::memcpy(address, &ifr.ifr_addr, sizeof(ifr.ifr_addr));
	  return 1;
	*/

                   
  TODO: MTU discovery:
         To get an initial estimate of the path MTU, connect a datagram
           socket to the destination address using connect(2) and
           retrieve the MTU by calling getsockopt(2) with the IP_MTU
           option.
         IP_MTU (since Linux 2.2)
              Retrieve the current known path MTU of the current socket.
              Returns an integer.
              IP_MTU is valid only for getsockopt(2) and can be employed
              only when the socket has been connected.
// Note: Must be connected before calling this
//         Datagram client sockets can call connect(address(...)),get_mtu(),connect(address())
int get_mtu() const {
  return this->get_option(IP_MTU);
}
static int discover_mtu(remote_address) {
  Socket s;
  s.connect(remote_address, SOCK_DGRAM);
  return s.get_option(IP_MTU);
}
  
    UDP recv from any:       bind(local_addr), recv()/recvmsg()/recvmmsg()
  x UDP recv from specified: recvfrom(remote_addr)
    UDP send to specified:   connect(remote_addr), send()/sendmsg()/sendmmsg()
  x                       OR sendto(remote_addr)
    UDP send to any:         sendto(remote_addr)/sendmsg(remote_addr)/sendmmsg(remote_addrs)
  ** TODO: Need to allow Socket to open (i.e., call socket()) separately from bind()
  So the three supported UDP socket modes should be:
      bind(local_addr), recvmmsg() (and sendmmsg(remote_addrs))
  connect(remote_addr), sendmmsg() (and recvmmsg())
                        sendmmsg(remote_addrs) (and recvmmsg())
    And so: bind(local_addr) is for when a socket needs to be referenced by name
            <nothing>        is for when a socket does not need to be referenced by name
            connect()        is for when a socket only talks to one named destination
    [socket(domain=[AF_UNIX,AF_INET,AF_INET6], type=[SOCK_STREAM,SOCK_DGRAM], protocol=0)]
    
    // TODO: Add static any_addr(int family(=AF_UNSPEC)) using INADDR_ANY or in6addr_any (UNIX version?)
    //       Add check for family==AF_UNSPEC before trying everything in inet_addr()
    //       Consider changing inet/unix_addr --> address(const char* addr, unsigned short port, int family=AF_UNSPEC)
    //         AF_UNSPEC: port==0 => AF_UNIX, else not UNIX
    
    // Client initialisation
    void connect(sockaddr_storage remote_address, int type);
      // TODO: If socket already created and is UDP and family is the same and type is the same (UDP),
      //         then don't re-create the socket, just call connect() again, which is allowed.
      //       If socket already created and is UDP and remote_address.sa_family==AF_UNSPEC, then call
      //         connect to cause 'unconnection', which is allowed.
      socket(sa.family, type); if( sa.addr ) connect(sa.addr); else if( type==TCP ) error();
    // Server initialisation
    void listen(sockaddr_storage local_address, int type);
      socket(sa.family, type); bind(); if(TCP) listen();
      
  Note: Calling connect() on a UDP socket with sa.family==AF_UNSPEC will 'unconnect' the socket

  Address
    Socket::address("localhost",   8100, AF_INET6);
    Socket::address("192.168.0.1", 8001);
    Socket::address("/mypipe",        0);
    Socket::any_address(AF_INET);
    Socket::any_address(AF_INET6);
    Socket::any_address(AF_UNIX);
    Socket::any_address(AF_UNSPEC); // For 'unconnecting' only; throws error if passed to connect(UDP) for first time
    //Socket::no_address(); (returns family AF_UNSPEC)
  UDP client
    Socket socket(SOCK_DGRAM); // ** TODO: Use this, and remove type parameter from initialisation methods
    socket.connect(Socket::any_address(...),  SOCK_DGRAM);
    socket.connect(Socket::address(...),      SOCK_DGRAM);
    socket.connect(Socket::address(...),      SOCK_DGRAM);
    socket.connect(Socket::any_address(),    SOCK_DGRAM);
  UDP server
    socket.listen(Socket::address(...), SOCK_DGRAM);
  TCP client
    socket.connect(Socket::address(...), SOCK_STREAM);
  TCP server
    socket.listen(Socket::address(...), SOCK_STREAM);
    client_socket.manage(socket.accept());


		//std::string recv_task_name = recv_task->name();
		//cout << recv_task_name << endl;
		//auto recv_task2 = _pipeline.get_task<RecvUDP>(recv_task_name);
		//cout << reorder_task->name() << endl;

/*
		auto tstop = std::async(std::launch::async, [&]() {
				std::this_thread::sleep_for(std::chrono::seconds(run_for_secs));
				_pipeline.request_stop();
			});
		*/

/*
template<class Allocator=std::allocator<uint8_t>,
         class CopyFiller=StdCopyFiller<uint8_t>,
         size_t MAX_READERS=RINGBUFFER_DEFAULT_MAX_READERS>
class RingBuffer {
public:
	
private:
	
};
template<typename T, class RingBufferType=RingBuffer<> >
class RingBlockWriter {
	
};
template<typename T, class RingBufferType=RingBuffer<> >
class RingStreamWriter {
	
};
template<typename T, class RingBufferType=RingBuffer<> >
class RingReader {
	
};
*/


class MyTask : public Task {
public:
	MyTask(std::string name="") : Task(name) {
		this->add_output("data", true);
		this->add_output("sizes");
		this->add_output("sources");
	}
	//void init(ring_type* data_ring) {
	//	_data_ring = data_ring;
	//}
	virtual void main() {
		auto data_output = this->output("data");
		auto size_output = this->output("sizes");
		auto addr_output = this->output("sources");
		// ...
	}
private:
	ring_type* _data_ring;
};

int main() {
	Pipeline pipeline;
	
	// ** TODO: Tasks need access to pipeline to allow lookup of ringbuffers
	//auto     task = pipeline.add_task(new RecvUDP);
	//RecvUDP* task = new RecvUDP("RecvUDP");
	RecvUDP* task = pipeline.create_task("RecvUDP");
	// TODO: How to register named creators?
	
	////task->init(pipeline.get_ring(task->name()+".data_ring"));
	
	task1b->output() = task1->output();
	
	//task2->input()        = task1->output();
	//task2->input("sizes") = task1->output("sizes");
	task1b->output()      = "task1";
	task2->input()        = "task1";
	task2->input("sizes") = "task1.sizes";
	
	task2->param("guarantee_reads") = false;
	task2->param("stats_interval")  = 1.0;
	
	//task2->launch();
}


class TaskPlugin {
public:
	TaskPlugin(std::string name) : _lib((name+".so").c_str(), _create(0) {}
	Task* create() {
		_lib.symbol("create");
		const std::string create_symbol = "create";
		assert(_lib);
		if( !_create ) {
			_create = (create_func_type)::dlsym(_lib, create_symbol.c_str());
			if( !_create ) {
				throw std::runtime_error("Failed to find symbol "+create_symbol);
			}
		}
		return (*_create)();
	}
private:
	typedef Task* (*create_func_type)();
	DynLib _lib;
	create_func_type _create;
};

int main() {
	
	typedef int (*some_func)(char *param);
	void* myso = dlopen("/path/to/my.so", RTLD_NOW);
some_func* func = dlsym(myso, "function_name_to_fetch");
func("foo");
dlclose(myso);
}


//data_ring_type* data_output() { return &_data_ring; }
	//size_ring_type* size_output() { return &_size_ring; }
	//addr_ring_type* addr_output() { return &_addr_ring; }

//typedef RingBuffer<value_type>       data_ring_type;
	//typedef RingBuffer<size_t>           size_ring_type;
	//typedef RingBuffer<sockaddr_storage> addr_ring_type;

//Value& pipeline_value = get_key<(document, "pipeline");
	//if( !pipeline_value.is<Object>() ) {
	//	throw std::runtime_error("Pipeline is not an object");
	//}


template<class Ringbuf>
class ReorderPackets {
public:
	ReorderPackets() {
		this->add_input("packet_data");
		this->add_input("packet_sizes");
		this->add_input("packet_sources");
		this->add_output();
	}
	virtual void init( ) {
		this->input("packet_data") = ...;
		this->input("packet_data")->request_size(...);
		// ...
		this->output()->set_frame_size(...);
		// ...
	}
	virtual void main() {
		// ...
		while( !this->stop_requested() ) {
			// ...
		}
	}
};

int main() {
	ReorderPackets reorder_task;
	reorder_task.input("packet_data") = recv_task.output("packet_data");
}

size_t buffer_size     =          buffer_npacket * packet_size_max;
	size_t   gulp_size     =            gulp_npacket * packet_size_max;

inline std::thread spawn_stats() {
		return std::thread(&RecvUDP::stats_main, this);
	}

/*
	void init(const char* addr, unsigned short port,
	          size_t packet_size_max=9000,
	          double timeout_secs=1.,
	          double stats_interval=1.) {
		size_t  chunk_size_min = 64*1024*1024;
		size_t buffer_size_min = chunk_size_min * 3;//256*1024*1024;
		// TODO: Clean this up a bit, and use a div_ceil function
		size_t buffer_nframe = ((buffer_size_min-1)/packet_size_max+1) / sizeof(value_type);
		       _chunk_nframe = (( chunk_size_min-1)/packet_size_max+1) / sizeof(value_type);
		_data_ring.set_frame_size(packet_size_max);
		_data_ring.request_size(_chunk_nframe, buffer_nframe);
		_size_ring.set_frame_size(1);
		_size_ring.request_size(_chunk_nframe, buffer_nframe);
		_addr_ring.set_frame_size(1);
		_addr_ring.request_size(_chunk_nframe, buffer_nframe);
		
		_socket.listen(Socket::address(addr, port));
		
		_timeout_secs   = timeout_secs;
		_stats_interval = stats_interval;
	}
	*/

//size_t      _chunk_nframe;
	//double      _timeout_secs;
	//double      _stats_interval;

data_ring_type _data_ring;
	size_ring_type _size_ring;
	addr_ring_type _addr_ring;

	try { substitute_reference(pipeline, process_attrs); }
	catch( std::out_of_range const& ) {
		try { substitute_reference(pipeline, pipeline); }
		catch( std::out_of_range const& ) {
			substitute_reference(pipeline, get_key<Object>(pipeline, "__tasks__"));
		}
	}
	
	// pipeline.attrs.$blah --> process_attrs.blah
	// pipeline.tasks.task1.$blah -->
	Object& dst = get_key<Object>(pipeline, "attributes");
	for( Object::iterator it=dst.begin(); it!=dst.end(); ++it ) {
		substitute_reference(it->second);
		// TODO: Apply globally or to specific subsets/objects?
		//         Start with src=process_attrs, then src=pipeline, then src=tasks
	}
	
	// Substitute all dst[] = "$KEY" --> src["KEY"]
	for( picojson::object::iterator it=dst.begin(); it!=dst.end(); ++it ) {
		picojson::value& val = it->second;
		if( val.is<std::string>() ) {
			std::string sval = val.get<std::string>();
			if( sval[0] == '$' ) {
				std::string key = sval.substr(1);
				picojson::object::const_iterator src_it = src.find(key);
				if( src_it == src.end() ) {
					throw std::runtime_error("Template key not found: "+key);
				}
				val = src_it->second;
			}
		}
	}


	
	try { sval = get_key<std::string>(val); }
	catch( TypeError const& ) { return val; }
	if( sval[0] == '$' ) {
		std::string key = sval.substr(1);
		val = substitute_reference(find_key_reference(key, src));
	}

		if( sval[0] == ref_symbol ) {

//// Perform template substitutions ("$KEY" --> process_attrs["KEY"])
		//apply_template_substitutions(task_attrs, pipeline, process_attrs);

"attributes": {


{
	"pipeline": {
		"__name__": "ADP-Pipeline",
		
		"__tasks__": {
			"PacketCapture": {
					...
			}
		}
	}
}

"packet_size_max": "$pipeline.__tasks__.PacketCapture/packet_size_max",
				"packet_size_max": "$PacketCapture.packet_size_max",
				"packet_size_max": "$pipeline.tasks.PacketCapture.attrs.packet_size_max",
				"packet_size_max": "$pipeline.__tasks__.PacketCapture.packet_size_max",

/*
	std::shared_ptr<T> add_task(T* task) {//, cpu_core_type cpu_core=-1) {
		auto ret = _tasks.insert(std::make_pair(task->name(),
		                                        std::make_pair(task_pointer(task))));
		if( !ret.second ) {
			throw std::runtime_error("Task "+task->name()+" already added");
		}
		task_pointer ptr = ret.first->second;//.first;
		return std::static_pointer_cast<T>(ptr);
	}
	template<class T>
	std::shared_ptr<T> get_task(std::string name) {
		task_map_type::iterator it = _tasks.find(name);
		if( it == _tasks.end() ) {
			throw std::runtime_error("No task named "+name+" in pipeline");
		}
		task_pointer ptr = it->second;//.first;
		return std::static_pointer_cast<T>(ptr);
	}
	*/

if( name.empty() ) {
			_name = "AnonymousTask"+std::to_string(Task::population++);
		}

Task(Pipeline*             pipeline,
     std::string           name,
     picojson::object      definition,
     std::set<std::string> inputs,
     std::set<std::string> outputs) {
	// TODO: Add all inputs/outputs to _inputs/_outputs maps (default-construct values to NULL)
	//       For each (input_name, ring_name) entry in definition["inputs"]:
	//         If input_name not in inputs: throw error("No input named "+input_name);
	//         _inputs[input_name] = pipeline->get_ring(ring_name);
	//       Repeat for outputs
	//       For each (output_name, ring) entry in _outputs:
	//         If ring is NULL: _outputs[output_name] = pipeline->get_ring(task_name+"."+output_name);
	//
	//       Pipeline::ring_type& get_output(std::string name) { return get_key(_outputs, name); }
	//       Pipeline::ring_type& get_input( std::string name) { ret = get_key(_inputs,  name); if( !ret ) { throw error("Input "+name+" is not connected"); } else return ret; }
	
	_inputs[input_name] = pipeline->get_ring(ring_name);
	_outputs[] = pipeline->get_ring(name+ );
}

std::atomic<size_t> Task::population(0);


	for( Object::iterator it=obj.begin(); it!=obj.end(); ++it ) {
		if( it->second.is<std::string>() ) {
			std::string s = it->second.get<std::string>();
			if( s.substr(0, prefix.size()) == prefix ) {
				s = replacement + s.substr(prefix.size());
			}
		}
		else if( it->second.is<Object>() ) {
			replace_all_string_prefix(it->second.get<Object>(), prefix, replacement);
		}
	}

cout << prefix << "->" << suffix << " in " << Value(src) << endl;
		cout << endl;
		cout << Value(get_key<Object>(src, prefix)) << endl;

// TODO: Generalise this to replace all references, including chained references (detect loops)
//         $TaskName.attr_name.subattr_name
//         $global_attr_name.subattr_name

// TODO: Add all inputs/outputs to _inputs/_outputs maps (default-construct values to NULL)
		//       For each (input_name, ring_name) entry in definition["inputs"]:
		//         If input_name not in inputs: throw error("No input named "+input_name);
		//         _inputs[input_name] = pipeline->get_ring(ring_name);
		//       Repeat for outputs
		//       For each (output_name, ring) entry in _outputs:
		//         If ring is NULL: _outputs[output_name] = pipeline->get_ring(task_name+"."+output_name);
		//


	//virtual void init(std::map<std::string,Object/Property> params) = 0;
	

	//ring_type*  output(std::string name="") { return &_outputs.find(name)->second; }
	//ring_type*&  input(std::string name="") { return   _inputs.find(name)->second; }
	
/*
		Value keyval = substitute_references
(Value(key), rootsrc,
		                                     ref_prefix, sep,
		                                     opensym, closesym);
		if( !keyval.is<std::string>() ) {
			throw std::invalid_argument("Referenced object key is not a string");
		}
		key = keyval.get<std::string>();
		*/

//else if( key.find('[') == 0 ) {


	if( key.substr(0,sep.size()) != sep &&
	    key.substr(0,1) != '[' ) {
		throw std::invalid_argument("Expected "+sep+" or '['");
	}
	if( 
	
	// "$.a"      --> root["a"]
	// "$.a.b"    --> root["a"]["b"]
	// "$.a[x]    --> root["a"]["x"]
	// ".a.b[x][y]"
	size_t objsplit = key.find(sep);
	size_t arrsplit = key.find("[");
	if( objsplit == std::string::npos &&
	    arrsplit == std::string::npos ) {
		return get_key(src, key);
	}
	else if( objsplit != std::string::npos &&
	         (arrsplit == std::string::npos ||
	          objsplit < arrsplit) ) {
		std::string prefix = key.substr(0, split);
		std::string suffix = key.substr(split+sep.size());
		// TODO: Can't support nested references with . notation (would need open/close, like [])
		//         Actually, this would work, but only for accessing root nodes
		//           E.g., "$a.b.c"  == root[a][b][c]
		//                 "$a.$b.c" == root[a][root[b]][c]
		//std::string key = substitute_references(Value(prefix), src, ref_prefix, sep);
		return find_key_by_reference(suffix, get_key<Object>(src, prefix),
		                             ref_prefix, sep);
	}
	else {
		size_t arrend = key.find(']');
		if( arrend == std::string::npos ) {
			throw std::invalid_argument("Syntax error: expected ']'");
		}
		std::string prefix    = key.substr(0, arrsplit);
		std::string index_key = key.substr(arrsplit+1, arrend);
		std::string suffix    = key.substr(arrend+1);
		get_key<List>(src, index_key
	}
	
	size_t split = key.find(sep);
	if( split == std::string::npos ) {
		return get_key(src, key);
	}
	else {
		std::string prefix = key.substr(0, split);
		std::string suffix = key.substr(split+sep.size());
		return find_key_by_reference(suffix, get_key<Object>(src, prefix), sep);
	}

if( cpu_core >= 0 ) {
			bind_to_core(cpu_core);
		}
		/*
		  if( gpu_index ) {
		    cudaSetDevice(gpu_index);
		  }
		 */

 
 //cout << Value(positional_args) << endl;
	//cout << Value(keyword_args) << endl;
	//process_attrs["DATA_IN_PORT"] = Value((int64_t)4015);
//pipeline.load("adp_pipeline.json", keyword_args);
 
/*
template<typename M>
typename M::mapped_type& get_key(M const& m, typename M::key_type const& key) {
	typename M::iterator it = m.find(key);
	if( it == m.end() ) {
		throw std::out_of_range(
	}
}
*/
//class Pipeline;

/*
MyTask::MyTask(Pipeline*        pipeline,
               std::string      name,
               picojson::object definition)
	: Task(pipeline, name, definition,
	       {"my_input1",
			"my_input2"},
	       {"my_output1",
			"my_output2"}) {
	Pipeline::ring_type& my_input1 = this->get_input("my_input1");
	my_input1.request_size(...);
}
*/
//template<class RingBuf>

//size_t seq = htonll(header->seq);
		//size_t idx = htonll(header->idx);
		//cout << "Packet: " << seq << ", " << idx << endl;
		//printf("%x:%hu\n", packet_source.sin_addr.s_addr, packet_source.sin_port);

//auto stats_thread = this->spawn_stats();


#Pipeline.o: Pipeline.hpp Pipeline.cpp
#	g++ -c -g -Wall -O3 -std=c++11 -fPIC Pipeline.cpp

#Task.o: Task.hpp Task.cpp
#	g++ -c -g -Wall -O3 -std=c++11 -fPIC Task.cpp

#RecvUDP.so: RecvUDP.cpp RecvUDP.hpp Pipeline.o Task.o
#	g++ -shared -Wl,--version-script=plugin.version,-soname,RecvUDP.so.1 -fPIC -o RecvUDP.so -g -Wall -O3 -std=c++11 RecvUDP.cpp Pipeline.o Task.o


template <template<class,class,class...> class C, typename K, typename V, typename... Args>
V GetWithDef(const C<K,V,Args...>& m, K const& key, const V & defval)
{
    typename C<K,V,Args...>::const_iterator it = m.find( key );
    if (it == m.end())
        return defval;
    return it->second;
}
/*
// Returns m[key], assigning it default_value after construction
template<typename MapType, typename T>
inline typename MapType::mapped_type& init_get(MapType& m,
                                               typename MapType::key_type const& key,
                                               T const& default_value) {
	auto it = m.find(key);
	if( it == m.end() ) {
		auto& val = m[key];
		val = default_value;
		return val;
	}
	else {
		return it->second;
	}
}
template<typename MapType>
inline typename MapType::mapped_type& init_get(MapType& m,
                                               typename MapType::key_type const& key) {
	return init_get(m, key, 0);
}
*/

// TODO: Could avoid this by replacing lookups in main with a function
	//         that default-constructs the (atomic) value to zero if it
	//         doesn't already exist.

/*
	struct Stats {
		std::atomic<size_t> nrecv;
		std::atomic<size_t> nrecv_bytes;
		std::atomic<size_t> ngood;
		std::atomic<size_t> ngood_bytes;
		std::atomic<size_t> nignored;
		std::atomic<size_t> nignored_bytes;
		std::atomic<size_t> nlate;
		std::atomic<size_t> nlate_bytes;
		std::atomic<size_t> nrepeat;
		std::atomic<size_t> nrepeat_bytes;
		std::atomic<size_t> noverwritten;
		std::atomic<size_t> noverwritten_bytes;
		std::atomic<size_t> nmissing_bytes;
		std::atomic<size_t> npending_bytes;
	} _stats;
	*/

_stats.nrecv_bytes = 0;
	_stats.nrecv       = 0;
	_stats.ndrop       = 0;
	Stats tot_stats;

tot_stats.nrecv_bytes = 0;
	tot_stats.nrecv       = 0;
	tot_stats.ndrop       = 0;

Stats new_stats;
		new_stats.nrecv_bytes = _stats.nrecv_bytes.exchange(0);
		new_stats.nrecv       = _stats.nrecv.exchange(0);
		new_stats.ndrop       = _stats.ndrop.exchange(0);
		tot_stats.nrecv_bytes += new_stats.nrecv_bytes;
		tot_stats.nrecv       += new_stats.nrecv;
		tot_stats.ndrop       += new_stats.ndrop;

/*
  do {
  nsuccess = 0;
  while( !new_tasks.empty() ) {
    try:   create_task(new_tasks.front()); ++nsuccess;
    catch: deferred_tasks.insert(new_tasks.front());
    new_tasks.erase(new_tasks.front());
    if( new_tasks.empty() ) {
      if( nsuccess==0 ) {
        throw std::runtime_error("Pipeline contains cyclic dependency");
      }
      std::swap(new_tasks, deferred_tasks);
    }
  }
  
 */


/*
template<typename F>
inline __device__
//T float2fixed(F x, int decimal_bits) {
int float2fixed(F x, float scale, float offset, float maxval) {
	//x *= ((T)1 << decimal_bits);
	x = x * scale + offset;
	// Saturate
	//x = max(x, (F)(-std::numeric_limits<T>::max()));
	//x = min(x, (F)(+std::numeric_limits<T>::max()));
	x = max(x, -maxval);
	x = min(x, +maxval);
	// Round to nearest even
	return __float2int_rn(x);
}
template<typename T, typename F>
inline __host__ __device__
F fixed2float(T x, int decimal_bits) {
	return (F)(x) * ((F)1 / ((T)1 << decimal_bits));
}
*/

/*
		case 4: {
			// TODO: Replace with call to unpack function
			ret = make_float2((int)val >> 4,
			                  (int)val << (32-4) >> (32-4));
			break;
			}*/

/*
	Object& tasks = get_key<Object>(pipeline, "__tasks__");
	for( Object::iterator it=tasks.begin(); it!=tasks.end(); ++it ) {
		std::string task_name    = it->first;
		if( task_name == "__comment__" ) {
			continue;
		}
		Object&     task_def     = it->second.get<Object>();
		std::string task_type    = get_key<std::string>(task_def, "__type__");
		//Object&     task_attrs   = get_key<Object>(task_def, "attributes");
		this->create_task(task_type, task_name, task_def);
		
		try {
			this->create_task(task_type, task_name, task_def);
		}
		catch( std::out_of_range const& e ) {
			// Move task def to "deferred_tasks" set
		}
	}
	*/

/*,
	     string_set const& inputs,
	     string_set const& outputs*/

/*
	ring_map::const_iterator it = _inputs.find(name);
	if( it == _inputs.end() ) {
		throw std::out_of_range("No input named "+name);
	}
	if( !it->second ) {
		throw std::runtime_error("Input "+name+" is not connected");
	}
	return *it->second;
	*/


		// Create all outputs
		// if( this->get_property<std::string>("output_space", "auto") == "auto" ) {
		//   // Note: get_input() will throw Task::DependencyError if the target is not yet created
		//   space = this->get_input("...");
		// }
		// this->create_output(space);


		for( auto const& kv : this->get_property<Object>("__inputs__") ) {
			std::string name = kv.first;
			//this->create_output(space=
		}

/*
			if( space == "auto" ) {
				space = this->get_input(input_name).space();
			}
			*/

// TODO: This
			//assert( this->get_input(input_name).space() ==
			//        this->get_output(input_name).space() );


// outputs: {{"data", "system"}, {"sizes", "system"}, {"sources": "system"}

/*
	struct Stats {
		size_t nrecv_bytes;
		size_t nrecv;
		size_t ndrop;
	};
	struct AtomicStats {
		std::atomic<size_t> nrecv_bytes;
		std::atomic<size_t> nrecv;
		std::atomic<size_t> ndrop;
	} _stats;
	*/

//{}, {"data", "sizes", "sources"}),

{"data", "sizes", "sources"},
	       {"data"}


template<typename T>
inline T get_key(Object const& obj, std::string key) {
	Value const& val = get_key(obj, key);
	typedef typename repr_type<T>::type RT;
	if( !val.is<RT>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<RT>();
}
template<typename T>
inline T get_key(Object const& obj, std::string key, T default_value) {
	try {
		return get_key<T>(obj, key);
	}
	catch( std::out_of_range const& ) {
		return default_value;
	}
}
template<>
inline Object& get_key<Object>(Object& obj, std::string key) {
	Value& val = get_key(obj, key);
	if( !val.is<Object>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<Object>();
}
template<>
inline Object const& get_key<Object>(Object const& obj, std::string key) {
	Value const& val = get_key(obj, key);
	if( !val.is<Object>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<Object>();
}
/*
template<typename T>
inline std::vector<T> get_key(Object const& obj, std::string key) {
	List vals = get_key<List>(obj, key);
	typedef typename repr_type<T>::type RT;
	std::vector<T> ret;
	for( List::const_iterator it=vals.begin(); it!=vals.end(); ++it ) {
		Value val = *it;
		if( !val.is<RT>() ) {
			throw TypeError("Wrong type for list element: "+key);
		}
		ret.push_back(val.get<RT>());
	}
	return ret;
}
template<typename T>
inline std::vector<T> get_key(Object const& obj, std::string key,
                              std::vector<T> default_value) {
	try {
		return get_key<std::vector<T> >(obj, key);
	}
	catch( std::out_of_range const& ) {
		return default_value;
	}
}
*/


/*
  //T            get<T>
  Object&      find_object
  List&        find_list
  std::vector<T> find_list<T>
  int64_t&     find_int
  std::string& find_string
 */

/*
// Convenience functions
template<typename T>
inline T& get_key(Object& obj, std::string key) {
	Value& val = get_key<Value>(obj, key);
	if( !val.is<T>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<T>();
}
template<typename T>
inline T const& get_key(Object const& obj, std::string key) {
	Value const& val = get_key<Value>(obj, key);
	if( !val.is<T>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<T>();
}
template<>
inline Value& get_key<Value>(Object& obj, std::string key) {
	Object::iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
template<>
inline Value const& get_key<Value>(Object const& obj, std::string key) {
	Object::const_iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
template<typename T>
inline T const& get_key(Object const& obj, std::string key, T const& default_value) {
	try {
		return get_key<T>(obj, key);
	}
	catch( std::out_of_range const& ) {
		return default_value;
	}
}
inline Value& get_key(Object& obj, std::string key) {
	return get_key<Value>(obj, key);
}
inline Value const& get_key(Object const& obj, std::string key) {
	return get_key<Value>(obj, key);
}

// Specialisation allowing direct conversion to typed (homogeneous) list
template<typename T>
inline std::vector<T> get_key<std::vector<T> >(Object const& obj,
                                               std::string   key) {
	List vals = get_key<List>(obj, key);
	std::vector<T> ret;
	for( List::const_iterator it=vals.begin(); it!=vals.end(); ++it ) {
		Value val = *it;
		if( !val.is<T>() ) {
			throw TypeError("Wrong type for list element: "+key);
		}
		ret.push_back(val.get<T>());
	}
	return ret;
}
template<typename T>
inline std::vector<T> get_key<std::vector<T> >(Object const&  obj,
                                               std::string    key,
                                               std::vector<T> default_value) {
	try {
		return get_key<std::vector<T> >(obj, key);
	}
	catch( std::out_of_range const& ) {
		return default_value;
	}
}
*/

inline Value& get_key(Object& obj, std::string key) {
	Object::iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
inline Value const& get_key(Object const& obj, std::string key) {
	Object::const_iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}

//auto it_b = _rings.insert(std::make_pair(name,
	//                                         std::move(ring_type(alloc))));
	// TODO: What happens if we need to pass >1 args to ring constructor here?
	//auto it_b = _rings.emplace(name, alloc);
	//auto it       = it_b.first;
	//bool inserted = it_b.second;
	//if( !inserted ) {
	//	throw std::invalid_argument("Ring already exists: "+name);
	//}
	//ring_type* ring = &it->second;
	//return ring;

/*
inline Value& lookup_value(Object& obj, std::string key) {
	Object::iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
inline Value const& lookup_value(Object const& obj, std::string key) {
	Object::const_iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
*/

 // TODO: How to automatically communicate these downstream?
	//         Copy/forward params to outputs and then access through inputs?
	//           E.g., RecvUDP.data_output.params.insert( "packet_size_max" );
	//                 RecvUDP.hdr_output.params.insert( "header_size_max" );
	//                 RecvUDP.data_output.params.insert( "shape: [%i]" );
	//         Inherit/forward all params?
	//           Overwrite when updated? E.g., old shape --> new shape

// TODO: Probably not worth using this vs. a traditional loop
		//for( auto p : xrange(_data_input) ) {

// Advance inputs
			++_data_input;
			 ++_hdr_input;
			++_size_input;
			++_addr_input;


	void stats_main();
	size_t      _packet_size_max;
	size_t      _gulp_npacket;
	unsigned    _scatter_factor;
	std::deque<ring_type::WriteBlock> _output_queue;
	std::deque<size_t>                _nbyte_queue;
	typedef std::map<std::string,std::atomic<size_t> > atomic_stats_map;
	typedef std::map<std::string,size_t>                      stats_map;
	atomic_stats_map _stats;

ssize_t Depacketize::decode_packet(value_type const*& packet,
                                   size_type&         packet_size,
                                   addr_type          packet_source) {
	// TODO: Make this much more general using definition parameters
	typedef uint64_t seq_type;
	uint64_t seq = be64toh(*(const seq_type*)packet);
	packet      += sizeof(seq_type);
	packet_size -= sizeof(seq_type);
	return seq;
}


	// Read value at nearest key <= s
	const mapped_type& at(key_type const& s) const {
		std::lock_guard<std::mutex> lock(_mutex);
		// TODO: Track current s for use in checking late commands
		return this->find(s)->second;
	}

void clear_before(key_type const& s) {
		std::lock_guard<std::mutex> lock(_mutex);
		_map.erase(_map.begin(), this->find(s));
	}

#if __cplusplus >= 201103L
	void insert(key_type const& s, mapped_type&& value) {
		std::lock_guard<std::mutex> lock(_mutex);
		this->get(s) = value;
	}
#endif

#if __cplusplus >= 201103L
	void reset(key_type const& s0, mapped_type&& value) {
		std::lock_guard<std::mutex> lock(_mutex);
		_map.clear();
		this->get(s0) = value;
	}
#endif

TODO: Decide which features can be left until later
          E.g., could remove input/output mgmt and just start with
            statically-defined configuration.
            Dynamic configuration is a bit of a rabbit hole

	void set_frame_size(size_t frame_size, size_t buffer_factor=3) {
		_data.set_frame_size(frame_size);
		//_ring->request_size(frame_size*sizeof(value_type),
	}
	void request_buffer_size(size_t nframe) {
		//_ring->request_size(..., nframe*frame_size()*sizeof(value_type));
	}

//RingAccessor(RingBuffer* ring,
	//             size_t      frame_size=1)
	//	: _ring(ring), _data(frame_size), _nframe(0) {}
	//void init(RingBuffer* ring,
	//          size_t      frame_size=1) {
	//	_ring = ring;
	//	_data.set_stride(frame_size);
	//	_nframe = 0;
	//}


// Memory management
enum space_type {
	SPACE_SYSTEM,
	SPACE_CUDA
};
template<class SrcIter, class DstIter>
void copy(SrcIter first, SrcIter last, DstIter result) {
	cuda::independent_stream s;
	cudaError_t ret = cudaMemcpyAsync((void*)result,
	                                  (void const*)first,
	                                  last - first,
	                                  cudaMemcpyDefault, s);
	if( ret != cudaSuccess ) {
		throw std::runtime_error(cudaGetErrorString(ret));
	}
}
template<class SrcIter, class DstIter>
void copy_async(SrcIter first, SrcIter last, DstIter result,
                cudaStream_t s) {
	cudaMemcpyAsync((void*)result, (void const*)first, last - first,
	                cudaMemcpyDefault, s);
}







class RecvUDP {
	void init() {
		// ...
		//_out.set_frame_size(pkt_size_max);
		//_out.set_buffer_factor(3);
		_output.request_buffer_factor(3);
	}
	void main() {
		//_out.open(gulp_npacket); // Reallocates if necessary
		while( running ) {
			RingWriteBlock<char> out_block(_output, pkt_size_max, gulp_npacket);
			size_t npkt = _socket.recv_block(&out_block[0],
			                                 out_block.size_bytes(),
			                                 ...);
			// ...
		}
	}
private:
	//RingWriteBlock<char> _out;
};

/*
  RingReader(RingBuffer* )
  RingStreamWriter(RingBuffer* )
  RingBlockWriter(RingBuffer* )
    RingWriteBlock(RingBlockWriter* )
 */

class Depacketize {
	void init() {
		// ...
		_in.set_frame_size(pkt_size_max);
		//_in.request_buffer_factor(3);
		_in.request_buffer_size(gulp_npacket*3);
		_in.set_nframe(gulp_npacket);
	}
	void main() {
		//_in.open(gulp_npacket); // Reallocates if necessary
		_in.open();
		//for( _in.open(gulp_npacket); !stop_requested(); ++_in ) {
		while( true ) {
			for( int p=0; p<_in.size(); ++p ) {
				char*  pkt_data = &_in[p];
				size_t pkt_size = _sizes[p];
			}
			for( auto& pkt : _in ) {
				// ...
				// pkt is iterator pointing to packet data
				_out_blocks.pop_front();
				// TODO: Need wrapper object for RingBuffer+frame_size that can be passed as first arg here
				//         Wrap block size too?
				//         BlockGenerator
				_out_blocks.emplace_back(_output, gulp_npacket);
				_out_blocks.push_back(_output.open_block());
			}
			// ...
			try {
				++_in;
			}
			catch( RingBuffer.Shutdown ) {
				break;
			}
		}
	}
	void request_stop() {
		_in.shutdown();
		// Sets shutdown flag in this task's reader object, which RingBuffer
		//   checks and throws RingBuffer.Shutdown if set. Also calls
		//   RingBuffer.read/write/realloc_condition.notify_all()
		//   (via RingBuffer.notify_shutdown()).
	}
private:
	RingReadStream<char> _in;
	std::deque<RingWriteBlock<char> > _out_blocks;
};


template<typename T>
class RingAccessor {
public:
	typedef T        value_type;
	typedef T*       pointer;
	typedef T const* const_pointer;
	typedef T&       reference;
	typedef T const& const_reference;
	typedef strided_iterator<T>             iterator;
	typedef strided_iterator<const T> const_iterator;
	typedef std::vector<size_t> shape_type;
	RingAccessor() : _ring(0), _data(), _nframe(0) {}
	void init(RingBuffer* ring) {
		shape_type frame_shape = lookup_list<size_t>(ring->params(), "shape");
		return this->init(ring, frame_shape);
	}
	void init(RingBuffer* ring,
	          size_t      frame_size) {
		shape_type frame_shape({frame_size});
		return this->init(ring, frame_shape);
	}
	void init(RingBuffer* ring,
	          shape_type  frame_shape) {
		_ring = ring;
		_data.set_stride(product_of(frame_shape));
		_nframe = 0;
	}
	void request_size_frames(size_t gulp_size, size_t buffer_size) {
		_ring->request_size(  gulp_size*frame_bytes(),
		                    buffer_size*frame_bytes());
	}
	void request_size_bytes(size_t gulp_size, size_t buffer_size) {
		this->request_size_frames(div_ceil(  gulp_size, frame_bytes()),
		                          div_ceil(buffer_size, frame_bytes()));
	}
	inline frame_idx_type frame0()      const { return _frame0; }
	inline size_t         frame_size()  const { return _data.stride(); }
	inline size_t         frame_bytes() const { return frame_size()*sizeof(value_type); }
	inline size_t         size()        const { return _nframe; }
	inline size_t         size_bytes()  const { return size()*frame_bytes(); }
	inline iterator       begin()       { return _data; }
	inline const_iterator begin() const { return _data; }
	inline iterator       end()         { return _data + _nframe; }
	inline const_iterator end()   const { return _data + _nframe; }
	inline reference       operator[](size_t n)       { return _data[n]; }
	inline const_reference operator[](size_t n) const { return _data[n]; }
	inline RingBuffer*       ring()        { return _ring; }
	inline RingBuffer const* ring() const  { return _ring; }
	// Warning: These values can potentially change immediately after the call
	inline ssize_t           head() const  { return _ring->head(); }
	inline ssize_t           tail() const  { return _ring->tail(); }
protected:
	//inline size_t offset_bytes() const { return frame0()*frame_bytes(); }
	inline size_t byte0() const { return frame0()*frame_bytes(); }
private:
	RingBuffer*      _ring;
	strided_iterator _data;
	frame_idx_type   _frame0;
	size_t           _nframe;
};


	// ** TODO: Store shape explicitly? Make it a proper member of RingBuffer?
	//            Yes. Just finish off StackVector and use that for storage.
	//            Same for dtype (implement/find some type serialisation code).
	//            ** Be careful with potentially different shape definitions
	//                 for readers vs. the ring itself.
	
auto in_data_shape = lookup_list<size_t>(in_data_ring.params(), "shape");
	auto  in_hdr_shape = lookup_list<size_t>( in_hdr_ring.params(), "shape");

auto in_data_ring = get_input_ring("data");
	auto  in_hdr_ring = get_input_ring("headers");


uint64_t PSRDADAReader::readHeader(uint64_t    header_size,
                                   const char* header_in) {
	char utc_start_str[32];
	if( ascii_header_get(header_in, "UTC_START", "%s", utc_start_str) < 0 ) {
		throw std::out_of_range("Missing header entry UTC_START");
	}
	tm utc_start_tm;
	if( !strptime(utc_start_str, "%Y-%m-%d-%H:%M:%S", &utc_start_tm) ) {
		cerr << "UTC_START = " << utc_start_str << endl;
		throw std::invalid_value("Failed to parse UTC_START");
	}
	std::string utc_epoch_str = lookup_string(params(), "utc_epoch",
	                                          "1970-01-01-00:00:00");
	tm utc_epoch_tm;
	strptime(utc_epoch_str.c_str(), "%Y-%m-%dT%H:%M:%S", &utc_epoch_tm);
	// TODO: This will break if a leap second occurs during the observation
	//         Solve by converting UTC to TAI via a table of leap seconds
	int64_t secs_since_epoch_utc = (date2mjds(utc_start_tm) -
	                                date2mjds(utc_epoch_tm));
	
	// ephem_mjds(utc_start_tm);
	// ** UTC->TAI is done by finding <=utc_secs_since_epoch in table
	//      and then adding the resulting TAI-UTC offset.
	//      E.g., the table here:
	//        https://www.ietf.org/timezones/data/leap-seconds.list
	// ** UTC cannot be represented uniquely as seconds-since-epoch
	//      Must use tuple/struct/string instead so that leap secs
	//        can be stored as 23:59:60.
	//      So, to reference an absolute time, either use TAI-secs-since-epoch
	//        or use a UTC tuple/struct/string.
	// utc2tai(
	
	// Note: Must convert UTC->TAI to remove leap seconds and get linear time
	// TODO: Need to look this up at utc_start in a (maintained) table
	//         See http://maia.usno.navy.mil/ser7/tai-utc.dat
	int leap_secs_since_epoch = 36;
	int64_t secs_since_epoch_tai = (secs_since_epoch_utc +
	                                leap_secs_since_epoch);
	double  frame_rate = lookup_float(params(),  "frame_rate");
	int64_t tick = (int64_t)round(secs_since_epoch * frame_rate);
	
	size_t bytes_per_read = _data_output.size_bytes();
	return bytes_per_read;
}

// ** TODO: This seems unnecessary
	//            Could just as easily communicate _tick_start and use
	//              frame0 to keep track of frame numbers downstream.
	//              In fact, that is preferable because it allows
	//                downstream tasks to use their own definition
	//                of a frame.
	//             ** So, how to communicate _tick0 (or utc_start if that's more sensible)?
	//for( auto& tick : time_block ) {
	//	tick = _tick++;
	//}


		//std::cout << lookup_list<std::string>(params(), "gpu_devices")[0] << std::endl;
		/*
		List gpu_devices = lookup_list<Value>(params(), "gpu_devices");
		if( !gpu_devices.empty() ) {
			std::string first_device = gpu_devices[0].to_str();
			int gpu_idx;
			if( first_device.find(":") != std::string::npos ) {
				cudaDeviceGetByPCIBusId(&gpu_idx, first_device);
			}
			else {
				gpu_idx = gpu_devices[0].get<int64_t>();
			}
			cudaSetDevice(gpu_idx);
		}
		*/


	// Full:  tcsp^ bdcsp -> bdct
	// Short: tcs^  bcs   -> bct
	//                    -> tcb
	// sct^ scb  -> tcb
	//// scb^ sct  -> bct


	if( shapeC.size() == 3 ) {
		size_t nbatch = shapeC[1];
		size_t bda = (shapeA[1] != 1) ? shapeA[2] : 0;
		size_t bdb = (shapeB[1] != 1) ? shapeB[2] : 0;
		size_t bdc = shapeC[2];
		std::vector<cuda::independent_stream> streams(nbatch);
		// Batch over middle dim
		
	}
	else {
		// Batch over frames
		throw std::runtime_erro("Frame-batching not yet implemented!");
	}
	
	for( size_t c=0; c<nchan; ++c ) {
		cuComplex const* a_ptr =  in_ptr + c*(nstand*npol);
		cuComplex const* b_ptr =   w_ptr + c*(nstand*npol);
		cuComplex*       c_ptr = out_ptr + c*ntime;
		_cublas.set_stream(streams[c]);
		_cublas.gemm(CUBLAS_OP_T, CUBLAS_OP_N,
		             m, n, k,
		             a_ptr, lda,
		             b_ptr, ldb,
		             c_ptr, ldc);
	}

get_output_ring("headers")->params()["shape"] = List({header_size_max});


template<class Semaphore, class >
class ConditionSemGuard {
	Semaphore& sem;
public:
	template<class Mutex, class CondVar, class Predicate>
	ConditionSemGuard(Mutex&     mtx,
	                  //CondVar&  cv,
	                  Signal&    sig,
	                  Predicate  pred)
	//: _sem(sem) {
		std::unique_lock<Mutex> lock(mtx);
		cv.wait(lock, pred);
		++_sem;
	}
	~ConditionSemGuard() {
		--_sem;
		// TODO: Notify all others who might be waiting on this sem
	}
};


				ssize_t block_idx    = (pkt_dst - block_head) / block_size;
				if( block_idx >= _data_output_blocks.size() ) {
					// The remaining packets will go in as-yet-unopened blocks
					break;
				}
				char* blk_ptr = &_data_output_blocks[block_idx][0];
				ssize_t segment_beg  = std::max(pkt_dst, block_head);
				ssize_t segment_end  = std::min(pkt_dst + pkt_size,
				                                block_head + block_size);
				ssize_t segment_size = segment_end - segment_beg;
				ssize_t pkt_offset   = segment_beg - pkt_dst;
				ssize_t blk_offset   = segment_beg - block_head;
				char const* src_ptr  = pkt_ptr + pkt_offset;
				char*       dst_ptr  = blk_ptr + blk_offset;
				PacketSegment segment(src_ptr, dst_ptr, segment_size);
				segments.push_back(segment);


			bool too_slow = false;
			// Loop through input packets
			size_t pkt_idx;
			for( pkt_idx=0; pkt_idx<_data_input.size(); ++pkt_idx ) {
				const char* pkt_ptr  = &_data_input[pkt_idx];
				const char* pkt_hdr  =  &_hdr_input[pkt_idx];
				size_type   pkt_size =  _size_input[pkt_idx];
				addr_type   pkt_addr =  _addr_input[pkt_idx];
				if( pkt_size == 0 ) {
					// TODO: Does this actually ever happen now?
					continue; // Skip empty entries
				}
				++_stats.at("nrecv");
				_stats.at("nrecv_bytes") += pkt_size;
			
				ssize_t payload_src;  // Byte offset from start of packet data
				ssize_t payload_dst;  // Byte offset from start of output stream
				ssize_t payload_size; // No. bytes comprising payload
				this->decode_packet(pkt_hdr, pkt_size, pkt_addr,
				                    &payload_src, &payload_dst, &payload_size);
				if( payload_size == 0 ) {
					++_stats.at("nignored");
					_stats.at("nignored_bytes") += pkt_size;
					continue;
				}
				pkt_ptr += payload_src;
			
				ssize_t dst_beg = payload_dst;
				ssize_t dst_end = dst_beg + payload_size;
				if( dst_end < _data_output.head() ) {
					++_stats.at("nlate");
					_stats.at("nlate_bytes") += payload_size;
					continue;
				}
			
				// Loop through the packet's bytes, copying to the output
				//   buffer blocks. This is to take care of the packet
				//   spanning output block boundaries.
				//const char* pkt_ptr0 = pkt_ptr;
				ssize_t     dst_byte = std::max(dst_beg, _data_output.head());
				while( dst_byte != dst_end ) {
					this->advance_output_to(dst_byte);
					//std::lock_guard<ExclusionSemaphore> guard(_write_block_sem);
					// Find which block this current byte falls into
					size_t  block_size   = _data_output.size();
					ssize_t block_head   = _data_output.head();
					//size_t  block        = (pkt_byte - block_head) / block_size;
					size_t  block        = (dst_byte - block_head) / block_size;
					// Copy the portion of the packet that overlaps the block
					ssize_t block_beg    = block_head + block*block_size;
					ssize_t block_end    = block_beg + block_size;
					ssize_t overlap_end  = std::min(dst_end, block_end);
					ssize_t overlap_size = overlap_end - dst_byte;
					char*   dst_ptr      = (&_data_output_blocks[block][0] +
					                        (dst_byte - block_beg));
					char*   mask_ptr     = (&_mask_output_blocks[block][0] +
					                        (dst_byte - block_beg));
					//auto const& copy = _data_input.get_copier();
					//auto const& fill = _data_input.get_filler();
					copy(pkt_ptr, pkt_ptr + overlap_size, dst_ptr);
					// Check that the data weren't overwritten prior to copying
					//if( !_data_input.still_valid(pkt_ptr - pkt_ptr0) ) {
					if( !_data_input.still_valid(pkt_idx) ) {
						// Blank-out bad data that were just copied
						//auto const& fill = _data_input.get_filler();
						fill(dst_ptr, dst_ptr + overlap_size, _fill_char,
						     data_space);
						too_slow = true;
						break;
					}
					// Update output mask
					fill(mask_ptr, mask_ptr + overlap_size, 1, mask_space);
				
					_nbyte_blocks[block]        += overlap_size;
					_stats.at("npending_bytes") += overlap_size;
					dst_byte += overlap_size;
					pkt_ptr  += overlap_size;
				}
				if( too_slow ) {
					break;
				}
				else {
					++_stats.at("ngood");
				}
			} // End of packet loop
		
			ssize_t npkt_advance = _data_input.size();
			if( too_slow ) {
				// Skip ahead to the latest data
				ssize_t  cur_pkt = _data_input.frame0();
				ssize_t head_pkt = _data_input.head();
				npkt_advance = head_pkt - cur_pkt;
			
				ssize_t npkt_completed = pkt_idx;
				_stats.at("noverwritten") += npkt_advance - npkt_completed;
			}
		
			try {
				// Advance inputs
				_data_input += npkt_advance;
				_hdr_input += npkt_advance;
				_size_input += npkt_advance;
				_addr_input += npkt_advance;
			}
			catch( RingBuffer::ShutdownError ) {}


TODO: SpanSequence: Use std::map not std::multimap
                    Make query return [lower_bound(start), lower_bound(end))
                      ** PLUS lower_bound(start)-1 if it exists and ends > start.
                    Add erase_before(key_type key) method that erases any entry ending <= key



		// Remove empty entries
		packets.resize(std::remove_if(packets.begin(), packets.end(),
		                              [&](PacketInfo const& pkt) {
			                              return std::get<2>(pkt) == 0;
		                              }) - packets.begin());
		// Sort packets by destination byte (first tuple element)
		std::sort(packets.begin(), packets.end());
		
		// Now generate info on contiguous segments of payload data that can
		//   be copied to output blocks.
		std::atomic_bool too_slow;
		too_slow = false;
		// Tuple contains (src_ptr,block_idx,block_offset,segment_size,pkt_idx)
		typedef std::tuple<char const*,ssize_t,ssize_t,ssize_t,ssize_t> PacketSegment;
		std::vector<PacketSegment> segments;
		size_t npkt    = packets.size();
		// Loop over the sorted packets
		for( size_t sorted_pkt_idx=0; sorted_pkt_idx<npkt; ) {
			// Advance output buffers based on span of packet destinations
			ssize_t first_pkt_dst  = std::get<0>(packets[sorted_pkt_idx]);
			ssize_t  last_pkt_dst  = std::get<0>(packets[npkt-1]);
			ssize_t  last_pkt_size = std::get<2>(packets[npkt-1]);
			this->advance_output_to(first_pkt_dst, last_pkt_dst+last_pkt_size);
			
			// Generate list of packet segments to copy
			segments.clear();
			//for( size_t pkt_idx=npkt_done; pkt_idx<npkt; ++pkt_idx,++npkt_done ) {
			// Loop over processable packets (note break below)
			for( ; sorted_pkt_idx<npkt; ++sorted_pkt_idx ) {
				PacketInfo const& packet = packets[sorted_pkt_idx];
				ssize_t     pkt_dst  = std::get<0>(packet);
				char const* pkt_ptr  = std::get<1>(packet);
				ssize_t     pkt_size = std::get<2>(packet);
				ssize_t     pkt_idx  = std::get<3>(packet);
				if( pkt_size == 0 ) {
					continue; // Skip over empty entries
				}
				ssize_t block_size   = _data_output.size_bytes();
				ssize_t output_head  = _data_output.head();
				
				ssize_t first_blk_idx = (pkt_dst              - output_head) / block_size;
				ssize_t  last_blk_idx = (pkt_dst + pkt_size-1 - output_head) / block_size;
				if( last_blk_idx >= (ssize_t)_data_output_blocks.size() ) {
					// The remaining packets will go in as-yet-unopened blocks
					break;
				}
				// For each output block touched by this packet
				for( ssize_t blk_idx=first_blk_idx; blk_idx!=last_blk_idx; ++blk_idx ) {
					ssize_t blk_head = output_head + blk_idx*block_size;
					//char*   blk_ptr  = &_data_output_blocks[blk_idx][0];
					
					ssize_t segment_beg  = std::max(pkt_dst, blk_head);
					ssize_t segment_end  = std::min(pkt_dst + pkt_size,
					                                blk_head + block_size);
					ssize_t segment_size = segment_end - segment_beg;
					ssize_t pkt_offset   = segment_beg - pkt_dst;
					ssize_t blk_offset   = segment_beg - blk_head;
					char const* src_ptr  = pkt_ptr + pkt_offset;
					//char*       dst_ptr  = blk_ptr + blk_offset;
					//PacketSegment segment(src_ptr, dst_ptr, segment_size);
					PacketSegment segment(src_ptr, blk_idx, blk_offset,
					                      segment_size, pkt_idx);
					segments.push_back(segment);
				}
			}
			// Do the actual copying of payload data
#pragma omp parallel for
			for( size_t seg=0; seg<segments.size(); ++seg ) {
				PacketSegment const& segment = segments[seg];
				char const* __restrict__ src = std::get<0>(segment);
				ssize_t blk_idx = std::get<1>(segment);
				ssize_t blk_off = std::get<2>(segment);
				ssize_t size    = std::get<3>(segment);
				ssize_t pkt_idx = std::get<4>(segment);
				char* __restrict__   blk_ptr = &_data_output_blocks[blk_idx][0] + blk_off;
				char* __restrict__  mask_ptr = &_mask_output_blocks[blk_idx][0] + blk_off;
				// Copy packet payload to destination
				copy(src, src + size, blk_ptr, data_space, data_space);
				// Check that the input data weren't overwritten
				if( _data_input.still_valid(pkt_idx) ) {
					// Update output mask
					fill(mask_ptr, mask_ptr + size, 1, mask_space);
					// Track valid bytes in the block
#pragma omp atomic
					_nbyte_blocks[blk_idx]      += size;
					_stats.at("npending_bytes") += size;
				}
				else {
					// Blank-out bad data that were just copied
					fill(blk_ptr, blk_ptr + size, _fill_char, data_space);
					too_slow = true;
					_stats.at("noverwritten_bytes") += size;
				}
			}
		} // End loop over sorted packets
		
		ssize_t npkt_advance = _data_input.size();
		if( too_slow ) {
			// Skip ahead to the latest data
			ssize_t  cur_pkt = _data_input.frame0();
			ssize_t head_pkt = _data_input.head();
			npkt_advance = head_pkt - cur_pkt;
		}
		try {
			// Advance inputs
			_data_input += npkt_advance;
			 _hdr_input += npkt_advance;
			_size_input += npkt_advance;
			_addr_input += npkt_advance;
		}
		catch( RingBuffer::ShutdownError ) {}
		
	} // End of main loop
}
void Depacketize::advance_output_to(ssize_t beg, ssize_t end) {
	ssize_t block_size = _data_output.size_bytes();
	space_type data_space = _data_output.space();
	space_type mask_space = _mask_output.space();
	//while( dst_byte >= _data_output.reserve_head() ) {
	while( _data_output.reserve_head()      <  end &&
	       _data_output.head() + block_size <= beg ) {
		// Need to open another block
		if( _data_output_blocks.size() == _scatter_factor ) {
			// Too many blocks already open, need to close one at the back
			_data_output_blocks.pop_front();
			_mask_output_blocks.pop_front();
			size_t nbyte_pending = _nbyte_blocks.front();
			_stats.at("npending_bytes")  -= nbyte_pending;
			_stats.at("ngood_bytes")     += nbyte_pending;
			_stats.at("nmissing_bytes")  += (_data_output.size() -
			                                 nbyte_pending);
			_nbyte_blocks.pop_front();
		}
		_data_output_blocks.push_back(_data_output.open());
		_mask_output_blocks.push_back(_mask_output.open());
		_nbyte_blocks.push_back(0);
		// Blank out the new output block
		// Note: There is no easy way to do this on a per-missing-byte,
		//         basis later, so we just do the whole block now.
		char* __restrict__  block_ptr = &_data_output_blocks.back()[0];
		char* __restrict__   mask_ptr = &_mask_output_blocks.back()[0];
		size_t block_size = _data_output.size();
		// TODO: Parallelise these(?)
		fill(block_ptr, block_ptr + block_size, _fill_char, data_space);
		fill( mask_ptr,  mask_ptr + block_size,          0, mask_space);
	}
}

const_iterator upper_bound_end(key_type point) const {
		const_iterator ret = this->lower_bound(point);
		// Must check if the previous element ends > point
		if( ret.first != this->begin() &&
		    ret.first != point ) {
			const_iterator prev = ret.first;
			--prev;
			key_type prev_end = prev->first + prev->second.first;
			if( prev_end > point ) {
				ret.first = prev;
			}
		}
		return ret;
	}


	// Note: Doing this here ensures no multi-threaded modifications to the map
	//_stats["nrecv_bytes"] = 0;
	//_stats["nrecv"]       = 0;
	//_stats["ndrop"]       = 0;


		/*Object stats_msg;
		for( auto const& stat : stats ) {
			stats_msg[stat.first] = Value(stat.second);
		}
		*/


		//Object stats_msg = make_Object(stats);
		//this->broadcast("stats", stats_msg);

while( true ) {
		//auto& pkt_block = _data_output.open_block();
		//auto& pkt_sizes = _size_output.open_block();
		//auto& pkt_addrs = _addr_output.open_block();
		//auto& pkt_hdrs  = _hdr_output.open_block();
		auto pkt_block = _data_output.open();
		auto pkt_sizes = _size_output.open();
		auto pkt_addrs = _addr_output.open();
		auto pkt_hdrs  = _hdr_output.open();
		size_t npkt = _socket.recv_block(&pkt_block[0],
		                                 pkt_block.size_bytes(),
		                                 pkt_block.frame_size(),
		                                 &pkt_sizes[0],
		                                 &pkt_addrs[0]);
		
		/*
		// ** TODO: New recv call API
		_socket.recv_block(npacket,
		                   &pkt_hdrs[0], 0, &header_size,
		                   &pkt_block[0], 0, pkt_block.frame_size(),
		                   &pkt_sizes[0],
		                   &pkt_addrs[0]);
		*/
		
		// Note: Shutdown is enacted via _socket.shutdown(),
		//         which will unblock the above recv call.
		if( this->shutdown_requested() ) {
			break;
		}
		stats.at("nrecv_bytes") += _socket.get_recv_size();
		stats.at("nrecv")       += npkt;
		stats.at("ndrop")       += _socket.get_drop_count();
		
		//nrecv_bytes += _socket.get_recv_size();
		
		// Extract headers from received packets
		for( size_t p=0; p<npkt; ++p ) {
			::memcpy(&pkt_hdrs[p], &pkt_block[p], pkt_hdrs.frame_size());
		}
		// Fill in possibly-unused end of block
		for( size_t p=npkt; p<pkt_block.size(); ++p ) {
			pkt_sizes[p] = 0;
			// Note: Ends of data, address and header blocks are left
			//         uninitialised, so size array should always be checked.
		}
		
		this->broadcast("stats", make_Object(stats));
	}


		//nrecv_bytes += _socket.get_recv_size();


		// Extract headers from received packets
		for( size_t p=0; p<npkt; ++p ) {
			::memcpy(&pkt_hdrs[p], &pkt_block[p], pkt_hdrs.frame_size());
		}

		//auto& pkt_block = _data_output.open_block();
		//auto& pkt_sizes = _size_output.open_block();
		//auto& pkt_addrs = _addr_output.open_block();
		//auto& pkt_hdrs  = _hdr_output.open_block();

/*
class StatsThread : public TaskMonitor {
public:
	StatsThread(RecvUDP const* task, double interval)
		: TaskMonitor(task, interval),
		  _tot_stats(task->_stats.begin(), task->_stats.end()),
		  _data_output(task->_data_output),
		  _stats(task->_stats) {
		printf("%7s %7s %7s %8s %11s %11s %11s %11s\n",
		       "Gb/s", "NPKT/s", "NDROP/s", "BYTES", "NPKT", "NDROP", "TAIL", "HEAD");
	}
	void update(double dt) {
		task_type::stats_map new_stats;
		for( auto& kv : _stats ) {
			auto new_val = kv.second.exchange(0);
			 new_stats[kv.first]  = new_val;
			_tot_stats[kv.first] += new_val;
		}
		//const double& dt = _interval_secs; // Shorter name
		printf("%7.3f %7.0f %7.0f %8.2e %11lu %11lu %11lu %11lu\n",
		       new_stats["nrecv_bytes"]/1e9*8/dt,
		       new_stats["nrecv"]/dt,
		       new_stats["ndrop"]/dt,
		       (double)_tot_stats["nrecv_bytes"],
		       _tot_stats["nrecv"],
		       _tot_stats["ndrop"],
		       _data_output.tail(), _data_output.head());
	}
private:
	typedef RecvUDP task_type;
	task_type::stats_map         _tot_stats;
	//task_type::ring_type::BlockWriter<char> const& _data_output;
	RingWriter<char> const&      _data_output;
	task_type::atomic_stats_map& _stats;
	//double                     _interval_secs;
};
*/

//StatsThread stats_thread(this, lookup_float(params(), "stats_interval"));


class ZMQBuffer : public std::streambuf {
public:
	enum {
		MIN_BUF_SIZE = 256
	};
protected:
	// Write single character
	inline virtual int overflow(int c) {
		size_t new_size = _size + 1;
		// Amortized reallocations
		if( _buf.size() < new_size ) {
			size_t capacity = std::max(new_size, _buf.size()*2);
			capacity = std::max(capacity, MIN_BUF_SIZE);
			zmq::message_t new_buf(capacity);
			::memcpy(new_buf.data(), _buf.data(), _size);
			_buf.move(&new_buf);
		}
		// Write to the buffer
		((char*)_buf.data())[_size] = c;
		_size = new_size;
	}
	// Synchronize internal buffer with external destination
	virtual int sync() {
		_socket->send(_buf);
		//this->str(""); // Clear buffer
		_size = 0;
		return 0;
	}
public:
	ZMQBuffer(ZMQSocket* sock) : _buf(), _sock(sock), _size(0) {}
private:
	zmq::message_t _buf;
	ZMQSocket*     _sock;
	size_t         _size;
};

/*
	void vlog(std::string topic, std::string msg, va_list args) {
		// Apply C-style formatting
		int ret = vsnprintf(&_log_buffer[0], _log_buffer.capacity(),
		                    msg.c_str(), args);
		if( (size_t)ret >= _buffer.capacity() ) {
			_log_buffer.resize(ret+1); // Note: +1 for NULL terminator
			ret = vsnprintf(&_log_buffer[0], _log_buffer.capacity(),
			                msg.c_str(), args);
		}
		if( ret < 0 ) {
			// TODO: How to handle encoding error?
		}
	}
	void log(std::string topic, std::string msg=, ...) {
		va_list va;
		va_start(va, msg);
		std::ostream& s = this->vlog(level, msg, va);
		va_end(va);
		return s;
	}*/


class Logger {
public:
	typedef std::function<void(const char*)> callback_type;
	Logger(callback_type callback) : _callback(callback) {}
	void log(std::string topic, std::string msg, ...) {
		std::lock_guard<std::mutex> lock(_mutex);
		va_list va;
		va_start(va, msg);
		int ret = vsnprintf(&_buffer[0], _buffer.capacity(),
		                    msg.c_str(), args);
		if( (size_t)ret >= _buffer.capacity() ) {
			_buffer.resize(ret+1); // Note: +1 for NULL terminator
			ret = vsnprintf(&_buffer[0], _buffer.capacity(),
			                msg.c_str(), args);
		}
		va_end(va);
		if( ret < 0 ) {
			// TODO: How to handle encoding error?
		}
		_callback(&_buffer[0]);
	}
private:
	std::vector<char> _buffer;
	std::mutex        _mutex;
	callback_type     _callback;
};

class TaskLogBuf : public std::stringbuf {
protected:
	virtual int sync() {
		_task->broadcast(
		_socket->send(this->str(), true, _flags);
		this->str(""); // Clear buffer
		return 0;
	}
public:
	TaskLogBuf(Task* task) : _task(task) {}
private:
	Task* _task;
};
class TaskLogStream : public std::ostream {
public:
	TaskLogStream(Task* task) : _task(task) {}
	
private:
	
	TaskLogBuf _buf;
	Task*      _task;
};


template<typename V, typename T, typename ... Types>
struct tuple_of {
	typedef decltype(std::tuple_catstd::tuple< V<T> >
};

template<typename ... Types>
struct Foo {
private:
	typedef typename tuple_of<std::vector, types...>::type vector_tuple;
	vector_tuple _values;
};


	// Advance inputs
	ssize_t npkt_advance = _data_input.size();
	// Check that input wasn't overwritten during processing
	if( !_data_input.still_valid() ) {
		ssize_t blk_size = _data_output.size_bytes();
		_stats.at("noverwritten_bytes") += blk_size;
		// TODO: Fill in overwritten blocks
		// TODO: Handle this!
			
		// Skip ahead to the latest data
		ssize_t  cur_pkt = _data_input.frame0();
		ssize_t head_pkt = _data_input.head();
		npkt_advance = head_pkt - cur_pkt;
	}
	try {
		// Advance inputs
		_data_input += npkt_advance;
		_hdr_input += npkt_advance;
		_size_input += npkt_advance;
		_addr_input += npkt_advance;
	}
	catch( RingBuffer::ShutdownError ) {}


	//ring_type::Reader<char>      _data_input;
	//ring_type::Reader<size_type> _size_input;
	//ring_type::Reader<addr_type> _addr_input;
	//ring_type::Reader<char>       _hdr_input;

/*
class StatsThread : public TaskMonitor {
public:
	StatsThread(Depacketize const* task, double interval)
		: TaskMonitor(task, interval),
		  _tot_stats(task->_stats.begin(), task->_stats.end()),
		  _data_output(task->_data_output),
		  _stats(task->_stats) {
		printf("%7s %7s %7s %7s %7s %7s "
		       "%8s %8s %8s %11s %11s\n",
		       "RECV", "GOOD", "MISSING", "NRECV", "NIGNORE", "NLATE",
		       "OVRWRTN", "PENDING", "MISSING", "TAIL", "HEAD");
		printf("%7s %7s %7s %7s %7s %7s "
		       "%8s %8s %8s %11s %11s\n",
		       "Gb/s", "Gb/s", "Bytes/s", "pkt/s", "pkt/s",   "pkt/s",
		       "Bytes",   "Bytes",   "Bytes",   "Byte", "Byte");
	}
	void update(double dt) {
		task_type::stats_map new_stats;
		for( auto& kv : _stats ) {
			auto new_val = kv.second.exchange(0);
			 new_stats[kv.first]  = new_val;
			_tot_stats[kv.first] += new_val;
		}
		printf("%7.3f %7.3f %8.2e %7.0f %7.0f %7.0f "
		       "%8.2e %8.2e %8.2e %11li %11li\n",
		       new_stats["nrecv_bytes"]/1e9*8/dt,
		       new_stats["ngood_bytes"]/1e9*8/dt,
		       (double)new_stats["nmissing_bytes"]/dt,
		       new_stats["nrecv"]/dt,
		       new_stats["nignore"]/dt,
		       new_stats["nlate"]/dt,
		       (double)_tot_stats["noverwritten_bytes"],
		       //(double)_tot_stats["noverwritten"],
		       (double)_tot_stats["npending_bytes"],
		       (double)_tot_stats["nmissing_bytes"],
		       _data_output.tail(),
		       _data_output.head());
	}
private:
	typedef Depacketize task_type;
	task_type::stats_map         _tot_stats;
	RingWriter<char> const&      _data_output;
	task_type::atomic_stats_map& _stats;
};
*/


class TaskMonitor {
public:
	TaskMonitor(Task const* task, double interval)
		: _task(task), _interval_ns((long long)(interval*1e9)),
		  _thread(&TaskMonitor::main, this) {}
	~TaskMonitor() {
		// Note: Parent task must be shutdown for this to return
		_thread.join();
	}
	inline double interval() const { return _interval_ns.count() / 1e9; }
	inline void   set_interval(double interval) {
		_interval_ns = std::chrono::nanoseconds((long long)(interval*1e9));
	}
protected:
	virtual void update(double elapsed) = 0;
	Task const* task() { return _task; }
private:
	void main() {
		using std::chrono::steady_clock;
		auto last_time = steady_clock::now();
		_task->sleep_until(last_time + _interval_ns);
		//last_time += _interval_ns;
		while( !_task->shutdown_requested() ) {
			double elapsed = (steady_clock::now() - last_time).count() / 1e9;
			this->update(elapsed);
			last_time += _interval_ns;
			_task->sleep_until(last_time + _interval_ns);
			//last_time += _interval_ns;
		}
	}
	Task const*              _task;
	std::chrono::nanoseconds _interval_ns;
	std::thread              _thread;
};

/*
		//this->create_output("
		struct call_create_output {
			Task* task;
			call_create_output(Task* task_) : task(task_) {}
			template<typename T>
			void operator()(T& output) const {
				task->create_output
			}
		};
		*/


void SendUDP::init() {
	//_data_input.init(get_input_ring("data"));
	// _hdr_input.init(get_input_ring("headers"));
	//_size_input.init(get_input_ring("sizes"));
	//_addr_input.init(get_input_ring("destinations"));
	/*
	size_t gulp_size_min = lookup_integer(params(), "gulp_size",
	                                      DEFAULT_GULP_SIZE_MIN);
	size_t buffer_factor = lookup_integer(params(), "buffer_factor",
	                                      DEFAULT_BUFFER_FACTOR);
	
	_data_input.request_size_bytes(gulp_size_min, buffer_factor*gulp_size_min);
	size_t nframe_in = _data_input.size();
	 _hdr_input.request_size_frames(nframe_in, buffer_factor*nframe_in);
	_size_input.request_size_frames(nframe_in, buffer_factor*nframe_in);
	_addr_input.request_size_frames(nframe_in, buffer_factor*nframe_in);
	*/
	// TODO: Bind to CPU cores(?)
}
void SendUDP::open() {
	/*
	bool guarantee_reads = lookup_bool(params(), "guarantee_reads", false);
	_data_input.open(guarantee_reads);
	_hdr_input.open(guarantee_reads);
	_size_input.open(guarantee_reads);
	_addr_input.open(guarantee_reads);
	*/
}


void SendUDP::still_valid() {
	/*
	return (_data_input.still_valid() &&
	        _hdr_input.still_valid()  &&
	        _addr_input.still_valid() &&
	        _size_input.still_valid());
	*/
}
void SendUDP::advance() {
	/*
	++_data_input;
	++_hdr_input;
	++_size_input;
	++_addr_input;
	*/
}

//RingReader<char>      _data_input;
	//RingReader<char>      _hdr_input;
	//RingReader<size_type> _size_input;
	//RingReader<addr_type> _addr_input;


template<typename InputTypes,
         typename OutputTypes=std::tuple<>,
         bool HAVE_INPUTS=(sizeof...(InputTypes)),
         bool HAVE_OUTPUTS=(sizeof...(OutputTypes))>
class ConsumerTask : ConsumerTaskBase {
	
}


		size_t buffer_factor = lookup_integer(params(), "buffer_factor",
		                                      DEFAULT_BUFFER_FACTOR);
		size_t nframe = 0;
		bool have_inputs  = (std::tuple_size< inputs_type>::value > 0);
		bool have_outputs = (std::tuple_size<outputs_type>::value > 0);
		if( have_inputs ) {
			auto& primary_input = std::get<0>(_inputs);
			if( params().count("gulp_nframe") ) {
				// Gulps specified exactly in no. frames
				size_t gulp_nframe = lookup_integer(params(), "gulp_nframe");
				primary_input.request_size_frames(gulp_nframe,
				                                  gulp_nframe*buffer_factor);
			}
			else {
				// Gulps specified by memory size upper limit
				size_t gulp_size_min = lookup_integer(params(), "gulp_size",
				                                      DEFAULT_GULP_SIZE_MIN);
				primary_input.request_size_bytes(gulp_size_min,
				                                 gulp_size_min*buffer_factor);
			}
			nframe = primary_input.size();
		}
		else if( have_outputs ) {
			auto& primary_output = std::get<0>(_outputs);
			if( params().count("gulp_nframe") ) {
				// Gulps specified exactly in no. frames
				size_t gulp_nframe = lookup_integer(params(), "gulp_nframe");
				primary_output.request_size_frames(gulp_nframe,
				                                   gulp_nframe*buffer_factor);
			}
			else {
				// Gulps specified by memory size upper limit
				size_t gulp_size_min = lookup_integer(params(), "gulp_size",
				                                      DEFAULT_GULP_SIZE_MIN);
				primary_output.request_size_bytes(gulp_size_min,
				                                  gulp_size_min*buffer_factor);
			}
			nframe = primary_output.size();
		}
		// ---

template<bool HAVE_INPUTS =(std::tuple_size<inputs_type>::value),
	         bool HAVE_OUTPUTS=(std::tuple_size<outputs_type>::value)>
	// No inputs or outputs
	size_t init_primary(size_t buffer_factor) { return 0; }
	template<> // Buffer sizes defined by primary input
	size_t init_primary<true,false>(size_t buffer_factor) {
		auto& primary_input = std::get<0>(_inputs);
		if( params().count("gulp_nframe") ) {
			// Gulps specified exactly in no. frames
			size_t gulp_nframe = lookup_integer(params(), "gulp_nframe");
			primary_input.request_size_frames(gulp_nframe,
			                                  gulp_nframe*buffer_factor);
		}
		else {
			// Gulps specified by memory size upper limit
			size_t gulp_size_min = lookup_integer(params(), "gulp_size",
			                                      DEFAULT_GULP_SIZE_MIN);
			primary_input.request_size_bytes(gulp_size_min,
			                                 gulp_size_min*buffer_factor);
		}
		return primary_input.size();
	}
	template<> // Buffer sizes defined by primary input
	size_t init_primary<true,true>(size_t buffer_factor) {
		return init_primary<true,false>(buffer_factor);
	}
	template<> // Buffer sizes defined by primary output
	size_t init_primary<false,true>(size_t buffer_factor) {
		auto& primary_output = std::get<0>(_outputs);
		if( params().count("gulp_nframe") ) {
			// Gulps specified exactly in no. frames
			size_t gulp_nframe = lookup_integer(params(), "gulp_nframe");
			primary_output.request_size_frames(gulp_nframe,
			                                   gulp_nframe*buffer_factor);
		}
		else {
			// Gulps specified by memory size upper limit
			size_t gulp_size_min = lookup_integer(params(), "gulp_size",
			                                      DEFAULT_GULP_SIZE_MIN);
			primary_output.request_size_bytes(gulp_size_min,
			                                  gulp_size_min*buffer_factor);
		}
		return primary_output.size();
	}

if( header_size > payload_size_max ) {
		throw std::invalid_argument("Header size must be <= "
		                            "max payload size");
	}


// TODO: Can use CRTP to automate even more of this?
//         E.g., create() function, pipeline+definition params, inputs&outputs
//         ACTUALLY, doing it statically would be painful (tuple metaprogramming sucks)
//           Could do it dynamically instead using char elements for everything
//             and having the subclass cast to their desired type when needed?
//             Could possibly reduce subclass effort to just implementing process()
class ConsumerTask : public Task {
public:
	typedef ssize_t cycle_type;
	ConsumerTask(Pipeline*     pipeline,
	             const Object* definition)
		: Task(pipeline, definition) {
		//// Default to numbering cycles from 0 and starting w/ the task enabled
		//this->reset_cycles(0, true);
	}
	//void enable(cycle_type cycle)  { _enabled_sequence.insert(cycle, true); }
	//void enable()                  { _enabled_sequence.insert(true); }
	//void disable(cycle_type cycle) { _enabled_sequence.insert(cycle, false); }
	//void disable()                 { _enabled_sequence.insert(false); }
	//// Note: Subclass must call this, else main will throw an exception
	//void reset_cycles(cycle_type cycle0, bool enable=false) {
	//	_enabled_sequence.reset(cycle0, enable);
	//}
protected:
	// Subclasses implement these
	// TODO: Add _inputs suffix to each?
	virtual void    open() = 0; // Open input buffers
	virtual void process() = 0; // Process input buffers
	virtual bool still_valid() = 0; // Input buffers still valid
	virtual void advance() = 0; // Advance input buffers
	virtual void   close() {} // TODO: Important? Destructor does the work?
	// Called in 'advance' if input buffers were overwritten during processing
	virtual void overwritten() {
		this->log_error("Input data overwritten");
		throw std::out_of_range("Input data overwritten");
	}
	
	virtual void main() {
		try {
			this->open();
		}
		catch( RingBuffer::ShutdownError ) {}
		while( !this->shutdown_requested() ) {
			{
				ScopedTracer trace(this->profiler(), "process");
				//if( _enabled_sequence.pop() ) {
				this->process();
				//}
			}
			{
				ScopedTracer trace(this->profiler(), "advance");
				if( this->still_valid() ) {
					try {
						this->advance();
					}
					catch( RingBuffer::ShutdownError ) {}
					//catch( std::out_of_range ) {
						//this->overwritten();
					//	throw std::runtime_error("Unexpected overwrite condition (check code!)");
					//}
				}
				else {
					this->overwritten();
				}
			}
			this->broadcast("perf", this->profiler().export_object());
			this->profiler().clear();
		}
		this->close();
	}
private:
	//sequence_map<cycle_type, bool> _enabled_sequence;
};

/*
		  // ** TODO: Find a way to do this!
		for( auto const& input_name : this->get_input_names() ) {
			// ** TODO: How to do this?
			auto& input = this->get_input(input_name);
			input.shutdown();
		}
		*/


