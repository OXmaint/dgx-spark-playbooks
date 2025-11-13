# """Camera viewer utility

# Usage examples:
# 	python3 camera.py --source rtsp://user:pass@192.168.1.100:554/stream1
# 	python3 camera.py --source /dev/video0 --width 1280 --height 720 --display
# 	python3 camera.py --source http://192.168.1.100:8080/video --output out.mp4

# Features:
#  - Accepts RTSP, HTTP MJPEG, or local device paths (e.g. /dev/video0)
#  - Attempts automatic reconnection on failure with exponential backoff
#  - Optional recording to a file using OpenCV VideoWriter
#  - Shows FPS overlay, frame count and a simple key interface

# Dependencies:
#  - Python 3.8+
#  - opencv-python (pip install opencv-python)

# This file is intentionally small and dependency-light so you can run it on
# headless systems (set --display off). It uses OpenCV's VideoCapture which
# works with most NVR / RTSP cameras.
# """

# from __future__ import annotations

# import argparse
# import time
# import sys
# from typing import Optional
# import socket


# def _get_cv2():
# 	"""Lazy import of cv2 with helpful error message if missing."""
# 	try:
# 		import cv2
# 		return cv2
# 	except Exception:
# 		print("Error: OpenCV (cv2) is required. Install with: pip install opencv-python")
# 		raise


# def open_capture(source: str, width: Optional[int] = None, height: Optional[int] = None):
# 	cv2 = _get_cv2()
# 	cap = cv2.VideoCapture(source)
# 	if width:
# 		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# 	if height:
# 		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# 	return cap


# def make_writer(filename: str, fourcc: str, fps: float, width: int, height: int):
# 	cv2 = _get_cv2()
# 	fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
# 	writer = cv2.VideoWriter(filename, fourcc_code, max(0.1, fps), (int(width), int(height)))
# 	return writer


# def discover_ssdp(timeout: float = 3.0):
# 	"""Perform a simple SSDP M-SEARCH and return parsed responses.

# 	Returns a list of dicts with headers and the sender address under keys '_addr' (ip,port)
# 	and '_raw' for the raw response.
# 	"""
# 	MSEARCH = '\r\n'.join([
# 		'M-SEARCH * HTTP/1.1',
# 		'HOST: 239.255.255.250:1900',
# 		'MAN: "ssdp:discover"',
# 		'MX: 2',
# 		'ST: ssdp:all',
# 		'',
# 		''
# 	])

# 	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
# 	sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
# 	sock.settimeout(timeout)

# 	try:
# 		sock.sendto(MSEARCH.encode('utf-8'), ('239.255.255.250', 1900))
# 	except Exception as e:
# 		print('SSDP send failed:', e)
# 		sock.close()
# 		return []

# 	responses = []
# 	start = time.time()
# 	while True:
# 		try:
# 			data, addr = sock.recvfrom(65536)
# 		except socket.timeout:
# 			break
# 		except Exception:
# 			break

# 		text = data.decode('utf-8', errors='ignore')
# 		headers = {}
# 		for line in text.split('\r\n'):
# 			if ':' in line:
# 				k, v = line.split(':', 1)
# 				headers[k.strip().upper()] = v.strip()
# 		headers['_raw'] = text
# 		headers['_addr'] = addr
# 		responses.append(headers)

# 		if time.time() - start > timeout:
# 			break

# 	sock.close()
# 	return responses


# def try_tcp_open(ip: str, port: int, timeout: float = 0.5) -> bool:
# 	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 	s.settimeout(timeout)
# 	try:
# 		s.connect((ip, port))
# 		s.close()
# 		return True
# 	except Exception:
# 		return False


# def report_discovery(results):
# 	"""Print discovery results and suggest likely camera URLs."""
# 	if not results:
# 		print('No SSDP/UPnP responses found.')
# 		return

# 	seen = {}
# 	for r in results:
# 		addr = r.get('_addr')
# 		if not addr:
# 			continue
# 		ip = addr[0]
# 		if ip not in seen:
# 			seen[ip] = []
# 		seen[ip].append(r)

# 	print(f'Found {len(seen)} unique IP(s) via SSDP:')
# 	for ip, entries in seen.items():
# 		print('\n- IP:', ip)
# 		for e in entries:
# 			st = e.get('ST') or e.get('NT') or ''
# 			usn = e.get('USN', '')
# 			location = e.get('LOCATION', '')
# 			server = e.get('SERVER', '')
# 			print(f'  ST={st} USN={usn}')
# 			if location:
# 				print(f'  LOCATION: {location}')

# 		# probe common camera ports
# 		ports = [(554, 'rtsp'), (80, 'http'), (8080, 'http')]
# 		for port, proto in ports:
# 			openp = try_tcp_open(ip, port, timeout=0.4)
# 			if openp:
# 				if proto == 'rtsp':
# 					print(f'  Port {port}/tcp open -> possible RTSP stream: rtsp://{ip}/')
# 				else:
# 					print(f'  Port {port}/tcp open -> possible HTTP stream: http://{ip}/')

# 	print('\nNotes: SSDP/UPnP discovery finds devices that advertise on the local network.\nYou may need to use camera/NVR credentials or consult the device manual for the exact RTSP path.')


# def get_local_network_cidr() -> Optional[str]:
# 	"""Try to infer a reasonable local /24 network CIDR from the default route.

# 	This function opens a UDP socket to a public IP to get the local interface IP,
# 	then assumes a /24 network. If that fails, returns None.
# 	"""
# 	try:
# 		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 		# doesn't actually send packets
# 		s.connect(('8.8.8.8', 80))
# 		ip = s.getsockname()[0]
# 		s.close()
# 		parts = ip.split('.')
# 		if len(parts) == 4:
# 			cidr = '.'.join(parts[:3]) + '.0/24'
# 			return cidr
# 	except Exception:
# 		return None


# def iter_hosts_from_cidr(cidr: str):
# 	import ipaddress
# 	net = ipaddress.ip_network(cidr, strict=False)
# 	for host in net.hosts():
# 		yield str(host)


# def tcp_probe_hosts(hosts, ports=(554, 80, 8080), timeout=0.4, workers=200):
# 	"""Probe a list of hosts for open TCP ports concurrently.

# 	Returns a dict: {ip: [open_ports]}
# 	"""
# 	import concurrent.futures

# 	def probe_one(args):
# 		ip, port = args
# 		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 		s.settimeout(timeout)
# 		try:
# 			s.connect((ip, port))
# 			s.close()
# 			return (ip, port, True)
# 		except Exception:
# 			return (ip, port, False)

# 	tasks = []
# 	for ip in hosts:
# 		for p in ports:
# 			tasks.append((ip, p))

# 	results = {}
# 	with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks) or 1)) as ex:
# 		for ip, port, ok in ex.map(probe_one, tasks):
# 			if ok:
# 				results.setdefault(ip, []).append(port)

# 	return results


# def report_scan(scan_results):
# 	if not scan_results:
# 		print('No responsive hosts found on probed ports.')
# 		return

# 	print(f'Found {len(scan_results)} responsive host(s):')
# 	for ip, ports in scan_results.items():
# 		print(f' - {ip} open ports: {ports}')
# 		if 554 in ports:
# 			print(f'    Suggestion: RTSP -> rtsp://{ip}/')
# 		if 80 in ports or 8080 in ports:
# 			print(f'    Suggestion: HTTP/UI -> http://{ip}/')


# COMMON_NVR_RTSP_PATHS = [
# 	'/Streaming/Channels/{ch}01',
# 	'/Streaming/Channels/{ch}02',
# 	'/cam/realmonitor?channel={ch}&subtype=0',
# 	'/h264Preview_0{ch}_main',
# 	'/h264Preview_0{ch}_sub',
# 	'/live.sdp',
# 	'/live/ch{ch}',
# 	'/videoMain',
# 	'/videoSub',
# 	'/media.smp',
# 	'/onvif/media_service'
# ]


# def _try_open_rtsp(url: str, timeout: float = 3.0) -> bool:
# 	"""Try to open an RTSP URL and read one frame using OpenCV within timeout.

# 	Returns True if a frame was captured, False otherwise.
# 	"""
# 	cv2 = _get_cv2()

# 	def attempt():
# 		try:
# 			cap = cv2.VideoCapture(url)
# 			if not cap or not cap.isOpened():
# 				try:
# 					cap.release()
# 				except Exception:
# 					pass
# 				return False

# 			# try a few reads
# 			for _ in range(3):
# 				ret, frame = cap.read()
# 				if ret and frame is not None:
# 					cap.release()
# 					return True
# 				time.sleep(0.3)

# 			try:
# 				cap.release()
# 			except Exception:
# 				pass
# 			return False
# 		except Exception:
# 			return False

# 	import concurrent.futures
# 	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
# 		fut = ex.submit(attempt)
# 		try:
# 			return fut.result(timeout=timeout)
# 		except concurrent.futures.TimeoutError:
# 			return False


# def probe_nvr_hosts(hosts, user: str = '', passwd: str = '', timeout: float = 3.0, workers: int = 10):
# 	"""Probe NVR hosts with common RTSP paths concurrently.

# 	Returns dict {host: [working_urls]}
# 	"""
# 	import concurrent.futures

# 	def build_urls(host):
# 		userpass = ''
# 		if user:
# 			# include credentials if provided
# 			userpass = f'{user}:{passwd}@'
# 		urls = []
# 		for ch in range(1, 9):
# 			ch_s = f'{ch}'
# 			for p in COMMON_NVR_RTSP_PATHS:
# 				path = p.format(ch=ch, ch_s=ch_s)
# 				# normalize leading slash
# 				if path.startswith('/'):
# 					url = f'rtsp://{userpass}{host}:554{path}'
# 				else:
# 					url = f'rtsp://{userpass}{host}:554/{path}'
# 				urls.append(url)
# 		# also add some generic non-channel paths
# 		urls.append(f'rtsp://{userpass}{host}:554/')
# 		urls.append(f'rtsp://{userpass}{host}:554/live.sdp')
# 		return urls

# 	all_tasks = []
# 	host_urls = {}
# 	for h in hosts:
# 		host_urls[h] = build_urls(h)
# 		for u in host_urls[h]:
# 			all_tasks.append((h, u))

# 	results = {}
# 	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
# 		future_to_task = {ex.submit(_try_open_rtsp, u, timeout): (h, u) for (h, u) in all_tasks}
# 		for fut in concurrent.futures.as_completed(future_to_task):
# 			h, u = future_to_task[fut]
# 			try:
# 				ok = fut.result()
# 			except Exception:
# 				ok = False
# 			if ok:
# 				results.setdefault(h, []).append(u)

# 	return results




# def run_viewer(source: str, display: bool = True, output: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None, reconnect_delay: float = 1.0):
# 	"""Main loop: open capture, show frames, optionally write to file.

# 	reconnect_delay: initial reconnect delay in seconds; will exponentially backoff up to 16s.
# 	"""
# 	backoff = reconnect_delay
# 	cap = open_capture(source, width, height)

# 	writer = None
# 	if output:
# 		# We'll set writer once we have first valid frame and known size/fps
# 		writer = None

# 	frame_count = 0
# 	last_fps_ts = time.time()
# 	fps_count = 0
# 	measured_fps = 0.0

# 	try:
# 		while True:
# 			if not cap.isOpened():
# 				# Try to (re)open
# 				print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Capture not open. Reconnecting to {source}...")
# 				cap.release()
# 				time.sleep(backoff)
# 				cap = open_capture(source, width, height)
# 				backoff = min(backoff * 2, 16.0)
# 				continue

# 			ret, frame = cap.read()
# 			if not ret or frame is None:
# 				# Lost connection / no frame; close and retry
# 				print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Frame read failed. Will reconnect in {backoff}s.")
# 				cap.release()
# 				time.sleep(backoff)
# 				cap = open_capture(source, width, height)
# 				backoff = min(backoff * 2, 16.0)
# 				continue

# 			# Reset backoff on successful read
# 			backoff = reconnect_delay

# 			frame_count += 1
# 			fps_count += 1

# 			# Measure FPS once per second
# 			now = time.time()
# 			if now - last_fps_ts >= 1.0:
# 				measured_fps = fps_count / (now - last_fps_ts)
# 				fps_count = 0
# 				last_fps_ts = now

# 			# Initialize writer lazily when we know frame size
# 			if output and writer is None:
# 				h, w = frame.shape[:2]
# 				# try to read FPS from capture
# 				cap_fps = cap.get(cv2.CAP_PROP_FPS) or measured_fps or 25.0
# 				print(f"Starting recording -> {output} (w={w}, h={h}, fps={cap_fps:.2f})")
# 				writer = make_writer(output, 'mp4v', cap_fps, w, h)

# 			# Overlay info
# 			cv2 = _get_cv2()
# 			overlay = f"FPS: {measured_fps:.1f}  Frame: {frame_count}  Source: {source}"
# 			cv2.putText(frame, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 			if display:
# 				cv2.imshow('camera', frame)

# 			if writer is not None:
# 				writer.write(frame)

# 			# Key handling: q to quit, s to save snapshot
# 			if display:
# 				key = cv2.waitKey(1) & 0xFF
# 				if key == ord('q'):
# 					print('Quit requested by user')
# 					break
# 				elif key == ord('s'):
# 					fname = f'snapshot_{int(time.time())}.jpg'
# 					cv2.imwrite(fname, frame)
# 					print(f'Snapshot saved to {fname}')
# 			else:
# 				# When not displaying, sleep a tiny bit to avoid busy-loop
# 				time.sleep(0.001)

# 	except KeyboardInterrupt:
# 		print('Interrupted by user')
# 	finally:
# 		try:
# 			cap.release()
# 		except Exception:
# 			pass
# 		if writer is not None:
# 			writer.release()
# 		if display:
# 			cv2.destroyAllWindows()


# def parse_args():
# 	p = argparse.ArgumentParser(description='View/record camera streams (RTSP/MJPEG/device)')
# 	p.add_argument('--source', '-s', required=False, help='Video source (RTSP URL, http MJPEG URL, or /dev/videoX)')
# 	p.add_argument('--discover', action='store_true', help='Run SSDP/UPnP discovery to find cameras on the local network and exit')
# 	p.add_argument('--discover-timeout', type=float, default=3.0, help='Discovery listen timeout in seconds (default 3.0)')
# 	p.add_argument('--scan', action='store_true', help='Scan local subnet for hosts with common camera ports (554/80/8080)')
# 	p.add_argument('--scan-net', help='Network CIDR to scan (e.g. 192.168.1.0/24). If omitted, a /24 containing the default IP will be used.')
# 	p.add_argument('--scan-timeout', type=float, default=0.4, help='Per-port probe timeout in seconds (default 0.4)')
# 	p.add_argument('--scan-workers', type=int, default=200, help='Number of concurrent probe workers (default 200)')
# 	p.add_argument('--probe-nvr', help='Comma-separated NVR IP(s) to probe with common RTSP paths (e.g. 10.0.0.1[,10.0.0.2])')
# 	p.add_argument('--probe-user', help='Username for RTSP (if needed)')
# 	p.add_argument('--probe-pass', help='Password for RTSP (if needed)')
# 	p.add_argument('--probe-timeout', type=float, default=3.0, help='Timeout (s) for each RTSP probe attempt')
# 	p.add_argument('--probe-workers', type=int, default=20, help='Concurrent workers for RTSP probe attempts')
# 	p.add_argument('--display', dest='display', action='store_true', help='Show video window (default: show)')
# 	p.add_argument('--no-display', dest='display', action='store_false', help='Do not show video window (headless)')
# 	p.set_defaults(display=True)
# 	p.add_argument('--output', '-o', help='Optional output file to record (mp4 suggested)')
# 	p.add_argument('--width', type=int, help='Preferred capture width')
# 	p.add_argument('--height', type=int, help='Preferred capture height')
# 	p.add_argument('--reconnect', type=float, default=1.0, help='Initial reconnect delay in seconds (default 1.0)')
# 	return p.parse_args()


# def main():
# 	args = parse_args()
# 	# If discovery requested, run discovery and exit
# 	if args.discover:
# 		results = discover_ssdp(timeout=args.discover_timeout)
# 		report_discovery(results)
# 		return
# 	# If scan requested, run TCP scan across local subnet and exit
# 	if args.scan:
# 		cidr = args.scan_net or get_local_network_cidr()
# 		if not cidr:
# 			print('Could not infer local network CIDR; please pass --scan-net (e.g. 192.168.1.0/24)')
# 			return
# 		print(f'Scanning network {cidr} for common camera ports...')
# 		hosts = list(iter_hosts_from_cidr(cidr))
# 		# To avoid scanning ourselves in huge nets, limit to first 254 hosts for /24 by default
# 		if len(hosts) > 1024:
# 			hosts = hosts[:1024]
# 		scan_results = tcp_probe_hosts(hosts, ports=(554, 80, 8080), timeout=args.scan_timeout, workers=args.scan_workers)
# 		report_scan(scan_results)
# 		return
# 	# If probe-nvr requested, try common RTSP URL patterns against NVR IP(s)
# 	if args.probe_nvr:
# 		hosts = [h.strip() for h in args.probe_nvr.split(',') if h.strip()]
# 		if not hosts:
# 			print('No hosts provided to --probe-nvr')
# 			return
# 		results = probe_nvr_hosts(hosts, user=args.probe_user or '', passwd=args.probe_pass or '', timeout=args.probe_timeout, workers=args.probe_workers)
# 		if not results:
# 			print('No working RTSP URLs found for given NVR(s).')
# 		else:
# 			print('\nWorking RTSP URLs:')
# 			for host, urls in results.items():
# 				print(f' - {host}:')
# 				for u in urls:
# 					print('    ', u)
# 		return
# 	# Basic contract / quick checks
# 	if args.output and not args.output.lower().endswith(('.mp4', '.avi', '.mkv')):
# 		print('Warning: output filename does not use common video extension (mp4/avi/mkv). Proceeding anyway.')

# 	if not args.source:
# 		print('Error: --source is required when not running --discover')
# 		return

# 	run_viewer(args.source, display=args.display, output=args.output, width=args.width, height=args.height, reconnect_delay=args.reconnect)


# if __name__ == '__main__':
# 	main()


import cv2

# Replace with your RTSP URL (example for channel 1, main stream)
rtsp_url = "rtsp://admin:admin@192.168.1.100:554/ch1_1.264"  # Or try other formats from Step 2

# Open the video stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream. Check URL, IP, credentials, and network.")
    exit()

print("Stream opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Display the frame
    cv2.imshow('Night Owl Camera Feed', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()