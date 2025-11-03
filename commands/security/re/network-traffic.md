# Network Traffic Analysis Command

**Category**: Reverse Engineering
**Phase**: 4 - Advanced Analysis
**Complexity**: Medium
**⚠️ SECURITY WARNING**: Analyze only authorized network traffic

## Purpose

Deep packet inspection and network traffic analysis for security research, protocol reverse engineering, and threat detection.

## Usage

```bash
/re:network-traffic [action] [options]

Actions:
  capture             Capture live traffic
  analyze             Analyze PCAP file
  decode              Decode protocols
  extract             Extract files/credentials
  correlate           Correlate with threat intelligence
  visualize           Generate network topology

Options:
  --interface <if>    Network interface (eth0, wlan0)
  --pcap <file>       PCAP file to analyze
  --filter <expr>     BPF filter expression
  --duration <sec>    Capture duration
  --protocol <name>   Target protocol (http, tls, dns, smtp)
  --extract-files     Extract transferred files
  --decrypt-tls       Attempt TLS decryption (with keys)
```

## Examples

```bash
# Capture HTTP traffic for 5 minutes
/re:network-traffic capture --interface eth0 --filter "tcp port 80" --duration 300

# Analyze existing PCAP file
/re:network-traffic analyze --pcap traffic.pcap

# Decode specific protocol
/re:network-traffic decode --pcap capture.pcap --protocol http

# Extract files from network traffic
/re:network-traffic extract --pcap traffic.pcap --extract-files

# Correlate with threat intelligence
/re:network-traffic correlate --pcap suspicious.pcap --virustotal

# Generate network visualization
/re:network-traffic visualize --pcap traffic.pcap --output network-graph.png
```

## Traffic Capture

### 1. tcpdump (Linux/macOS)

```bash
# Capture all traffic on eth0
tcpdump -i eth0 -w capture.pcap

# Capture HTTP traffic
tcpdump -i eth0 'tcp port 80' -w http.pcap

# Capture DNS queries
tcpdump -i eth0 'udp port 53' -w dns.pcap

# Capture specific IP
tcpdump -i eth0 'host 192.168.1.100' -w target.pcap

# Capture with snaplen (full packets)
tcpdump -i eth0 -s 65535 -w full.pcap

# Capture with timestamp precision
tcpdump -i eth0 -tttt -w timestamped.pcap

# Rotate capture files (100MB each)
tcpdump -i eth0 -w capture.pcap -C 100
```

### 2. Wireshark/tshark

```bash
# Capture with tshark (CLI Wireshark)
tshark -i eth0 -w capture.pcap

# Capture with display filter
tshark -i eth0 -f "port 443" -w tls.pcap

# Live capture with protocol decode
tshark -i eth0 -Y http -T fields -e http.request.uri

# Capture and display statistics
tshark -i eth0 -z io,stat,1

# Extract HTTP objects
tshark -r capture.pcap --export-objects http,output/
```

## Protocol Analysis

### HTTP/HTTPS Analysis

```bash
# Extract HTTP requests
tshark -r traffic.pcap -Y http.request -T fields \
  -e ip.src -e ip.dst -e http.request.method -e http.request.uri

# Extract HTTP headers
tshark -r traffic.pcap -Y http -T fields \
  -e http.user_agent -e http.host -e http.cookie

# Decode HTTP traffic
tcpflow -r traffic.pcap -o http_flows/

# Extract credentials
tshark -r traffic.pcap -Y "http.authorization" -T fields \
  -e http.authorization
```

### DNS Analysis

```bash
# Extract DNS queries
tshark -r traffic.pcap -Y dns.qry.name -T fields \
  -e dns.qry.name -e dns.resp.addr

# Detect DNS tunneling
tshark -r traffic.pcap -Y "dns.qry.name contains '.'" -T fields \
  -e dns.qry.name | awk 'length > 50'

# DNS statistics
tshark -r traffic.pcap -z dns,tree

# Suspicious DNS patterns
tshark -r traffic.pcap -Y 'dns.qry.name matches "^[a-f0-9]{32}"'
```

### TLS/SSL Analysis

```bash
# Extract TLS handshakes
tshark -r traffic.pcap -Y tls.handshake -T fields \
  -e tls.handshake.type -e tls.handshake.version

# Extract TLS certificates
tshark -r traffic.pcap -Y tls.handshake.certificate -T fields \
  -e x509sat.uTF8String

# Decrypt TLS with key log
tshark -r traffic.pcap -o tls.keylog_file:sslkeys.log -Y http

# Detect weak ciphers
tshark -r traffic.pcap -Y "tls.handshake.ciphersuite == 0x002f"
```

## File Extraction

### Extract Files from PCAP

```python
# extract_files.py
from scapy.all import *
import hashlib

def extract_files(pcap_file):
    packets = rdpcap(pcap_file)
    files = {}

    for packet in packets:
        if packet.haslayer(Raw):
            # HTTP file extraction
            if b'HTTP' in bytes(packet[Raw]):
                payload = bytes(packet[Raw].load)
                if b'Content-Type' in payload:
                    # Extract file
                    file_data = payload.split(b'\r\n\r\n', 1)[1]
                    file_hash = hashlib.md5(file_data).hexdigest()
                    files[file_hash] = file_data

    return files

# Usage
files = extract_files('traffic.pcap')
for hash, data in files.items():
    with open(f'extracted/{hash}', 'wb') as f:
        f.write(data)
```

### Extract Credentials

```bash
# Extract FTP credentials
tshark -r traffic.pcap -Y "ftp.request.command == USER || ftp.request.command == PASS" \
  -T fields -e ftp.request.arg

# Extract HTTP Basic Auth
tshark -r traffic.pcap -Y http.authorization -T fields \
  -e http.authorization | base64 -d

# Extract SMTP credentials
tshark -r traffic.pcap -Y "smtp.req.command == AUTH" -T fields \
  -e smtp.req.parameter
```

## Threat Detection

### Suspicious Patterns

```python
# detect_threats.py
from scapy.all import *

def detect_threats(pcap_file):
    packets = rdpcap(pcap_file)
    threats = []

    for packet in packets:
        # Detect port scanning
        if packet.haslayer(TCP) and packet[TCP].flags == 'S':
            threats.append({
                'type': 'port_scan',
                'src': packet[IP].src,
                'dst_port': packet[TCP].dport
            })

        # Detect large DNS responses (tunneling)
        if packet.haslayer(DNS) and len(packet) > 512:
            threats.append({
                'type': 'dns_tunneling',
                'query': packet[DNS].qd.qname
            })

        # Detect non-standard HTTP methods
        if packet.haslayer(Raw):
            payload = bytes(packet[Raw].load)
            if any(method in payload for method in [b'PROPFIND', b'TRACK', b'TRACE']):
                threats.append({
                    'type': 'suspicious_http_method',
                    'src': packet[IP].src
                })

    return threats
```

### Network Anomaly Detection

```python
# anomaly_detection.py
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(pcap_file):
    packets = rdpcap(pcap_file)

    # Extract features
    features = []
    for packet in packets:
        if packet.haslayer(IP):
            features.append({
                'packet_len': len(packet),
                'ip_len': packet[IP].len,
                'ttl': packet[IP].ttl,
                'protocol': packet[IP].proto
            })

    df = pd.DataFrame(features)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.1)
    predictions = model.fit_predict(df)

    # Identify anomalies
    anomalies = df[predictions == -1]
    return anomalies
```

## Protocol Reverse Engineering

### Unknown Protocol Analysis

```python
# protocol_analysis.py
from scapy.all import *
import collections

def analyze_unknown_protocol(pcap_file, port):
    packets = rdpcap(pcap_file)
    protocol_data = []

    for packet in packets:
        if packet.haslayer(TCP) and packet[TCP].dport == port:
            if packet.haslayer(Raw):
                payload = bytes(packet[Raw].load)
                protocol_data.append(payload)

    # Statistical analysis
    byte_freq = collections.Counter()
    for data in protocol_data:
        byte_freq.update(data)

    # Detect patterns
    patterns = find_common_patterns(protocol_data)

    return {
        'byte_frequency': byte_freq.most_common(10),
        'common_patterns': patterns,
        'avg_packet_size': sum(len(d) for d in protocol_data) / len(protocol_data)
    }

def find_common_patterns(data_list):
    patterns = collections.Counter()
    for data in data_list:
        # Look for 4-byte patterns
        for i in range(len(data) - 4):
            pattern = data[i:i+4]
            patterns[pattern] += 1
    return patterns.most_common(5)
```

## Network Topology Visualization

```python
# network_topology.py
import networkx as nx
import matplotlib.pyplot as plt
from scapy.all import *

def visualize_network(pcap_file):
    packets = rdpcap(pcap_file)
    G = nx.DiGraph()

    # Build graph
    for packet in packets:
        if packet.haslayer(IP):
            src = packet[IP].src
            dst = packet[IP].dst

            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)

    # Draw graph
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.savefig('network_topology.png')
    plt.close()

    return G
```

## TLS Decryption

### With SSLKEYLOGFILE

```bash
# Set environment variable before running browser/app
export SSLKEYLOGFILE=/tmp/sslkeys.log

# Capture traffic
tcpdump -i eth0 -w encrypted.pcap

# Decrypt with Wireshark
wireshark -o tls.keylog_file:/tmp/sslkeys.log encrypted.pcap

# Or with tshark
tshark -r encrypted.pcap -o tls.keylog_file:/tmp/sslkeys.log -Y http
```

### With Private Key

```bash
# Decrypt RSA TLS traffic (if you have server private key)
tshark -r traffic.pcap \
  -o tls.keys_list:"0.0.0.0,443,http,/path/to/server.key" \
  -Y http
```

## Performance Analysis

```bash
# Throughput analysis
tshark -r traffic.pcap -z io,stat,1,"AVG(frame.len)frame.len"

# Top talkers
tshark -r traffic.pcap -z ip_hosts,tree

# Protocol hierarchy
tshark -r traffic.pcap -z io,phs

# Conversation statistics
tshark -r traffic.pcap -z conv,tcp
```

## Wireshark Display Filters

```
# HTTP requests
http.request

# HTTP errors
http.response.code >= 400

# TLS handshakes
tls.handshake.type == 1

# DNS queries to specific domain
dns.qry.name contains "malicious"

# Large packets (potential data exfiltration)
frame.len > 1000

# Non-standard ports
tcp.port > 49152

# SYN flood detection
tcp.flags.syn == 1 && tcp.flags.ack == 0
```

## Best Practices

1. **Legal Authorization**: Only capture authorized traffic
2. **Privacy**: Redact sensitive data before sharing
3. **Storage**: Encrypt PCAPs containing sensitive data
4. **Retention**: Follow data retention policies
5. **Analysis Scope**: Focus on relevant time windows
6. **Documentation**: Document findings thoroughly
7. **Chain of Custody**: Maintain for legal cases
8. **Tool Validation**: Verify tool accuracy

## Integration Points

- **/re:malware-sandbox** - Malware network behavior
- **/re:memory-dump** - Network artifacts in memory
- **Flow-Nexus Sandboxes** - Cloud-based packet capture

## Agent Integration

```javascript
// Network Analyst: Automated traffic analysis
mcp__ruv-swarm__agent_spawn({
  type: "analyst",
  capabilities: ["network-analysis", "threat-detection"]
})

// Threat Intel: IOC extraction and correlation
mcp__ruv-swarm__task_orchestrate({
  task: "Analyze PCAP for malicious indicators",
  strategy: "parallel"
})
```

## Related Commands

- `/re:malware-sandbox` - Malware analysis
- `/re:memory-dump` - Memory forensics
- `/security:audit` - Security auditing

---

**Implementation Status**: Production-Ready
**Security Level**: HIGH
**Last Updated**: 2025-11-01
**Maintainer**: Network Security Specialist
