# NCCL Testing Over RoCE Fabric - Complete Tutorial

## Overview

This guide covers setting up and running NCCL (NVIDIA Collective Communications Library) tests between bare metal GPU nodes using RoCE (RDMA over Converged Ethernet) fabric for high-performance inter-node communication.

### Why Run NCCL Tests?

NCCL tests are essential benchmarking tools that validate and measure GPU-to-GPU communication performance in distributed computing environments. These tests help ensure your multi-GPU cluster is working optimally for:

- **Distributed Deep Learning**: Training large neural networks across multiple GPUs and nodes
- **High-Performance Computing**: Scientific computing workloads requiring massive parallel processing
- **AI/ML Workloads**: Large-scale machine learning model training and inference
- **Performance Validation**: Ensuring hardware investments deliver expected performance

### Expected Performance Results

#### Single Node Testing with NVSwitch
For single-node testing with NVSwitch interconnect (like H100 systems), you can expect:

- **Peak Bandwidth**: 300-400+ GB/s aggregate bandwidth for large message sizes (>64MB)
- **Low Latency**: Base latency of 10-20μs for small messages
- **Linear Scaling**: Near-perfect scaling across all GPUs within the node
- **High Efficiency**: >90% of theoretical interconnect bandwidth utilization
- **Consistent Performance**: Minimal variance across test runs

**Example Single-Node Performance (8x H100 with NVSwitch):**
- Small messages (1KB-32KB): Latency-dominated, ~15-25μs
- Medium messages (256KB-16MB): Rapid bandwidth scaling to 100+ GB/s
- Large messages (64MB+): Peak performance 350-450 GB/s aggregate

#### Multi-Node Testing with RoCE Fabric
For multi-node testing over RoCE (RDMA over Converged Ethernet), expect:

- **Network Bandwidth**: 50-120 GB/s aggregate bandwidth (depends on adapter speed)
- **RDMA Latency**: 15-50μs for small messages with RDMA optimization
- **Scaling Efficiency**: Near-linear scaling with RDMA direct memory access
- **Transport Protocol**: InfiniBand verbs over Ethernet (RoCE v2)
- **Performance Scaling**: 3-7 GB/s effective per-GPU bandwidth for inter-node communication

**Example Multi-Node Performance (2 nodes, 16 GPUs total over 400Gbps RoCE):**
- Small messages: RDMA-optimized latency <100μs
- Large messages: 80-120 GB/s peak aggregate bandwidth
- Efficiency: Direct GPU memory access without CPU copy overhead

### RoCE vs. TCP/IP Performance Comparison

| Metric | RoCE (RDMA) | TCP/IP |
|--------|-------------|---------|
| Latency | 15-50μs | 100-500μs |
| CPU Overhead | Minimal (kernel bypass) | High (CPU processing) |
| Bandwidth Efficiency | 85-95% | 60-80% |
| Memory Access | Direct GPU memory | System memory copy |
| Protocol | InfiniBand verbs | Standard networking stack |

## Prerequisites

- Multiple bare metal servers with NVIDIA GPUs

## Quick Installation

```bash
apt install -y git
rm -rf nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests 
make -j16 MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/
```

## Table of Contents

1. [Hardware and Network Setup](#1-hardware-and-network-setup)
2. [Software Installation](#2-software-installation)
3. [Quick Setup](#3-quick-setup)
4. [Single Node Test](#4-single-node-test)
5. [Multi-Node Test](#5-multi-node-test)
6. [Troubleshooting](#6-troubleshooting)

## 1. Hardware and Network Setup

### Network Infrastructure Requirements

**Note: DigitalOcean Pre-Configured Infrastructure**

DigitalOcean bare metal GPU instances come with pre-configured high-performance networking infrastructure optimized for GPU workloads. The following components are already set up and managed by DigitalOcean:

- **Switch Configuration**: PFC (Priority Flow Control) and ECN (Explicit Congestion Notification) are pre-configured on the network fabric
- **RoCE Infrastructure**: RDMA over Converged Ethernet is enabled and optimized for GPU-to-GPU communication
- **Cabling**: High-speed network connections (typically 400GbE) between nodes are professionally installed
- **Network Topology**: Low-latency, high-bandwidth connectivity with optimized routing between GPU nodes
- **VLAN Configuration**: Dedicated VLANs for GPU backend communication are pre-configured
- **Mellanox ConnectX Adapters**: High-performance RDMA-capable network adapters are installed and configured

**What This Means for You:**
- No manual network configuration required
- Optimal performance out-of-the-box
- Enterprise-grade network reliability
- Professional cable management and topology design

### Verify GPU and Network Hardware

```bash
# Check GPUs
nvidia-smi

# Check network interfaces
ip link show
ibv_devices  # Should show RDMA devices

# Check RoCE capability
show_gids | grep -i roce

# Method 1: Use rdma command (more modern approach)
rdma link show

# Method 2: Use ibv_devinfo to check port capabilities
ibv_devinfo -v | grep -i roce
```

## 2. Software Installation

### Install Required Packages

```bash
# Update system 
sudo apt update && sudo apt upgrade -y

# Install essential packages (optional: already in the system)
sudo apt install -y build-essential cmake git
sudo apt install -y libibverbs-dev librdmacm-dev
sudo apt install -y rdma-core perftest

# Install NVIDIA drivers (if not already installed) 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-drivers cuda-toolkit-12-4
```

### Install NCCL

```bash
# Download and install NCCL 
wget https://developer.download.nvidia.com/compute/redist/nccl/v2.19.3/nccl_2.19.3-1+cuda12.0_x86_64.txz
tar -xvf nccl_2.19.3-1+cuda12.0_x86_64.txz
sudo cp -r nccl_2.19.3-1+cuda12.0_x86_64/* /usr/local/
sudo ldconfig

# Verify installation 
pkg-config --modversion nccl
```

### Install NCCL Tests (if not installed in above step)

```bash
# Clone NCCL tests repository 
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Build NCCL tests 
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local
```

## 3. Quick Setup

### 1. Install NCCL Tests

```bash
# Install prerequisites
apt install -y git openmpi-bin openmpi-common libopenmpi-dev

# Build NCCL tests
cd ~
rm -rf nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests 
make -j16 MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/
```

### 2. Find Your RoCE Interfaces

```bash
# Find interfaces with RDMA devices (these are your RoCE interfaces)
for iface in $(ls /sys/class/net/); do
    rdma_dev=$(ls /sys/class/net/$iface/device/infiniband/ 2>/dev/null)
    if [ -n "$rdma_dev" ]; then
        echo "RoCE Interface: $iface → RDMA device: $rdma_dev"
    fi
done

# Alternative: Use ibdev2netdev to see mappings
ibdev2netdev
```

**Sample Output:**
```
RoCE Interface: enp156s0np0 → RDMA device: mlx5_6
```

### 3. Check GPU-NIC Topology

```bash
# See which NICs are closest to which GPUs
nvidia-smi topo -m
```

## 4. Single Node Test

### Create Test Script

```bash
cat > ~/test_single_node.sh << 'EOF'
#!/bin/bash

echo "=== Single Node NCCL Test ==="

# Get GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Testing with $GPU_COUNT GPUs"

cd ~/nccl-tests

# Run the test (target: >400 GB/s bus bandwidth)
./build/all_reduce_perf -b 1G -f 2 -e 1G -g $GPU_COUNT
EOF

chmod +x ~/test_single_node.sh
```

### Run Single Node Test

```bash
~/test_single_node.sh
```

**Expected Output:** Look for `busbw` > 400 GB/s in the results.

**Sample Output:**
```
  size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
  1073741824     268435456     float     sum      -1   5110.3  210.11  367.70      0   5112.6  210.02  367.53      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 367.616 
#
# Collective test concluded: all_reduce_perf
```

## 5. Multi-Node Test

### Create Multi-Node Test Script

```bash
cat > ~/test_multi_node.sh << 'EOF'
#!/bin/bash

# Usage: ./test_multi_node.sh "ip1,ip2"
HOSTS=$1

if [ -z "$HOSTS" ]; then
    echo "Usage: $0 \"ip1,ip2,ip3\""
    echo "Example: $0 \"192.168.1.10,192.168.1.11\""
    exit 1
fi

# Parse hosts
IFS=',' read -r -a ip_array <<< "$HOSTS"
NP=$(echo "${ip_array[@]}" | tr ' ' '\n' | sort -u | wc -l)

# Assuming 8 GPUs per node (adjust as needed)
NEW_NP=$(( NP * 8 ))
NEW_HOSTS=$(echo $HOSTS | sed 's/\([0-9\.]*\)/\1:8/g')

echo "Testing $NP nodes, $NEW_NP total GPUs"

cd ~/nccl-tests

# Run multi-node test
mpirun --bind-to none --mca pml ucx --allow-run-as-root \
        -x NCCL_CROSS_NIC=0 \
        -np $NEW_NP -H ${NEW_HOSTS} \
        ./build/all_reduce_perf -b 1G -f 2 -e 1G
EOF

chmod +x ~/test_multi_node.sh
```

### Run Multi-Node Test

```bash
# Replace with your actual node IPs
~/test_multi_node.sh "10.45.169.116,10.45.169.109"
```

**Expected Output:** Look for `busbw` > 400 GB/s across nodes.

## 6. Troubleshooting

### Abrupt Test Termination

An abrupt test termination is usually associated with hardware problems, like a misconfiguration or a bad cable, so it is worth checking the following:

- All GPU NICs are up
- They are all on the same VLAN
- They are plugged in the right leaf switches

**These are example outputs associated to good nodes:**

#### Interfaces are up
```bash
# Currently back-end network is addressed via IPv6, this may be subject to change to enable PFC.
# ip -br a
lo               UNKNOWN        127.0.0.1/8 ::1/128 
eno8303          DOWN           
eno8403          DOWN           
enp26s0np0       UP             fe80::a288:c2ff:fee4:446a/64 
enp27s0f0np0     UP             
enp27s0f1np1     UP             
enp60s0np0       UP             fe80::a288:c2ff:fee3:638a/64 
enp77s0np0       UP             fe80::a288:c2ff:feda:31ea/64 
enp94s0np0       UP             fe80::a288:c2ff:feda:d712/64 
enp156s0np0      UP             fe80::a288:c2ff:feda:d6c2/64 
enp157s0f0np0    UP             
enp157s0f1np1    UP             
enp188s0np0      UP             fe80::5aa2:e1ff:fe11:5ffc/64 
enp204s0np0      UP             fe80::a288:c2ff:fee4:4322/64 
enp220s0np0      UP             fe80::a288:c2ff:feda:d6e2/64 
bond0            UP             10.45.169.109/24 fe80::9e63:c0ff:fef6:1eb2/64 
```

#### Interfaces are on the same VLAN
```bash
# lldpctl | grep -i VLAN
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
  VLAN:         108, pvid: yes vlan-108
```

#### Interfaces are ordered correctly on leaf switches (compare output on both nodes - order must match)
```bash
# lshw -c network -businfo | grep X-7 | awk '{print $2}' | xargs lldpctl | grep SysName
    SysName:      am4-t0-g1-leaf07
    SysName:      am4-t0-g1-leaf04
    SysName:      am4-t0-g1-leaf03
    SysName:      am4-t0-g1-leaf08
    SysName:      am4-t0-g1-leaf06
    SysName:      am4-t0-g1-leaf05
    SysName:      am4-t0-g1-leaf01
    SysName:      am4-t0-g1-leaf02
```

---

### If Test Hangs

Add debug info:

```bash
# Add to your multi-node script
mpirun --bind-to none --mca pml ucx --allow-run-as-root \
        -x NCCL_CROSS_NIC=0 \
        -x NCCL_DEBUG=INFO \
        -np $NEW_NP -H ${NEW_HOSTS} \
        ./build/all_reduce_perf -b 1G -f 2 -e 1G
```

Look for messages like:
- **Good**: `via NET/IB/1/GDRDMA` (RDMA working)
- **Bad**: `via NET/IB/1` (falling back to system memory)

**Expected Output:**
```
atl1g3r4u9gpu:146600:146782 [5] transport/net_ib.cc:2244 NCCL WARN NET/IB: Got completion from peer 10.46.70.21<50854> with status=5 opcode=31245 len=0 vendor err 249 (Recv) localGid fe80::9e63:c0ff:feab:b8da remoteGidsfe80::9e63:c0ff:feab:b9ca hca mlx5_9
atl1g3r4u9gpu:146600:146782 [5] NCCL INFO transport/net.cc:1362 -> 6
atl1g3r4u9gpu:146600:146782 [5] NCCL INFO proxy.cc:728 -> 6
atl1g3r4u9gpu:146600:146782 [5] NCCL INFO proxy.cc:912 -> 6 [Progress Thread]

atl1g3r1u16gpu:496000:496186 [6] transport/net_ib.cc:2244 NCCL WARN NET/IB: Got completion from peer 10.46.70.22<39226> with status=12 opcode=32224 len=0 vendor err 129 (Recv) localGid fe80::9e63:c0ff:feab:938a remoteGidsfe80::9e63:c0ff:feab:b6b2 hca mlx5_10
atl1g3r1u16gpu:496000:496186 [6] NCCL INFO transport/net.cc:1362 -> 6
atl1g3r1u16gpu:496000:496186 [6] NCCL INFO proxy.cc:728 -> 6
atl1g3r1u16gpu:496000:496186 [6] NCCL INFO proxy.cc:912 -> 6 [Progress Thread]
```

They indicate that pairs of nodes (Node-A and Node-B), in this case 10.46.70.21 and 10.46.70.22, are having communication issues over the GPU fabric. This can be verified by using ping6 from Node-A:

```bash
ping6 -c 2 -I enp156s0np0 fe80::9e63:c0ff:feab:b9ca
```

Where `enp156s0np0` is one of the 8 400G NICs, and `fe80::9e63:c0ff:feab:b9ca` is the IPv6 address of the same NIC on Node-B.

---

### Slow Bus Bandwidth

In this case, you need to isolate the culprit. The most obvious thing to rule out is software misconfiguration. What happens most often than not is that the GPU ends up talking to the NIC off system memory, so RDMA is not in place. In other words you would need to see RDMA in-place and NCCL tells you so in this form when you pass `-x NCCL_DEBUG=info` to its arguments:

**Good output (RDMA working):**
```
[...]
am4g1r34bm1:23858:23875 [0] NCCL INFO Channel 00/0 : 0[6] -> 1[6] [send] via NET/IB/1/GDRDMA
am4g1r34bm1:23858:23875 [0] NCCL INFO Channel 01/0 : 0[6] -> 1[6] [send] via NET/IB/1/GDRDMA
am4g1r34bm1:23858:23875 [0] NCCL INFO Channel 02/0 : 0[6] -> 1[6] [send] via NET/IB/1/GDRDMA
am4g1r34bm1:23858:23875 [0] NCCL INFO Channel 03/0 : 0[6] -> 1[6] [send] via NET/IB/1/GDRDMA
```

**Bad output (falling back to system memory):**
```
[...]
am4g1r35bm1:1226446:1226462 [0] NCCL INFO Channel 00/0 : 0[6] -> 1[6] [receive] via NET/IB/1
am4g1r35bm1:1226446:1226462 [0] NCCL INFO Channel 01/0 : 0[6] -> 1[6] [receive] via NET/IB/1
am4g1r35bm1:1226446:1226462 [0] NCCL INFO Channel 00/0 : 1[6] -> 0[6] [send] via NET/IB/1
am4g1r35bm1:1226446:1226462 [0] NCCL INFO Channel 01/0 : 1[6] -> 0[6] [send] via NET/IB/1
```

And this is bad. In this case check that all the NVIDIA drivers are loaded correctly:

```bash
# lsmod | grep nvid
nvidia_uvm           4956160  0
nvidia_peermem         16384  0
ib_uverbs             200704  3 nvidia_peermem,rdma_ucm,mlx5_ib
nvidia_drm            122880  0
nvidia_modeset       1355776  1 nvidia_drm
nvidia              54292480  63 nvidia_uvm,nvidia_peermem,nvidia_modeset
video                  73728  2 dell_wmi,nvidia_modeset
```

`nvidia_peermem` is the module in question. If that is not loaded, then you probably need to rebuild it and install it again. Check what you have:

```bash
# dkms status | grep nv
nvidia/550.90.07, 6.8.0-52-generic, x86_64: installed
```

Then do:

```bash
# Remove nvidia driver
dkms remove nvidia/550.90.07

# Install nvidia driver
dkms install nvidia/550.90.07
```

Once that is done, modprobe module `nvidia_peermem` in and verify that is loaded correctly. You can attempt now to perform a new test, if it is still not in the clear, then performance slowness could be the result of the following:

- Bad optics
- Bad fiber
- Bad NIC (if a card is replaced, remember to reset ETH_MODE and ATS attributes, check the appendix for more details)
- Bad GPU

#### Check NVIDIA Drivers

```bash
# Verify nvidia_peermem is loaded (required for RDMA)
lsmod | grep nvidia_peermem

# If missing, reload drivers
sudo dkms remove nvidia/$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
sudo dkms install nvidia/$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
sudo modprobe nvidia_peermem
```

#### Test Individual GPU-NIC Pairs

If you get low bandwidth, test each GPU-NIC combination:

```bash
cat > ~/test_gpu_nic_pair.sh << 'EOF'
#!/bin/bash

# Usage: ./test_gpu_nic_pair.sh "ip1,ip2" gpu_id nic_id
HOSTS=$1
GPU_ID=$2
NIC_ID=$3

NEW_HOSTS=$(echo $HOSTS | sed 's/\([0-9\.]*\)/\1:1/g')

echo "Testing GPU$GPU_ID with NIC$NIC_ID"

cd ~/nccl-tests

mpirun --bind-to none --mca pml ucx --allow-run-as-root \
        -x NCCL_CROSS_NIC=0 \
        -x NCCL_DEBUG=info \
        -x CUDA_VISIBLE_DEVICES=$GPU_ID \
        -x NCCL_IB_HCA=mlx5_$NIC_ID \
        -np 2 -H ${NEW_HOSTS} \
        ./build/all_reduce_perf -b 1G -f 2 -e 1G
EOF

chmod +x ~/test_gpu_nic_pair.sh

# Test specific GPU-NIC pairs (based on nvidia-smi topo -m)
~/test_gpu_nic_pair.sh "ip1,ip2" 0 0
~/test_gpu_nic_pair.sh "ip1,ip2" 1 3
# etc.
```

**Expected single GPU-NIC bandwidth:** ~48 GB/s

---

### Network Verification

#### Check Interface Status

```bash
# Check interfaces are up
ip -br a

# Check VLAN (all interfaces should be on same VLAN)
lldpctl | grep -i VLAN

# Check switch connectivity
lshw -c network -businfo | grep ConnectX | awk '{print $2}' | xargs lldpctl | grep SysName
```

#### Test Network Connectivity

```bash
# Ping between nodes using RoCE interfaces
ping6 -c 2 -I enp156s0np0 fe80::target_ipv6_address
```

## Additional Resources

- [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [OpenMPI Documentation](https://www.open-mpi.org/)

## Contributing

Contributions to this guide are welcome! Please feel free to submit issues or pull requests.

## License

This guide is provided under the MIT License.
