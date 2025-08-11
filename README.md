# NCCL Guide for DigitalOcean Bare Metal Machines

This guide provides step-by-step instructions for setting up and running NCCL (NVIDIA Collective Communications Library) on DigitalOcean bare metal machines with GPUs.

## Overview

NCCL is NVIDIA's library for multi-GPU and multi-node communication primitives that are performance-optimized for NVIDIA GPUs. This guide will walk you through the process of configuring and testing NCCL across multiple DigitalOcean bare metal instances.

## Understanding NCCL Tests

### What are NCCL Tests?

NCCL tests are benchmarking tools that validate and measure the performance of GPU-to-GPU communication using NVIDIA's NCCL library. These tests help ensure that your multi-GPU setup is working correctly and performing optimally.

### Intra-Node Testing

**Intra-node** testing refers to communication between GPUs within a single machine (node). This tests:

- **GPU-to-GPU communication** within the same server
- **PCIe bandwidth** and efficiency between GPUs
- **NVLink performance** (if available) for direct GPU-to-GPU connections
- **Memory bandwidth** and latency between GPUs on the same motherboard
- **Local NUMA effects** and CPU-GPU memory transfer rates

Intra-node tests are crucial for:
- Validating that all GPUs in a single machine can communicate effectively
- Measuring baseline performance before scaling to multiple nodes
- Identifying hardware issues like faulty PCIe slots or NVLink connections
- Optimizing GPU placement and affinity settings

### Inter-Node Testing

**Inter-node** testing refers to communication between GPUs across different machines (nodes) over the network. This tests:

- **Network bandwidth** between different servers
- **Network latency** and communication overhead
- **RoCE (RDMA over Converged Ethernet) performance** for GPU-to-GPU communication across nodes
- **RDMA capabilities** and network fabric efficiency
- **Scalability** of communication patterns as you add more nodes

Inter-node tests are essential for:
- Validating distributed training and multi-node workloads
- Measuring network bottlenecks in multi-GPU clusters
- Testing fault tolerance and network reliability
- Optimizing network topology and configuration
- Benchmarking real-world distributed computing scenarios

### Why Both Tests Matter

Running both intra-node and inter-node tests provides a complete picture of your GPU cluster's communication capabilities. Intra-node tests establish your baseline single-machine performance, while inter-node tests validate that your network infrastructure can effectively scale your workloads across multiple machines.



Before starting, ensure you have:

- [ ] DigitalOcean account with access to GPU-enabled bare metal instances
- [ ] Basic understanding of Linux command line
- [ ] SSH access to your instances
- [ ] NVIDIA drivers and CUDA toolkit installed

## Table of Contents

1. [Setting Up DigitalOcean Bare Metal Instances](#setting-up-digitalocean-bare-metal-instances)
2. [Installing Dependencies](#installing-dependencies)
3. [Configuring NCCL](#configuring-nccl)
4. [Running Tests](#running-tests)
5. [Troubleshooting](#troubleshooting)
6. [Performance Optimization](#performance-optimization)

## Setting Up DigitalOcean Bare Metal Instances

### Step 1: Create GPU Instances

To get started with DigitalOcean bare metal GPU instances, you'll need to contact the sales team as these are specialized resources that require custom provisioning.

1. **Contact Sales Team**: Fill out the form at https://www.digitalocean.com/products/gradient/bare-metal-gpus?referrer=pdocs&utm_campaign=how-to-create-gradient-bare-metal-gpus#sales-form

2. **Review Documentation**: Refer to the official guide on creating bare metal GPU instances: https://docs.digitalocean.com/products/bare-metal-gpus/how-to/create/

3. **Instance Provisioning**: Once you've contacted sales and your requirements are approved:
   - The DigitalOcean team will create your bare metal GPU instances
   - You'll receive the necessary credentials and connection information
   - The instances will be configured with your specified GPU configuration

4. **What You'll Receive**:
   - SSH access credentials (private key or password)
   - IP addresses of your instances
   - Instance specifications and GPU configuration details
   - Network configuration information

> **Note**: Bare metal GPU instances are custom-provisioned resources and may take some time to set up. Plan accordingly and ensure you have your requirements clearly defined when contacting the sales team.

### Step 2: Initial Configuration

Once you receive your DigitalOcean bare metal GPU instances, follow these initial setup steps to verify connectivity and prepare the systems for NCCL installation.

#### 2.1 Connect to Your Instances

Use the SSH credentials provided by DigitalOcean to connect to each instance:

```bash
# Using SSH key (recommended)
ssh -i /path/to/your/private-key root@<instance-ip>

# Or using password if provided
ssh root@<instance-ip>
```

#### 2.2 Verify System Access

Confirm you can access all your instances and check basic system information:

```bash
# Check hostname and system info
hostname
uname -a

# Verify GPU detection (should show your GPUs even without drivers)
lspci | grep -i nvidia

# Check available memory and CPU
free -h
lscpu
```

#### 2.3 Update System Packages

Update the system packages on all instances before proceeding:

```bash
# Update Ubuntu system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools that may be needed
sudo apt install -y curl wget vim htop
```

#### 2.4 Verify Hostnames and Basic Networking

Check your current hostname and network configuration:

```bash
# Check current hostname (keep the existing one)
hostname

# Verify network interfaces - focus on bond0 interface
ip addr show bond0

# The output should show something like:
# bond0: <BROADCAST,MULTICAST,MASTER,UP,LOWER_UP> mtu 9000 qdisc noqueue state UP
#   inet 10.x.x.x/24 brd 10.x.x.255 scope global bond0
```

**Important Network Configuration Notes:**
- DigitalOcean bare metal instances use a **bond0** interface for high-performance networking
- Each instance has a **private IP** on the 10.x.x.x range - this is what you'll use for NCCL communication
- The public IP is managed through VPC Gateway with 1:1 NAT - **do not use public IPs for NCCL tests**

```bash
# Test network connectivity between nodes using PRIVATE IPs
# Replace with your actual private IPs from bond0 interface
ping 10.x.x.x
```

> **Note**: DigitalOcean assigns unique hostnames to each instance. Keep these existing hostnames as they help identify your specific bare metal instances.

> **Network Important**: Always use the **private IP addresses** from the bond0 interface for NCCL communication, not the public IPs. NCCL will utilize the backend network infrastructure for optimal GPU-to-GPU communication.

#### 2.5 Create Hosts File (Multi-Node Setup)

If you're setting up multiple instances, create a hosts file using the private IP addresses from bond0:

```bash
# Edit /etc/hosts on each node
sudo nano /etc/hosts

# Add entries using PRIVATE IPs from bond0 interface (example):
# 10.45.170.76  node1-hostname
# 10.45.170.77  node2-hostname  
# 10.45.170.78  node3-hostname
```

> **Note**: Use the actual hostnames from the `hostname` command and the **private IP addresses** from `ip addr show bond0`. These private IPs are essential for NCCL inter-node communication.

> **Note**: Make sure all instances are accessible and can communicate with each other before proceeding to the dependency installation steps. This basic setup ensures a solid foundation for the NCCL configuration that follows.

## Installing Dependencies

### Step 3: Install NVIDIA Drivers

Before installing new drivers, let's check what's currently installed on your DigitalOcean bare metal instances.

#### 3.1 Check Current NVIDIA and CUDA Installation

First, verify if NVIDIA drivers and CUDA are already installed:

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Check NVIDIA driver version (if installed)
cat /proc/driver/nvidia/version

# Check CUDA version (if installed)
nvcc --version

# Alternative CUDA check
cat /usr/local/cuda/version.txt

# Check what NVIDIA packages are installed via apt
dpkg -l | grep -i nvidia

# Check CUDA packages
dpkg -l | grep -i cuda
```

#### 3.2 Verify GPU Detection

Even without drivers, you should be able to see your GPUs:

```bash
# List all GPUs detected by the system
lspci | grep -i nvidia

# Get more detailed GPU information
lspci -v | grep -i nvidia -A 12
```

#### 3.3 Installation Resources

If drivers or CUDA are not available, refer to NVIDIA's official documentation:

**NVIDIA Driver Installation:**
- **Official Guide**: [NVIDIA Driver Installation Guide for Linux](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- **Ubuntu-specific**: [NVIDIA Driver Installation on Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

**CUDA Toolkit Installation:**
- **Official Guide**: [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- **Ubuntu Package Installation**: [CUDA Ubuntu Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)

> **Important**: After any driver or CUDA installation, reboot your instances and verify the installation using the commands from section 3.1 before proceeding to the next steps.

### Step 4: Install NCCL Tests

Now that you have NVIDIA drivers and CUDA working (verified in Step 3), let's install the NCCL tests that we'll use for benchmarking.

#### Install NCCL Tests

Install the NCCL tests repository and build the test suite:

```bash
# Install build tools if not already present
sudo apt install -y build-essential git

# Clone NCCL tests repository
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Build the tests
make

# Verify the build was successful
ls -la build/
```

The tests should now be available in the `build/` directory and ready for use in the testing steps.

> **Note**: If you encounter any build errors, ensure that your CUDA environment variables are properly set and that you have the necessary development tools installed.

## Configuring NCCL

### Step 5: Environment Setup

Now that NCCL is installed, let's configure the environment for optimal multi-GPU communication.

#### 5.1 Configure CUDA Environment Variables

Set up the necessary CUDA environment variables:

```bash
# Add CUDA to your environment (add to ~/.bashrc for persistence)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make changes persistent
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload environment
source ~/.bashrc
```

#### 5.2 Configure NCCL Environment Variables

Set up NCCL-specific environment variables for debugging and optimization:

```bash
# Enable NCCL debugging (useful for troubleshooting)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Configure network interface (use bond0 for DigitalOcean)
export NCCL_SOCKET_IFNAME=bond0

# Add to ~/.bashrc for persistence
echo 'export NCCL_DEBUG=INFO' >> ~/.bashrc
echo 'export NCCL_DEBUG_SUBSYS=ALL' >> ~/.bashrc
echo 'export NCCL_SOCKET_IFNAME=bond0' >> ~/.bashrc

# Reload environment
source ~/.bashrc
```

#### 5.3 Set Up SSH Keys for Multi-Node Communication

For multi-node testing, configure passwordless SSH between nodes:

```bash
# Generate SSH key pair (if not already present)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copy public key to other nodes (replace with actual hostnames/IPs)
ssh-copy-id root@<other-node-hostname>
ssh-copy-id root@<another-node-hostname>

# Test passwordless SSH connectivity
ssh <other-node-hostname> "hostname"
```

#### 5.4 Verify Environment Configuration

Test that your environment is properly configured:

```bash
# Verify CUDA environment
nvcc --version
echo $CUDA_HOME
echo $PATH | grep cuda

# Verify NCCL environment variables
echo $NCCL_DEBUG
echo $NCCL_SOCKET_IFNAME

# Test multi-node SSH connectivity
for node in <node1-hostname> <node2-hostname>; do
    echo "Testing SSH to $node:"
    ssh $node "hostname && nvidia-smi --query-gpu=name --format=csv,noheader,nounits"
done
```

> **Note**: Replace `<node-hostnames>` with your actual node hostnames from Step 2. Ensure all nodes can communicate via SSH before proceeding to the testing steps.

## Running Tests

### Step 6: Single Node Testing

Single node testing validates that all GPUs within one machine can communicate effectively through NCCL. This establishes your baseline performance before attempting multi-node scaling.

#### 6.1 Verify GPU Availability

First, confirm that all GPUs are detected and accessible:

```bash
# Check GPU count and status
nvidia-smi

# Verify NCCL can detect all GPUs
cd ~/nccl-tests
./build/all_reduce_perf -b 8 -e 128M -f 2 -g $(nvidia-smi -L | wc -l)
```

#### 6.2 Basic NCCL All-Reduce Test

Start with a basic all-reduce test to verify GPU-to-GPU communication:

```bash
# Navigate to NCCL tests directory
cd ~/nccl-tests

# Set NCCL to show only errors for cleaner output
export NCCL_DEBUG=ERROR

# Run basic all-reduce test with automatic GPU detection
# This tests communication between all available GPUs
./build/all_reduce_perf -b 1K -e 1G -f 2

# Run with specific GPU count (replace X with your GPU count)
./build/all_reduce_perf -b 1K -e 1G -f 2 -g X
```

**NCCL Debug Level Options:**
- `NCCL_DEBUG=ERROR` - Show only errors (recommended for clean output)
- `NCCL_DEBUG=WARN` - Show warnings and errors
- `NCCL_DEBUG=INFO` - Show informational messages (verbose)
- `NCCL_DEBUG=TRACE` - Show all debug information (very verbose)

```
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 2 minBytes 1024 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  15386 on am4g2r31bm1 device  0 [0000:19:00] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid  15386 on am4g2r31bm1 device  1 [0000:3b:00] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
        1024           256     float     sum      -1    15.75    0.07    0.07      0    14.46    0.07    0.07      0
        2048           512     float     sum      -1    14.16    0.14    0.14      0    13.56    0.15    0.15      0
        4096          1024     float     sum      -1    13.10    0.31    0.31      0    13.10    0.31    0.31      0
        8192          2048     float     sum      -1    12.32    0.66    0.66      0    12.92    0.63    0.63      0
       16384          4096     float     sum      -1    12.17    1.35    1.35      0    12.66    1.29    1.29      0
       32768          8192     float     sum      -1    12.68    2.58    2.58      0    12.61    2.60    2.60      0
       65536         16384     float     sum      -1    14.05    4.67    4.67      0    13.58    4.83    4.83      0
      131072         32768     float     sum      -1    15.62    8.39    8.39      0    15.44    8.49    8.49      0
      262144         65536     float     sum      -1    17.68   14.83   14.83      0    16.94   15.47   15.47      0
      524288        131072     float     sum      -1    17.23   30.43   30.43      0    17.50   29.96   29.96      0
     1048576        262144     float     sum      -1    26.30   39.88   39.88      0    26.02   40.29   40.29      0
     2097152        524288     float     sum      -1    30.67   68.38   68.38      0    30.24   69.34   69.34      0
     4194304       1048576     float     sum      -1    35.13  119.39  119.39      0    35.01  119.81  119.81      0
     8388608       2097152     float     sum      -1    47.86  175.26  175.26      0    46.89  178.88  178.88      0
    16777216       4194304     float     sum      -1    73.28  228.94  228.94      0    71.02  236.24  236.24      0
    33554432       8388608     float     sum      -1    127.7  262.68  262.68      0    126.0  266.33  266.33      0
    67108864      16777216     float     sum      -1    223.4  300.35  300.35      0    223.4  300.34  300.34      0
   134217728      33554432     float     sum      -1    415.0  323.41  323.41      0    415.3  323.19  323.19      0
   268435456      67108864     float     sum      -1    792.7  338.64  338.64      0    793.8  338.18  338.18      0
   536870912     134217728     float     sum      -1   1541.7  348.22  348.22      0   1543.5  347.82  347.82      0
  1073741824     268435456     float     sum      -1   2996.1  358.38  358.38      0   3006.4  357.15  357.15      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 125.437 
#
# Collective test concluded: all_reduce_perf
```

**Understanding These Results:**

**Test Configuration:**
- **2 GPUs**: Two NVIDIA H100 80GB HBM3 GPUs (device 0 and device 1)
- **Message Range**: Testing from 1KB to 1GB message sizes
- **Algorithm**: All-reduce with sum operation
- **Iterations**: 5 warmup iterations + 20 measurement iterations per test

**Performance Analysis:**

1. **Small Messages (1KB - 32KB)**: 
   - **Latency-dominated**: Time is relatively constant (~12-16μs)
   - **Low bandwidth**: 0.07 - 2.6 GB/s
   - **Communication overhead** is the main factor, not data transfer

2. **Medium Messages (64KB - 4MB)**:
   - **Transition zone**: Bandwidth grows from 4.67 GB/s to 119 GB/s
   - **Efficiency improves**: Better utilization of GPU interconnect
   - **Sweet spot** for many distributed training workloads

3. **Large Messages (8MB - 1GB)**:
   - **Bandwidth-dominated**: Peak performance of 358 GB/s
   - **Excellent scaling**: Shows the full capability of H100 interconnect
   - **Consistent performance**: Both out-of-place and in-place show similar results

**Key Performance Indicators:**
- ✅ **No errors**: `#wrong = 0` for all tests
- ✅ **High peak bandwidth**: 358 GB/s is excellent for H100 GPUs
- ✅ **Good scaling**: Bandwidth increases consistently with message size
- ✅ **Low latency**: ~12-16μs base latency is very good
- ✅ **Average bus bandwidth**: 125.437 GB/s across all message sizes

**What This Means:**
- Your GPU setup is working optimally
- Inter-GPU communication is performing at expected H100 levels
- Ready for distributed training workloads
- No hardware or configuration issues detected

### Step 7: Multi-Node Testing

Multi-node testing validates NCCL communication between GPUs across different machines over the network. This tests your distributed GPU cluster's ability to handle real-world workloads like distributed training.

#### 7.1 RoCE Performance Validation Test

This test validates RoCE (RDMA over Converged Ethernet) performance between two nodes using all 16 GPUs (8 GPUs per node) leveraging the Mellanox ConnectX adapters for high-performance inter-node GPU communication.

```bash
# On the MASTER node (first node)
cd ~/nccl-tests

# First, verify RoCE adapters are available
ibstat | grep -E "(CA|State|Rate)" | head -20

# Set clean output for RoCE performance testing
export NCCL_DEBUG=ERROR

# Configure NCCL to use RoCE transport
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_IB_HCA=mlx5

# Create hostfile for 2 nodes with 8 GPUs each
cat > ~/hostfile << EOF
<node1-hostname> slots=8
<node2-hostname> slots=8
EOF

# Install OpenMPI if not already present
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

# RoCE Comprehensive Performance Test
echo "=== RoCE Multi-Node Performance Validation (16 GPUs) ==="
mpirun -np 16 -hostfile ~/hostfile \
    --mca btl openib,self --mca btl_openib_allow_ib true \
    --bind-to none \
    -x NCCL_DEBUG=ERROR \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_IB_HCA=mlx5 \
    ./build/all_reduce_perf -b 1K -e 1G -f 2
```

**Expected RoCE Performance Characteristics:**

- **Latency**: 15-50μs for small messages (RDMA optimization)
- **Bandwidth**: 80-120 GB/s peak aggregate (400Gbps Mellanx adapters)
- **Scaling**: Linear scaling across 16 GPUs with RDMA efficiency
- **Transport**: Uses InfiniBand verbs over Ethernet (RoCE v2)

**RoCE Validation Criteria:**
- ✅ No communication errors (#wrong = 0)
- ✅ NCCL uses "NET/IB" transport (not "NET/Socket")
- ✅ Network latency <100μs with RDMA
- ✅ Peak bandwidth >50 GB/s aggregate
- ✅ Consistent scaling pattern across message sizes

> **Note**: Replace `<node1-hostname>` and `<node2-hostname>` with your actual node hostnames. This test validates that the Mellanox ConnectX RoCE adapters can efficiently handle GPU-to-GPU communication using RDMA over Ethernet, bypassing TCP/IP overhead for maximum performance.

#### 7.2 Expected RoCE Performance Analysis

#### 7.2 Expected RoCE Performance Analysis

RoCE performance with 16 GPUs across 2 nodes will show different characteristics compared to single-node performance:

**Performance Expectations:**
- **Base Latency**: 15-50μs with RDMA optimization (vs 12-16μs intra-node)
- **Peak Aggregate Bandwidth**: 50-120 GB/s (leveraging 400Gbps Mellanox adapters)
- **Per-GPU Effective Bandwidth**: 3-7 GB/s per GPU for inter-node communication
- **Scaling Pattern**: Near-linear scaling with RDMA efficiency

**RoCE-Specific Characteristics:**
- **RDMA Efficiency**: Direct GPU memory access across nodes without CPU involvement
- **Network Utilization**: Can approach 400Gbps per adapter theoretical limit
- **Transport Layer**: InfiniBand verbs over Ethernet (RoCE v2)
- **Message Size Impact**: 
  - Small messages (1KB-64KB): Latency-dominated, ~15-30μs with RDMA
  - Medium messages (256KB-16MB): Transition zone, RDMA efficiency gains
  - Large messages (32MB+): Bandwidth-dominated, peak RoCE performance

**Comparison to Single-Node:**
- **Latency Increase**: ~2-4x higher due to network hops (RDMA optimized)
- **Bandwidth Scaling**: ~3-6x higher aggregate than TCP-based networking
- **Efficiency Trade-off**: RDMA provides direct memory access vs GPU interconnect

**What Good RoCE Performance Looks Like:**
- ✅ Consistent latency <100μs across all message sizes
- ✅ Aggregate bandwidth >50 GB/s for large messages
- ✅ NCCL logs show "NET/IB" transport (not "NET/Socket")
- ✅ No RDMA completion errors or timeouts
- ✅ Linear scaling with message size growth

#### 7.3 Interpreting RoCE Results

#### 7.3 Interpreting RoCE Results

Key metrics to validate RoCE performance with 16 GPUs across 2 nodes:

**Primary RoCE Metrics:**
- **Inter-Node Latency**: Base latency should be <100μs for optimal RoCE with RDMA
- **Network Bandwidth Utilization**: Should approach 50-120 GB/s aggregate
- **RDMA Efficiency**: Direct GPU memory access without CPU copy overhead
- **Transport Validation**: NCCL must use "NET/IB" not "NET/Socket"

**Performance Analysis Framework:**
- **Latency Validation**: Measure small message (1KB-8KB) RDMA roundtrip time
- **Bandwidth Saturation**: Identify message size where Mellanox adapters peak
- **Scaling Efficiency**: Compare 16-GPU vs theoretical 8-GPU × 2 performance
- **RDMA Health**: Verify InfiniBand verbs are being used effectively

**RoCE Health Indicators:**
- ✅ **Low RDMA latency**: Consistent timing <100μs with direct memory access
- ✅ **Linear scaling**: Bandwidth increases predictably with message size
- ✅ **No RDMA errors**: All NCCL operations complete without InfiniBand faults
- ✅ **High memory efficiency**: Effective GPU memory bandwidth via RDMA

**Troubleshooting RoCE Issues:**
- **High latency (>100μs)**: Check RDMA stack, not falling back to TCP
- **Low bandwidth (<30 GB/s)**: Verify RoCE is enabled, InfiniBand verbs working
- **"NET/Socket" in logs**: NCCL falling back to TCP, check IB configuration
- **RDMA completion errors**: Validate Mellanox drivers and adapter status

#### 7.4 Troubleshooting RoCE Issues

#### 7.4 Troubleshooting RoCE Issues

Common RoCE-specific problems and diagnostic approaches:

```bash
# Problem: NCCL falling back to TCP instead of RoCE
# Solution: Verify InfiniBand/RoCE configuration and NCCL transport selection
export NCCL_DEBUG=INFO  # Temporarily enable verbose logging
# Look for "NET/IB" in NCCL output, NOT "NET/Socket"

# Problem: High inter-node latency (>100μs) with RoCE
# Solution: Check RDMA stack and Mellanox adapter status
ibstat  # Verify all adapters show "State: Active"
ibv_devinfo  # Check RDMA device capabilities

# Problem: Low RoCE bandwidth (<30 GB/s aggregate)
# Solution: Validate Mellanx adapter performance and RDMA functionality
# Test raw RDMA performance between nodes
ib_write_bw -d mlx5_0 -a  # On first node
ib_write_bw -d mlx5_0 <remote-node-ip>  # From second node

# Problem: NCCL cannot find InfiniBand/RoCE devices
# Solution: Check RDMA drivers and device visibility
lsmod | grep mlx  # Verify Mellanox drivers are loaded
ls /dev/infiniband/  # Should show uverbs* and rdma_cm devices
ibv_devices  # List available RDMA devices

# Problem: Inconsistent RoCE performance across runs
# Solution: Monitor Mellanox adapter statistics and errors
ibstat | grep -A 20 "mlx5_0"  # Check specific adapter status
cat /sys/class/infiniband/mlx5_0/ports/1/counters/*  # Monitor counters

# Problem: GPU memory allocation errors in multi-node RoCE
# Solution: Check GPU memory fragmentation and RDMA registration
for node in <node1> <node2>; do
    echo "=== GPU Memory Status on $node ==="
    ssh $node "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,nounits,noheader"
done

# Problem: MPI process coordination failures with RoCE
# Solution: Verify hostfile and OpenMPI InfiniBand configuration
cat ~/hostfile  # Ensure correct node names and slot counts
mpirun -np 16 -hostfile ~/hostfile hostname  # Test basic MPI coordination

# Advanced RoCE Diagnostics
# Check specific Mellanox adapter details and performance
sudo mlxconfig -d /dev/mst/mt4129_pciconf0 query  # Mellanox adapter config
perfquery -a  # InfiniBand performance counters (if available)

# Verify RoCE is active and not falling back to standard Ethernet
ethtool -k mlx5_0 | grep -i roce  # Should show RoCE capabilities
```

**RoCE Performance Optimization Tips:**
- Ensure NCCL uses InfiniBand transport (look for "NET/IB" in logs)
- Verify all `mlx5_*` adapters show "State: Active" in `ibstat`
- Check that RoCE Priority Flow Control (PFC) is configured on switches
- Monitor Mellanox adapter statistics for errors or retransmissions
- Validate that RDMA verbs are working with `ibv_devinfo`

> **Important**: Successful RoCE testing with Mellanox ConnectX adapters validates that your DigitalOcean bare metal setup can handle production distributed training workloads. The RoCE backend with RDMA provides the low-latency, high-bandwidth GPU-to-GPU communication essential for scaling deep learning across multiple nodes at near-native InfiniBand performance levels.

### Step 8: Performance Benchmarking

*Instructions to be added*

## Troubleshooting

### Common Issues

*Common problems and solutions to be added*

### Debugging Commands

*Useful debugging commands to be added*

## Performance Optimization

### Best Practices

*Performance optimization tips to be added*

### Monitoring and Profiling

*Monitoring tools and techniques to be added*

## Additional Resources

- [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [DigitalOcean GPU Documentation](https://docs.digitalocean.com/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## Contributing

Contributions to this guide are welcome! Please feel free to submit issues or pull requests.

## License

This guide is provided under the MIT License.
