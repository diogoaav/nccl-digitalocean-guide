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

# Run basic all-reduce test with automatic GPU detection
# This tests communication between all available GPUs
./build/all_reduce_perf -b 1K -e 1G -f 2

# Run with specific GPU count (replace X with your GPU count)
./build/all_reduce_perf -b 1K -e 1G -f 2 -g X
```

**Expected Output Example:**
```
# nThread 1 nGpus X test 0
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           1K             256     float     sum      -1    15.2    0.07    0.12      0    15.1    0.07    0.12      0
           2K             512     float     sum      -1    15.4    0.13    0.23      0    15.3    0.13    0.23      0
           4K            1024     float     sum      -1    15.6    0.26    0.45      0    15.5    0.26    0.45      0
```

#### 6.3 Comprehensive Single Node Test Suite

Run multiple NCCL operations to thoroughly test GPU communication:

```bash
# All-Reduce test (most common operation)
echo "=== Running All-Reduce Test ==="
./build/all_reduce_perf -b 8 -e 128M -f 2

# All-Gather test
echo "=== Running All-Gather Test ==="
./build/all_gather_perf -b 8 -e 128M -f 2

# Broadcast test
echo "=== Running Broadcast Test ==="
./build/broadcast_perf -b 8 -e 128M -f 2

# Reduce-Scatter test
echo "=== Running Reduce-Scatter Test ==="
./build/reduce_scatter_perf -b 8 -e 128M -f 2
```

#### 6.4 Bandwidth and Latency Analysis

Test different message sizes to analyze bandwidth and latency characteristics:

```bash
# Test small messages (latency-bound)
echo "=== Testing Small Messages (Latency) ==="
./build/all_reduce_perf -b 4 -e 8K -f 2

# Test medium messages
echo "=== Testing Medium Messages ==="
./build/all_reduce_perf -b 32K -e 1M -f 2

# Test large messages (bandwidth-bound)
echo "=== Testing Large Messages (Bandwidth) ==="
./build/all_reduce_perf -b 16M -e 512M -f 2
```

#### 6.5 Performance Validation

Create a simple script to run comprehensive single-node validation:

```bash
# Create a single-node test script
cat > ~/single_node_test.sh << 'EOF'
#!/bin/bash

echo "=== NCCL Single Node Performance Test ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Count: $(nvidia-smi -L | wc -l)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo ""

cd ~/nccl-tests

# Set NCCL debug level for detailed output
export NCCL_DEBUG=INFO

# Test 1: Quick validation
echo "=== Test 1: Quick All-Reduce Validation ==="
./build/all_reduce_perf -b 1M -e 16M -f 2 -g $(nvidia-smi -L | wc -l)
echo ""

# Test 2: Latency test
echo "=== Test 2: Latency Test (Small Messages) ==="
./build/all_reduce_perf -b 4 -e 1K -f 2
echo ""

# Test 3: Bandwidth test
echo "=== Test 3: Bandwidth Test (Large Messages) ==="
./build/all_reduce_perf -b 32M -e 128M -f 2
echo ""

# Test 4: Multiple operations
echo "=== Test 4: Multiple NCCL Operations ==="
for op in all_reduce all_gather broadcast reduce_scatter; do
    echo "Testing $op..."
    ./build/${op}_perf -b 1M -e 4M -f 2
done

echo "=== Single Node Testing Complete ==="
EOF

# Make script executable and run it
chmod +x ~/single_node_test.sh
~/single_node_test.sh
```

#### 6.6 Interpreting Results

Understanding the output from NCCL tests:

- **size (B)**: Message size in bytes
- **count (elements)**: Number of elements being processed
- **time (us)**: Time taken in microseconds
- **algbw (GB/s)**: Algorithm bandwidth (actual data transfer rate)
- **busbw (GB/s)**: Bus bandwidth (includes communication overhead)
- **#wrong**: Number of errors (should be 0)

**Good Performance Indicators:**
- No errors (#wrong = 0)
- Consistent timing across runs
- High bandwidth for large messages (>50GB/s for modern GPUs)
- Low latency for small messages (<20μs)

#### 6.7 Troubleshooting Single Node Issues

If you encounter issues, try these debugging steps:

```bash
# Check for GPU memory issues
nvidia-smi

# Run with verbose debugging
export NCCL_DEBUG=TRACE
./build/all_reduce_perf -b 1M -e 1M -f 1

# Test with fewer GPUs to isolate issues
./build/all_reduce_perf -b 1M -e 1M -f 1 -g 2

# Check CUDA context creation
./build/all_reduce_perf -b 1M -e 1M -f 1 -w 5 -n 5
```

> **Note**: Single node testing should complete without errors and show reasonable performance numbers. If you see consistent errors or very low bandwidth (<10GB/s), check your GPU installation and CUDA environment before proceeding to multi-node testing.

### Step 7: Multi-Node Testing

*Instructions to be added*

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
