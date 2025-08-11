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
4. [Network Configuration](#network-configuration)
5. [Running Tests](#running-tests)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

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

#### 3.3 Installation Decision and Next Steps

Based on the output from sections 3.1 and 3.2, determine your next steps:

**Scenario 1: Both NVIDIA Drivers and CUDA are installed and working**
- `nvidia-smi` shows GPU information successfully
- `nvcc --version` shows CUDA version
- **Action**: Skip to Step 4 (Install NCCL) - you're ready to proceed

**Scenario 2: Drivers or CUDA not present or not working properly**
- `nvidia-smi` command not found, fails, or returns errors
- `nvcc --version` fails or returns errors
- Commands exist but show inconsistent information
- **Action**: Install them using step 3.4 below

#### 3.4 Installation Resources

If you need to install NVIDIA drivers or CUDA, refer to NVIDIA's official documentation:

**NVIDIA Driver Installation:**
- **Official Guide**: [NVIDIA Driver Installation Guide for Linux](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- **Ubuntu-specific**: [NVIDIA Driver Installation on Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

**CUDA Toolkit Installation:**
- **Official Guide**: [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- **Ubuntu Package Installation**: [CUDA Ubuntu Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)

> **Important**: After any driver or CUDA installation, reboot your instances and verify the installation using the commands from section 3.1 before proceeding to the next steps.

### Step 4: Install NCCL

Now that you have NVIDIA drivers and CUDA working (verified in Step 3), let's install NCCL for multi-GPU communication.

#### 4.1 Check if NCCL is Already Installed

First, check if NCCL is already available on your system:

```bash
# Check for NCCL library files
find /usr -name "*nccl*" 2>/dev/null

# Check for NCCL in common locations
ls /usr/lib/x86_64-linux-gnu/libnccl* 2>/dev/null
ls /usr/local/cuda/lib64/libnccl* 2>/dev/null

# Check NCCL version (if installed)
cat /usr/include/nccl.h | grep NCCL_VERSION_CODE 2>/dev/null
```

#### 4.2 Install NCCL

If NCCL is not installed or you need a specific version, install it using one of these methods:

**Method 1: Install via APT (Recommended for Ubuntu)**

```bash
# Add NVIDIA package repository (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install NCCL runtime and development packages
sudo apt install libnccl2 libnccl-dev
```

**Method 2: Download from NVIDIA (Alternative)**

Visit [NVIDIA NCCL Download Page](https://developer.nvidia.com/nccl/nccl-download) and follow the instructions for your specific CUDA version.

#### 4.3 Verify NCCL Installation

After installation, verify NCCL is properly installed:

```bash
# Check NCCL library files are present
ls -la /usr/lib/x86_64-linux-gnu/libnccl*

# Verify NCCL version
cat /usr/include/nccl.h | grep NCCL_VERSION

# Check if NCCL can be found by the linker
ldconfig -p | grep nccl
```

#### 4.4 Install NCCL Tests (Important for our guide)

Install the NCCL tests that we'll use for benchmarking:

```bash
# Clone NCCL tests repository
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Build the tests
make

# Verify the build was successful
ls -la build/
```

The tests should now be available in the `build/` directory and ready for use in the testing steps.

> **Note**: If you encounter any build errors, ensure that your CUDA environment variables are properly set and that you have development tools installed (`build-essential` package).

### Step 5: Install NCCL

*Instructions to be added*

## Configuring NCCL

### Step 6: Environment Setup

*Instructions to be added*

### Step 7: Network Configuration

*Instructions to be added*

## Network Configuration

### Step 8: Configure RoCE (RDMA over Converged Ethernet)

*Instructions to be added*

### Step 9: Optimize Network Settings

*Instructions to be added*

## Running Tests

### Step 10: Single Node Testing

*Instructions to be added*

### Step 11: Multi-Node Testing

*Instructions to be added*

### Step 12: Performance Benchmarking

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
