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

*Instructions to be added*

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
