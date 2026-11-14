# Testing Data / Testing Data

### Left Panel
**p2pBandwidthLatencyTest (NVLink P2P)**  
**GPU Burn Stress Stability / NVBandwidth (Upstream Bandwidth Test)**

Multiple terminal screenshots showing:
- Unidirectional / Bidirectional P2P Bandwidth Matrix (GB/s)
- P2P Enabled Latency Matrix (µs)
- Various CUDA / GPU test outputs
- Bottom terminal: `nvbandwidth` test results (host-to-device, device-to-host, device-to-device) with Tesla V100-SXM2-16GB devices detected

### Right Panel
**nvidia-smi nvlink -s**  
**nvidia-smi -q -d ecc**

- Two Tesla V100-SXM2-16GB GPUs detected  
- All 6 NVLink links per GPU reporting **25.781 GB/s**  
- ECC Mode: Enabled  
- Current / Pending: Enabled  
- Zero single-bit and double-bit errors across all categories  
- Driver Version: 575.57.08  
- CUDA Version: 12.9  

**Page numbers:** -3- / -4-

---

# Power / FAN Guide | LED Guide

### Left Side – Power / FAN Interface Description

**Callouts on card photo:**
- **02** GPU 2x4Pin ×4  
- **04** FAN 4Pin ×4  

**Pinout diagrams:**
- PCIe 6-pin and 8-pin power connectors (GND / +12V)  
- FAN connector: FAN PWM / FAN IN / FAN PWR / GND  

**Power Supply Description:**  
Based on V100 300W full-load power, it is necessary to ensure that 2 independent GPU power cables are connected. The 1-to-4 power supply method is absolutely rejected.

**Fan Description:**  
The fan speed control of this product is based on the current GPU load. The greater the load, the faster the fan speed.

### Right Side – LED Description

**LED Callouts:**
- **01** GPU Over-temperature Indicator Light  
- **02** GPU Plug-in Indicator Light  
- **03** GPU Power-on Indicator Light  
- **04** GPU Power OK Light  
- **05** Backplane-on Indicator Light  
- **06** Backplane Standby Indicator Light  

| Serial Number | Description                              | Normal      | Abnormal   |
|---------------|------------------------------------------|-------------|------------|
| 1#            | GPU Over-temperature, Uninstalled and Abnormal | NA         | Red Light  |
| 2#            | GPU Installed                            | Green Light | NA         |
| 3#            | GPU Power Supply Has Been Connected      | Green Light | NA         |
| 4#            | GPU Core Power Supply Working Normally   | Green Light | NA         |
| 5#            | Host Power-on Indicator Light            | Green Light | NA         |
| 6#            | Host Standby Indicator Light             | Blue Light  | NA         |

**Page numbers:** -3- / -4-

---

# Basic Information | Product View

### Left Side – Basic Information

#### Basic Information
| Item                    | Value                          |
|-------------------------|--------------------------------|
| Material Name           | SXM2 PCIe Riser Card           |
| Material Model          | RTE16SXM2B-S                   |
| Material Number         | —                              |
| Specification           | 2 × SXM2 GPU                   |
| Bus Type                | PCIe x16                       |
| PCIe Bandwidth          | PCIe 3.0 8GT/s                 |
| Power Supply            | 4 × GPU 8Pin                   |
| Port Type               | PCIe Gold Finger               |
| Upstream Port           | PCIe 3.0 x16                   |
| Fan Ports               | 4                              |
| Downstream Port         | SXM2 PCIe x8                   |
| Specification Protocol  | PCIe 3.0                       |
| NVLink                  | 6 channels 300GB/s             |

#### Component Specifications
| Item                    | Value                                      |
|-------------------------|--------------------------------------------|
| Typical Power Consumption | 650W                                     |
| PCB Thickness           | 1.6mm                                      |
| PCB Material            | FR4                                        |
| PCB Version             | Rev 5.2                                    |
| PCB Color               | Black                                      |
| Physical Dimensions     | 315(L) × 134.5(W) mm                       |
| PCB Copper Foil         | 1oz                                        |
| Capacitor Specification | Tantalum solid capacitors, Samsung MLCC    |
| SXM Slot Specification  | 30u                                        |
| Baffle Specification    | Full Height                                |
| Fan Temperature Control | Power Detection Control                    |

### Right Side – Product View
**1. Top View**

**Callouts:**
- **01** SXM2 GPU  
- **02** GPU 2x4Pin  
- **03** PCIe x16  
- **04** FAN 4Pin  
- **05** Status Indicator Light  