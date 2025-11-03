#!/bin/bash
# QEMU Firmware Emulation Setup Script
# Automated QEMU configuration for firmware emulation with network bridging
# Version: 1.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_BIN=""
ARCH=""
FIRMWARE_ROOT=""
NETWORK_BRIDGE="br0"
SNAPSHOT_MODE=false
MONITOR_TRAFFIC=false
PCAP_FILE=""
LOG_FILE="qemu-emulation.log"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

usage() {
    cat <<EOF
QEMU Firmware Emulation Setup

Usage: $0 <firmware-root> [OPTIONS]

Arguments:
    firmware-root       Path to extracted firmware filesystem root

Options:
    --arch ARCH        Architecture (mips, mipsel, arm, armeb, aarch64, x86, x86_64)
                       If not specified, auto-detected from binaries
    --network-bridge BR Network bridge interface (default: br0)
    --snapshot         Enable QEMU snapshot mode (safe execution)
    --monitor-traffic  Capture network traffic to PCAP file
    --log-file FILE    Log file path (default: qemu-emulation.log)
    --help             Show this help message

Examples:
    $0 ./extracted/squashfs-root/
    $0 ./extracted/squashfs-root/ --arch mipsel --snapshot
    $0 ./extracted/squashfs-root/ --network-bridge virbr0 --monitor-traffic

EOF
    exit 0
}

detect_architecture() {
    local firmware_root="$1"
    log "Detecting architecture from binaries..."

    # Check common binaries for architecture
    local test_binaries=(
        "$firmware_root/bin/busybox"
        "$firmware_root/sbin/init"
        "$firmware_root/usr/bin/httpd"
        "$firmware_root/bin/sh"
    )

    for binary in "${test_binaries[@]}"; do
        if [[ -f "$binary" ]]; then
            local file_output=$(file "$binary")
            log "Analyzing: $binary"
            log "File output: $file_output"

            if [[ $file_output =~ MIPS.*LSB ]]; then
                echo "mipsel"
                return 0
            elif [[ $file_output =~ MIPS.*MSB ]]; then
                echo "mips"
                return 0
            elif [[ $file_output =~ ARM.*EABI ]]; then
                if [[ $file_output =~ aarch64 ]]; then
                    echo "aarch64"
                else
                    echo "arm"
                fi
                return 0
            elif [[ $file_output =~ x86-64 ]]; then
                echo "x86_64"
                return 0
            elif [[ $file_output =~ 80386 ]]; then
                echo "x86"
                return 0
            fi
        fi
    done

    error "Could not detect architecture from binaries"
    return 1
}

setup_qemu_binary() {
    local arch="$1"

    case "$arch" in
        mipsel)
            QEMU_BIN="qemu-mipsel-static"
            ;;
        mips)
            QEMU_BIN="qemu-mips-static"
            ;;
        arm)
            QEMU_BIN="qemu-arm-static"
            ;;
        armeb)
            QEMU_BIN="qemu-armeb-static"
            ;;
        aarch64)
            QEMU_BIN="qemu-aarch64-static"
            ;;
        x86)
            QEMU_BIN="qemu-i386-static"
            ;;
        x86_64)
            QEMU_BIN="qemu-x86_64-static"
            ;;
        *)
            error "Unsupported architecture: $arch"
            return 1
            ;;
    esac

    # Check if QEMU binary exists
    if ! command -v "$QEMU_BIN" &> /dev/null; then
        error "$QEMU_BIN not found. Install: sudo apt install qemu-user-static"
        return 1
    fi

    log "Using QEMU binary: $QEMU_BIN"
    return 0
}

setup_library_paths() {
    local firmware_root="$1"

    log "Setting up library paths..."

    # Common library paths in firmware
    local lib_paths=(
        "$firmware_root/lib"
        "$firmware_root/usr/lib"
        "$firmware_root/lib64"
        "$firmware_root/usr/lib64"
    )

    local ld_library_path=""
    for lib_path in "${lib_paths[@]}"; do
        if [[ -d "$lib_path" ]]; then
            ld_library_path="$lib_path:$ld_library_path"
            log "Added library path: $lib_path"
        fi
    done

    export LD_LIBRARY_PATH="$ld_library_path"
}

copy_qemu_to_chroot() {
    local firmware_root="$1"
    local qemu_path=$(command -v "$QEMU_BIN")

    log "Copying QEMU binary to chroot..."

    # Copy QEMU binary to firmware root
    sudo cp "$qemu_path" "$firmware_root/usr/bin/"
    sudo chmod +x "$firmware_root/usr/bin/$QEMU_BIN"

    log "QEMU binary copied to $firmware_root/usr/bin/$QEMU_BIN"
}

setup_network_bridge() {
    local bridge="$1"

    log "Setting up network bridge: $bridge"

    # Check if bridge exists
    if ! ip link show "$bridge" &> /dev/null; then
        warn "Bridge $bridge does not exist. Creating..."

        sudo ip link add name "$bridge" type bridge
        sudo ip addr add 192.168.1.1/24 dev "$bridge"
        sudo ip link set "$bridge" up

        log "Bridge $bridge created with IP 192.168.1.1/24"
    else
        log "Bridge $bridge already exists"
    fi

    # Enable IP forwarding
    sudo sysctl -w net.ipv4.ip_forward=1 > /dev/null

    # Setup NAT for internet access
    local external_if=$(ip route | grep default | awk '{print $5}' | head -n1)
    sudo iptables -t nat -A POSTROUTING -o "$external_if" -j MASQUERADE
    sudo iptables -A FORWARD -i "$bridge" -o "$external_if" -j ACCEPT
    sudo iptables -A FORWARD -i "$external_if" -o "$bridge" -m state --state RELATED,ESTABLISHED -j ACCEPT

    log "NAT configured for bridge $bridge via $external_if"
}

start_traffic_capture() {
    local interface="$1"

    PCAP_FILE="qemu-traffic-$(date +%Y%m%d-%H%M%S).pcap"

    log "Starting traffic capture on $interface -> $PCAP_FILE"

    sudo tcpdump -i "$interface" -w "$PCAP_FILE" &> /dev/null &
    local tcpdump_pid=$!

    log "tcpdump PID: $tcpdump_pid"
    echo "$tcpdump_pid" > tcpdump.pid
}

emulate_binary() {
    local firmware_root="$1"
    local binary_path="$2"
    shift 2
    local args=("$@")

    log "Emulating binary: $binary_path"
    log "Arguments: ${args[*]}"

    # Setup library paths
    setup_library_paths "$firmware_root"

    # Build chroot command
    local cmd=(
        sudo chroot "$firmware_root"
        "$QEMU_BIN"
    )

    if [[ "$SNAPSHOT_MODE" = true ]]; then
        cmd+=("-snapshot")
        log "Snapshot mode enabled (no persistent changes)"
    fi

    cmd+=("$binary_path" "${args[@]}")

    log "Executing: ${cmd[*]}"

    # Execute with timeout
    timeout 300s "${cmd[@]}" || {
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            warn "Emulation timed out after 300 seconds"
        else
            error "Emulation failed with exit code: $exit_code"
        fi
    }
}

full_system_emulation() {
    local firmware_root="$1"

    log "Starting full system emulation..."

    # Find kernel image
    local kernel_image=""
    for possible_kernel in "$firmware_root/boot/vmlinux" "$firmware_root/vmlinux" "./vmlinux"; do
        if [[ -f "$possible_kernel" ]]; then
            kernel_image="$possible_kernel"
            break
        fi
    done

    if [[ -z "$kernel_image" ]]; then
        error "Kernel image not found. Cannot perform full system emulation."
        return 1
    fi

    log "Using kernel: $kernel_image"

    # Create disk image from firmware root
    local disk_image="firmware-disk.qcow2"
    qemu-img create -f qcow2 "$disk_image" 1G

    # Format and populate disk
    log "Creating disk image from firmware root..."

    # Use appropriate QEMU system emulator
    local qemu_system="qemu-system-${ARCH}"

    if ! command -v "$qemu_system" &> /dev/null; then
        error "$qemu_system not found. Install: sudo apt install qemu-system"
        return 1
    fi

    # Boot with QEMU system emulator
    local qemu_cmd=(
        "$qemu_system"
        -M malta  # Generic MIPS/ARM machine
        -kernel "$kernel_image"
        -hda "$disk_image"
        -append "root=/dev/sda1 console=ttyS0"
        -nographic
    )

    if [[ "$SNAPSHOT_MODE" = true ]]; then
        qemu_cmd+=(-snapshot)
    fi

    if [[ "$MONITOR_TRAFFIC" = true ]]; then
        qemu_cmd+=(-netdev "bridge,id=net0,br=$NETWORK_BRIDGE")
        qemu_cmd+=(-device "virtio-net-pci,netdev=net0")
    fi

    log "Starting QEMU system emulation: ${qemu_cmd[*]}"

    "${qemu_cmd[@]}" &
    local qemu_pid=$!

    log "QEMU system emulation PID: $qemu_pid"
    echo "$qemu_pid" > qemu-system.pid
}

cleanup() {
    log "Cleaning up..."

    # Stop tcpdump
    if [[ -f tcpdump.pid ]]; then
        local tcpdump_pid=$(cat tcpdump.pid)
        sudo kill "$tcpdump_pid" 2> /dev/null || true
        rm tcpdump.pid
        log "Stopped traffic capture"
    fi

    # Stop QEMU system emulation
    if [[ -f qemu-system.pid ]]; then
        local qemu_pid=$(cat qemu-system.pid)
        kill "$qemu_pid" 2> /dev/null || true
        rm qemu-system.pid
        log "Stopped QEMU system emulation"
    fi
}

trap cleanup EXIT

main() {
    log "========================================="
    log "QEMU Firmware Emulation Setup v1.0.0"
    log "========================================="

    # Parse arguments
    if [[ $# -eq 0 ]]; then
        usage
    fi

    FIRMWARE_ROOT="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --arch)
                ARCH="$2"
                shift 2
                ;;
            --network-bridge)
                NETWORK_BRIDGE="$2"
                shift 2
                ;;
            --snapshot)
                SNAPSHOT_MODE=true
                shift
                ;;
            --monitor-traffic)
                MONITOR_TRAFFIC=true
                shift
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            --help)
                usage
                ;;
            *)
                error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate firmware root
    if [[ ! -d "$FIRMWARE_ROOT" ]]; then
        error "Firmware root not found: $FIRMWARE_ROOT"
        exit 1
    fi

    # Auto-detect architecture if not specified
    if [[ -z "$ARCH" ]]; then
        ARCH=$(detect_architecture "$FIRMWARE_ROOT")
        if [[ $? -ne 0 ]]; then
            error "Architecture detection failed"
            exit 1
        fi
    fi

    log "Architecture: $ARCH"

    # Setup QEMU binary
    setup_qemu_binary "$ARCH"

    # Copy QEMU to chroot
    copy_qemu_to_chroot "$FIRMWARE_ROOT"

    # Setup network if monitoring enabled
    if [[ "$MONITOR_TRAFFIC" = true ]]; then
        setup_network_bridge "$NETWORK_BRIDGE"
        start_traffic_capture "$NETWORK_BRIDGE"
    fi

    # Start emulation
    log "Firmware emulation environment ready"
    log "You can now execute binaries from the firmware root:"
    log "  sudo chroot $FIRMWARE_ROOT $QEMU_BIN /bin/busybox"
    log "  sudo chroot $FIRMWARE_ROOT $QEMU_BIN /usr/sbin/httpd"

    # Keep script running for monitoring
    if [[ "$MONITOR_TRAFFIC" = true ]]; then
        log "Press Ctrl+C to stop traffic capture and exit"
        wait
    fi
}

main "$@"
