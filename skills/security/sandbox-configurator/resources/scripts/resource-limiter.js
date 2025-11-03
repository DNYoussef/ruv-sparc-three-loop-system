#!/usr/bin/env node
/**
 * Resource Limiter - CPU and Memory Constraints for Sandboxes
 * ============================================================
 *
 * Production-ready script for enforcing CPU, memory, and process limits
 * on sandbox environments using cgroups v2.
 *
 * Features:
 * - cgroups v2 integration for resource control
 * - CPU quota enforcement (percentage or cores)
 * - Memory limit with swap control
 * - PID limit to prevent fork bombs
 * - I/O throttling (disk read/write)
 * - Real-time monitoring and alerting
 * - Graceful degradation on limit breaches
 * - Comprehensive error handling
 *
 * Usage:
 *   node resource-limiter.js --cpu 2 --memory 2048 --pids 100
 *   node resource-limiter.js --profile production --monitor
 *   node resource-limiter.js --cleanup --cgroup my-sandbox
 *
 * Requirements:
 *   - Node.js 18+
 *   - Linux with cgroups v2 support
 *   - Optional: root privileges for cgroup management
 *
 * @author Claude Code Sandbox Configurator Skill
 * @version 1.0.0
 * @license MIT
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { program } = require('commander');

// ANSI color codes
const colors = {
    reset: '\x1b[0m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m'
};

// Logging utilities
const logger = {
    info: (msg) => console.log(`${colors.blue}[INFO]${colors.reset} ${msg}`),
    success: (msg) => console.log(`${colors.green}[SUCCESS]${colors.reset} ${msg}`),
    warning: (msg) => console.warn(`${colors.yellow}[WARNING]${colors.reset} ${msg}`),
    error: (msg) => console.error(`${colors.red}[ERROR]${colors.reset} ${msg}`),
    debug: (msg) => {
        if (process.env.DEBUG) {
            console.log(`[DEBUG] ${msg}`);
        }
    }
};

/**
 * Resource limit profiles
 */
const PROFILES = {
    minimal: {
        cpu: 0.5,
        memory_mb: 512,
        pids: 50,
        io_weight: 10
    },
    development: {
        cpu: 2,
        memory_mb: 2048,
        pids: 100,
        io_weight: 100
    },
    production: {
        cpu: 4,
        memory_mb: 4096,
        pids: 200,
        io_weight: 500
    },
    heavy: {
        cpu: 8,
        memory_mb: 8192,
        pids: 500,
        io_weight: 1000
    }
};

/**
 * Resource Limiter Class
 * Manages cgroups v2 resource constraints
 */
class ResourceLimiter {
    /**
     * Initialize resource limiter
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        this.cgroupPath = options.cgroupPath || '/sys/fs/cgroup';
        this.cgroupName = options.cgroupName || 'claude-sandbox';
        this.fullPath = path.join(this.cgroupPath, this.cgroupName);

        this.limits = {
            cpu: options.cpu || 2,
            memory_mb: options.memory_mb || 2048,
            pids: options.pids || 100,
            io_weight: options.io_weight || 100
        };

        this.monitoring = {
            interval: null,
            alertThreshold: 0.9 // 90% of limit
        };

        logger.info('Resource Limiter initialized');
        logger.debug(`Cgroup path: ${this.fullPath}`);
    }

    /**
     * Check if cgroups v2 is available
     * @returns {boolean}
     */
    isCgroupV2Available() {
        try {
            const cgroupVersion = fs.readFileSync('/proc/self/cgroup', 'utf8');
            return cgroupVersion.includes('0::');
        } catch (error) {
            logger.error(`Failed to check cgroup version: ${error.message}`);
            return false;
        }
    }

    /**
     * Create cgroup for sandbox
     * @returns {boolean} Success status
     */
    createCgroup() {
        try {
            if (!this.isCgroupV2Available()) {
                logger.error('cgroups v2 not available. Upgrade to Linux kernel 5.0+');
                return false;
            }

            // Create cgroup directory
            if (!fs.existsSync(this.fullPath)) {
                fs.mkdirSync(this.fullPath, { recursive: true });
                logger.success(`Created cgroup: ${this.cgroupName}`);
            } else {
                logger.info(`Cgroup already exists: ${this.cgroupName}`);
            }

            return true;
        } catch (error) {
            logger.error(`Failed to create cgroup: ${error.message}`);
            return false;
        }
    }

    /**
     * Set CPU limit
     * @returns {boolean} Success status
     */
    setCPULimit() {
        try {
            const cpuMaxFile = path.join(this.fullPath, 'cpu.max');

            // Convert CPU cores to quota
            // cpu.max format: "quota period"
            // Example: "200000 100000" = 2.0 CPU cores
            const period = 100000; // 100ms in microseconds
            const quota = Math.floor(this.limits.cpu * period);

            fs.writeFileSync(cpuMaxFile, `${quota} ${period}`);

            logger.success(`CPU limit set: ${this.limits.cpu} cores`);
            logger.debug(`CPU quota: ${quota}µs / ${period}µs`);

            return true;
        } catch (error) {
            logger.error(`Failed to set CPU limit: ${error.message}`);
            return false;
        }
    }

    /**
     * Set memory limit
     * @returns {boolean} Success status
     */
    setMemoryLimit() {
        try {
            const memoryMaxFile = path.join(this.fullPath, 'memory.max');
            const memorySwapFile = path.join(this.fullPath, 'memory.swap.max');

            const memoryBytes = this.limits.memory_mb * 1024 * 1024;

            // Set memory limit
            fs.writeFileSync(memoryMaxFile, memoryBytes.toString());

            // Disable swap (set swap.max to 0)
            fs.writeFileSync(memorySwapFile, '0');

            logger.success(`Memory limit set: ${this.limits.memory_mb} MB (swap disabled)`);

            return true;
        } catch (error) {
            logger.error(`Failed to set memory limit: ${error.message}`);
            return false;
        }
    }

    /**
     * Set PID limit to prevent fork bombs
     * @returns {boolean} Success status
     */
    setPIDLimit() {
        try {
            const pidsMaxFile = path.join(this.fullPath, 'pids.max');

            fs.writeFileSync(pidsMaxFile, this.limits.pids.toString());

            logger.success(`PID limit set: ${this.limits.pids} processes`);

            return true;
        } catch (error) {
            logger.error(`Failed to set PID limit: ${error.message}`);
            return false;
        }
    }

    /**
     * Set I/O weight (relative I/O priority)
     * @returns {boolean} Success status
     */
    setIOWeight() {
        try {
            const ioWeightFile = path.join(this.fullPath, 'io.weight');

            // I/O weight range: 1-10000 (default: 100)
            const weight = Math.min(10000, Math.max(1, this.limits.io_weight));

            fs.writeFileSync(ioWeightFile, weight.toString());

            logger.success(`I/O weight set: ${weight}`);

            return true;
        } catch (error) {
            logger.warning(`Failed to set I/O weight: ${error.message}`);
            // I/O weight is optional, don't fail
            return true;
        }
    }

    /**
     * Apply all resource limits
     * @returns {boolean} Success status
     */
    applyLimits() {
        logger.info('Applying resource limits...');

        if (!this.createCgroup()) {
            return false;
        }

        const results = [
            this.setCPULimit(),
            this.setMemoryLimit(),
            this.setPIDLimit(),
            this.setIOWeight()
        ];

        if (results.every(r => r)) {
            logger.success('All resource limits applied successfully');
            this.printLimitsSummary();
            return true;
        } else {
            logger.error('Some resource limits failed to apply');
            return false;
        }
    }

    /**
     * Move current process to cgroup
     * @param {number} pid - Process ID to move (default: current process)
     * @returns {boolean} Success status
     */
    moveProcessToCgroup(pid = process.pid) {
        try {
            const procsFile = path.join(this.fullPath, 'cgroup.procs');

            fs.writeFileSync(procsFile, pid.toString());

            logger.success(`Process ${pid} moved to cgroup ${this.cgroupName}`);

            return true;
        } catch (error) {
            logger.error(`Failed to move process to cgroup: ${error.message}`);
            return false;
        }
    }

    /**
     * Get current resource usage
     * @returns {Object} Resource usage stats
     */
    getCurrentUsage() {
        try {
            const stats = {};

            // CPU usage
            const cpuStatFile = path.join(this.fullPath, 'cpu.stat');
            if (fs.existsSync(cpuStatFile)) {
                const cpuStat = fs.readFileSync(cpuStatFile, 'utf8');
                const usageMatch = cpuStat.match(/usage_usec (\d+)/);
                if (usageMatch) {
                    stats.cpu_usage_usec = parseInt(usageMatch[1]);
                }
            }

            // Memory usage
            const memoryCurrentFile = path.join(this.fullPath, 'memory.current');
            if (fs.existsSync(memoryCurrentFile)) {
                const memoryBytes = parseInt(fs.readFileSync(memoryCurrentFile, 'utf8'));
                stats.memory_mb = Math.round(memoryBytes / 1024 / 1024);
                stats.memory_percent = (stats.memory_mb / this.limits.memory_mb * 100).toFixed(1);
            }

            // PID usage
            const pidsCurrentFile = path.join(this.fullPath, 'pids.current');
            if (fs.existsSync(pidsCurrentFile)) {
                stats.pids_current = parseInt(fs.readFileSync(pidsCurrentFile, 'utf8'));
                stats.pids_percent = (stats.pids_current / this.limits.pids * 100).toFixed(1);
            }

            return stats;
        } catch (error) {
            logger.error(`Failed to get current usage: ${error.message}`);
            return {};
        }
    }

    /**
     * Start monitoring resource usage
     * @param {number} intervalMs - Monitoring interval in milliseconds
     */
    startMonitoring(intervalMs = 5000) {
        logger.info(`Starting resource monitoring (interval: ${intervalMs}ms)`);

        this.monitoring.interval = setInterval(() => {
            const usage = this.getCurrentUsage();

            console.log('\n--- Resource Usage ---');
            console.log(`Memory: ${usage.memory_mb || 'N/A'} MB / ${this.limits.memory_mb} MB (${usage.memory_percent || 'N/A'}%)`);
            console.log(`PIDs: ${usage.pids_current || 'N/A'} / ${this.limits.pids} (${usage.pids_percent || 'N/A'}%)`);

            // Check for alerts
            if (usage.memory_percent > this.monitoring.alertThreshold * 100) {
                logger.warning(`Memory usage high: ${usage.memory_percent}%`);
            }

            if (usage.pids_percent > this.monitoring.alertThreshold * 100) {
                logger.warning(`PID usage high: ${usage.pids_percent}%`);
            }
        }, intervalMs);
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        if (this.monitoring.interval) {
            clearInterval(this.monitoring.interval);
            this.monitoring.interval = null;
            logger.info('Monitoring stopped');
        }
    }

    /**
     * Remove cgroup and cleanup
     * @returns {boolean} Success status
     */
    cleanup() {
        try {
            this.stopMonitoring();

            if (fs.existsSync(this.fullPath)) {
                // Remove cgroup directory (only works if empty)
                fs.rmdirSync(this.fullPath);
                logger.success(`Cgroup ${this.cgroupName} removed`);
            }

            return true;
        } catch (error) {
            logger.error(`Failed to cleanup cgroup: ${error.message}`);
            return false;
        }
    }

    /**
     * Print limits summary
     */
    printLimitsSummary() {
        console.log('\n' + colors.green + '=== Resource Limits Summary ===' + colors.reset);
        console.log(`Cgroup: ${this.cgroupName}`);
        console.log(`CPU Cores: ${this.limits.cpu}`);
        console.log(`Memory: ${this.limits.memory_mb} MB`);
        console.log(`Max PIDs: ${this.limits.pids}`);
        console.log(`I/O Weight: ${this.limits.io_weight}`);
        console.log('================================\n');
    }
}

/**
 * Main CLI handler
 */
async function main() {
    program
        .name('resource-limiter')
        .description('CPU and memory constraints for sandboxes')
        .version('1.0.0');

    program
        .option('--cpu <cores>', 'CPU limit in cores', parseFloat)
        .option('--memory <mb>', 'Memory limit in MB', parseInt)
        .option('--pids <count>', 'Maximum number of processes', parseInt)
        .option('--io-weight <weight>', 'I/O weight (1-10000)', parseInt)
        .option('--profile <name>', 'Use predefined profile (minimal|development|production|heavy)')
        .option('--cgroup <name>', 'Cgroup name', 'claude-sandbox')
        .option('--monitor', 'Enable real-time monitoring')
        .option('--interval <ms>', 'Monitoring interval in milliseconds', parseInt, 5000)
        .option('--cleanup', 'Remove cgroup and cleanup')
        .option('--status', 'Show current resource usage')
        .option('--debug', 'Enable debug logging');

    program.parse(process.argv);

    const options = program.opts();

    // Enable debug logging
    if (options.debug) {
        process.env.DEBUG = 'true';
    }

    try {
        // Determine limits from profile or individual options
        let limits = {};

        if (options.profile) {
            if (!PROFILES[options.profile]) {
                logger.error(`Invalid profile: ${options.profile}`);
                logger.info(`Available profiles: ${Object.keys(PROFILES).join(', ')}`);
                process.exit(1);
            }
            limits = { ...PROFILES[options.profile] };
            logger.info(`Using profile: ${options.profile}`);
        }

        // Override with individual options
        if (options.cpu) limits.cpu = options.cpu;
        if (options.memory) limits.memory_mb = options.memory;
        if (options.pids) limits.pids = options.pids;
        if (options.ioWeight) limits.io_weight = options.ioWeight;

        // Initialize limiter
        const limiter = new ResourceLimiter({
            ...limits,
            cgroupName: options.cgroup
        });

        // Handle cleanup
        if (options.cleanup) {
            limiter.cleanup();
            return;
        }

        // Handle status
        if (options.status) {
            const usage = limiter.getCurrentUsage();
            console.log(JSON.stringify(usage, null, 2));
            return;
        }

        // Apply limits
        if (!limiter.applyLimits()) {
            process.exit(1);
        }

        // Move current process to cgroup
        limiter.moveProcessToCgroup();

        // Start monitoring if requested
        if (options.monitor) {
            limiter.startMonitoring(options.interval);

            // Keep process running
            process.on('SIGINT', () => {
                limiter.stopMonitoring();
                limiter.cleanup();
                process.exit(0);
            });
        }

    } catch (error) {
        logger.error(`Unexpected error: ${error.message}`);
        if (options.debug) {
            console.error(error.stack);
        }
        process.exit(2);
    }
}

// Run if executed directly
if (require.main === module) {
    main().catch(error => {
        logger.error(`Fatal error: ${error.message}`);
        process.exit(2);
    });
}

module.exports = { ResourceLimiter, PROFILES };
