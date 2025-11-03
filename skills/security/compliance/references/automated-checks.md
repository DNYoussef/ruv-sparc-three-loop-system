# Automated Compliance Checks

## Overview

Automated compliance checks are essential for continuous compliance monitoring, reducing manual effort, and ensuring consistent control validation. This document provides comprehensive guidance on implementing automated compliance validation for GDPR, HIPAA, SOC 2, PCI-DSS, and ISO 27001.

## Benefits of Automated Compliance Checks

1. **Continuous Monitoring**: Real-time compliance validation instead of periodic manual checks
2. **Consistency**: Eliminate human error and ensure uniform application of standards
3. **Efficiency**: Reduce manual effort and audit preparation time
4. **Early Detection**: Identify compliance drift before audits
5. **Evidence Collection**: Automated evidence generation for audits
6. **Cost Reduction**: Lower compliance program costs through automation
7. **Scalability**: Support growth without proportional compliance team growth

## Automated Compliance Check Categories

### 1. Access Control Validation

#### User Access Management

**Check: Principle of Least Privilege**
```javascript
// Automated check for excessive permissions
async function checkLeasePrivilege() {
  const users = await db.users.findAll({ where: { status: 'active' } });
  const violations = [];

  for (const user of users) {
    const permissions = await getUserPermissions(user.id);
    const role = user.role;
    const expectedPermissions = getRolePermissions(role);

    const excessivePermissions = permissions.filter(
      p => !expectedPermissions.includes(p)
    );

    if (excessivePermissions.length > 0) {
      violations.push({
        user_id: user.id,
        role: role,
        excessive_permissions: excessivePermissions,
        severity: 'medium',
        framework: ['GDPR Art. 32', 'HIPAA §164.312(a)(1)', 'SOC 2 CC6.1', 'ISO 27001 A.5.15']
      });
    }
  }

  return {
    check: 'least_privilege',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    compliant_users: users.length - violations.length,
    total_users: users.length
  };
}
```

**Check: Inactive User Accounts**
```javascript
// Identify inactive accounts that should be disabled
async function checkInactiveAccounts() {
  const inactivityThreshold = 90; // days
  const cutoffDate = new Date(Date.now() - inactivityThreshold * 24 * 60 * 60 * 1000);

  const inactiveUsers = await db.users.findAll({
    where: {
      status: 'active',
      last_login_at: { [Op.lt]: cutoffDate }
    }
  });

  return {
    check: 'inactive_accounts',
    status: inactiveUsers.length === 0 ? 'pass' : 'fail',
    violations: inactiveUsers.map(u => ({
      user_id: u.id,
      email: u.email,
      last_login: u.last_login_at,
      days_inactive: Math.floor((Date.now() - u.last_login_at) / (24 * 60 * 60 * 1000)),
      severity: 'medium',
      framework: ['SOC 2 CC6.1', 'ISO 27001 A.5.18', 'PCI-DSS Req. 8.1.4']
    })),
    total_inactive: inactiveUsers.length
  };
}
```

**Check: Multi-Factor Authentication (MFA) Enforcement**
```javascript
// Verify MFA is enabled for all privileged users
async function checkMFAEnforcement() {
  const privilegedRoles = ['admin', 'super_admin', 'developer', 'security'];

  const usersWithoutMFA = await db.users.findAll({
    where: {
      status: 'active',
      role: { [Op.in]: privilegedRoles },
      mfa_enabled: false
    }
  });

  return {
    check: 'mfa_enforcement',
    status: usersWithoutMFA.length === 0 ? 'pass' : 'fail',
    violations: usersWithoutMFA.map(u => ({
      user_id: u.id,
      email: u.email,
      role: u.role,
      severity: 'high',
      framework: ['HIPAA §164.312(a)(2)(i)', 'SOC 2 CC6.1', 'PCI-DSS Req. 8.3', 'ISO 27001 A.8.5']
    })),
    total_violations: usersWithoutMFA.length
  };
}
```

**Check: Shared Accounts**
```javascript
// Detect shared accounts (HIPAA violation)
async function checkSharedAccounts() {
  // Method 1: Check account_type field
  const sharedAccounts = await db.users.findAll({
    where: { account_type: 'shared', status: 'active' }
  });

  // Method 2: Detect multiple concurrent sessions from different IPs
  const suspiciousAccounts = await db.query(`
    SELECT user_id, COUNT(DISTINCT ip_address) as ip_count
    FROM sessions
    WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY user_id
    HAVING COUNT(DISTINCT ip_address) > 2
  `);

  const violations = [
    ...sharedAccounts.map(u => ({
      user_id: u.id,
      email: u.email,
      reason: 'shared_account_type',
      severity: 'critical',
      framework: ['HIPAA §164.312(a)(2)(i)', 'SOC 2 CC6.1', 'ISO 27001 A.5.18']
    })),
    ...suspiciousAccounts.map(s => ({
      user_id: s.user_id,
      reason: 'concurrent_sessions_multiple_ips',
      ip_count: s.ip_count,
      severity: 'high',
      framework: ['HIPAA §164.312(a)(2)(i)', 'SOC 2 CC6.1']
    }))
  ];

  return {
    check: 'shared_accounts',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    total_violations: violations.length
  };
}
```

#### Access Review Automation

**Check: Periodic Access Review Completion**
```javascript
// Verify quarterly access reviews are completed
async function checkAccessReviewCompliance() {
  const quarterlyReviewDue = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);

  const overdueReviews = await db.access_reviews.findAll({
    where: {
      status: 'pending',
      review_due: { [Op.lt]: new Date() },
      created_at: { [Op.lt]: quarterlyReviewDue }
    }
  });

  const usersWithoutRecentReview = await db.users.findAll({
    where: {
      status: 'active',
      id: {
        [Op.notIn]: db.sequelize.literal(`(
          SELECT user_id FROM access_reviews
          WHERE status = 'completed'
          AND completed_at > NOW() - INTERVAL '90 days'
        )`)
      }
    }
  });

  return {
    check: 'access_review_compliance',
    status: overdueReviews.length === 0 && usersWithoutRecentReview.length === 0 ? 'pass' : 'fail',
    overdue_reviews: overdueReviews.length,
    users_without_review: usersWithoutRecentReview.length,
    severity: 'medium',
    framework: ['SOC 2 CC6.1', 'ISO 27001 A.5.18', 'HIPAA §164.308(a)(3)(ii)(C)']
  };
}
```

### 2. Encryption Validation

#### Data at Rest Encryption

**Check: Database Encryption**
```javascript
// Verify database encryption is enabled
async function checkDatabaseEncryption() {
  // AWS RDS example
  const rdsInstances = await aws.rds.describeDBInstances().promise();
  const unencryptedInstances = rdsInstances.DBInstances.filter(
    instance => !instance.StorageEncrypted
  );

  return {
    check: 'database_encryption_at_rest',
    status: unencryptedInstances.length === 0 ? 'pass' : 'fail',
    violations: unencryptedInstances.map(i => ({
      instance_id: i.DBInstanceIdentifier,
      engine: i.Engine,
      severity: 'critical',
      framework: ['GDPR Art. 32', 'HIPAA §164.312(a)(2)(iv)', 'SOC 2 CC6.7', 'PCI-DSS Req. 3.4', 'ISO 27001 A.8.24']
    })),
    total_instances: rdsInstances.DBInstances.length,
    encrypted_instances: rdsInstances.DBInstances.length - unencryptedInstances.length
  };
}
```

**Check: S3 Bucket Encryption**
```javascript
// Verify S3 buckets have default encryption enabled
async function checkS3Encryption() {
  const buckets = await aws.s3.listBuckets().promise();
  const unencryptedBuckets = [];

  for (const bucket of buckets.Buckets) {
    try {
      await aws.s3.getBucketEncryption({ Bucket: bucket.Name }).promise();
    } catch (error) {
      if (error.code === 'ServerSideEncryptionConfigurationNotFoundError') {
        unencryptedBuckets.push({
          bucket_name: bucket.Name,
          created_date: bucket.CreationDate,
          severity: 'critical',
          framework: ['GDPR Art. 32', 'HIPAA §164.312(a)(2)(iv)', 'SOC 2 CC6.7', 'PCI-DSS Req. 3.4']
        });
      }
    }
  }

  return {
    check: 's3_bucket_encryption',
    status: unencryptedBuckets.length === 0 ? 'pass' : 'fail',
    violations: unencryptedBuckets,
    total_buckets: buckets.Buckets.length,
    encrypted_buckets: buckets.Buckets.length - unencryptedBuckets.length
  };
}
```

**Check: EBS Volume Encryption**
```javascript
// Verify EBS volumes are encrypted
async function checkEBSEncryption() {
  const volumes = await aws.ec2.describeVolumes().promise();
  const unencryptedVolumes = volumes.Volumes.filter(v => !v.Encrypted);

  return {
    check: 'ebs_volume_encryption',
    status: unencryptedVolumes.length === 0 ? 'pass' : 'fail',
    violations: unencryptedVolumes.map(v => ({
      volume_id: v.VolumeId,
      size: v.Size,
      state: v.State,
      attached_instances: v.Attachments.map(a => a.InstanceId),
      severity: 'critical',
      framework: ['GDPR Art. 32', 'HIPAA §164.312(a)(2)(iv)', 'SOC 2 CC6.7', 'PCI-DSS Req. 3.4']
    })),
    total_volumes: volumes.Volumes.length
  };
}
```

#### Data in Transit Encryption

**Check: TLS Version Enforcement**
```javascript
// Verify minimum TLS 1.2 (preferably TLS 1.3)
async function checkTLSVersion() {
  const loadBalancers = await aws.elbv2.describeLoadBalancers().promise();
  const violations = [];

  for (const lb of loadBalancers.LoadBalancers) {
    const listeners = await aws.elbv2.describeListeners({
      LoadBalancerArn: lb.LoadBalancerArn
    }).promise();

    for (const listener of listeners.Listeners) {
      if (listener.Protocol === 'HTTPS' || listener.Protocol === 'TLS') {
        const sslPolicy = listener.SslPolicy;

        // Check if SSL policy allows TLS 1.0 or 1.1
        if (sslPolicy.includes('TLSv1-0') || sslPolicy.includes('TLSv1-1')) {
          violations.push({
            load_balancer: lb.LoadBalancerName,
            listener_port: listener.Port,
            ssl_policy: sslPolicy,
            issue: 'weak_tls_version',
            severity: 'high',
            framework: ['GDPR Art. 32', 'HIPAA §164.312(e)(1)', 'SOC 2 CC6.7', 'PCI-DSS Req. 4.1']
          });
        }
      }
    }
  }

  return {
    check: 'tls_version_enforcement',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    total_load_balancers: loadBalancers.LoadBalancers.length
  };
}
```

**Check: Certificate Expiration**
```javascript
// Monitor SSL/TLS certificate expiration
async function checkCertificateExpiration() {
  const certificates = await aws.acm.listCertificates().promise();
  const expiringCertificates = [];

  for (const cert of certificates.CertificateSummaryList) {
    const details = await aws.acm.describeCertificate({
      CertificateArn: cert.CertificateArn
    }).promise();

    const daysUntilExpiration = Math.floor(
      (details.Certificate.NotAfter - Date.now()) / (24 * 60 * 60 * 1000)
    );

    if (daysUntilExpiration < 30) {
      expiringCertificates.push({
        domain: cert.DomainName,
        certificate_arn: cert.CertificateArn,
        expiration_date: details.Certificate.NotAfter,
        days_until_expiration: daysUntilExpiration,
        severity: daysUntilExpiration < 7 ? 'critical' : 'high',
        framework: ['SOC 2 CC6.7', 'PCI-DSS Req. 4.1']
      });
    }
  }

  return {
    check: 'certificate_expiration',
    status: expiringCertificates.length === 0 ? 'pass' : 'fail',
    violations: expiringCertificates,
    total_certificates: certificates.CertificateSummaryList.length
  };
}
```

### 3. Logging and Monitoring Validation

#### Audit Logging

**Check: Audit Log Completeness**
```javascript
// Verify critical events are being logged
async function checkAuditLogCompleteness() {
  const requiredEvents = [
    'user_login', 'user_logout', 'failed_login',
    'user_created', 'user_deleted', 'user_permission_changed',
    'data_accessed', 'data_modified', 'data_deleted',
    'configuration_changed', 'security_setting_modified'
  ];

  const recentLogs = await db.audit_logs.findAll({
    where: {
      timestamp: {
        [Op.gte]: new Date(Date.now() - 24 * 60 * 60 * 1000) // Last 24 hours
      }
    },
    attributes: ['action'],
    group: ['action']
  });

  const loggedEvents = recentLogs.map(log => log.action);
  const missingEvents = requiredEvents.filter(e => !loggedEvents.includes(e));

  return {
    check: 'audit_log_completeness',
    status: missingEvents.length === 0 ? 'pass' : 'warning',
    missing_events: missingEvents,
    logged_events: loggedEvents.length,
    required_events: requiredEvents.length,
    framework: ['HIPAA §164.312(b)', 'SOC 2 CC7.2', 'PCI-DSS Req. 10.1', 'ISO 27001 A.8.16']
  };
}
```

**Check: Log Retention Policy**
```javascript
// Verify logs are retained per policy
async function checkLogRetention() {
  const retentionPolicies = {
    'audit_logs': 2555, // 7 years (HIPAA requirement)
    'access_logs': 90,
    'security_logs': 365,
    'application_logs': 30
  };

  const violations = [];

  for (const [logType, retentionDays] of Object.entries(retentionPolicies)) {
    const oldestLog = await db[logType].findOne({
      order: [['timestamp', 'ASC']],
      attributes: ['timestamp']
    });

    if (oldestLog) {
      const actualRetentionDays = Math.floor(
        (Date.now() - oldestLog.timestamp) / (24 * 60 * 60 * 1000)
      );

      if (actualRetentionDays < retentionDays) {
        violations.push({
          log_type: logType,
          required_retention_days: retentionDays,
          actual_retention_days: actualRetentionDays,
          oldest_log: oldestLog.timestamp,
          severity: 'medium',
          framework: ['HIPAA §164.316(b)(2)', 'SOC 2 CC7.2', 'PCI-DSS Req. 10.7', 'ISO 27001 A.8.15']
        });
      }
    }
  }

  return {
    check: 'log_retention_policy',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations
  };
}
```

**Check: Log Integrity Protection**
```javascript
// Verify logs cannot be modified (write-once)
async function checkLogIntegrity() {
  // Check if audit logs table has UPDATE/DELETE triggers disabled
  const logTables = ['audit_logs', 'security_logs', 'access_logs'];
  const violations = [];

  for (const table of logTables) {
    // Check database permissions
    const permissions = await db.query(`
      SELECT grantee, privilege_type
      FROM information_schema.table_privileges
      WHERE table_name = '${table}'
      AND privilege_type IN ('UPDATE', 'DELETE')
    `);

    if (permissions.length > 0) {
      violations.push({
        table: table,
        issue: 'update_delete_permissions_granted',
        grantees: permissions.map(p => p.grantee),
        severity: 'high',
        framework: ['HIPAA §164.312(c)(1)', 'SOC 2 CC7.2', 'PCI-DSS Req. 10.5']
      });
    }
  }

  return {
    check: 'log_integrity_protection',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations
  };
}
```

#### Security Monitoring

**Check: SIEM Integration**
```javascript
// Verify security events are being sent to SIEM
async function checkSIEMIntegration() {
  const siemEndpoint = process.env.SIEM_ENDPOINT;
  const recentEvents = await db.security_events.findAll({
    where: {
      timestamp: {
        [Op.gte]: new Date(Date.now() - 60 * 60 * 1000) // Last hour
      }
    }
  });

  // Check if events were successfully sent to SIEM
  const failedEvents = recentEvents.filter(e => !e.siem_sent || e.siem_error);

  return {
    check: 'siem_integration',
    status: failedEvents.length === 0 ? 'pass' : 'fail',
    total_events: recentEvents.length,
    failed_events: failedEvents.length,
    success_rate: ((recentEvents.length - failedEvents.length) / recentEvents.length * 100).toFixed(2) + '%',
    severity: failedEvents.length > 0 ? 'high' : 'pass',
    framework: ['SOC 2 CC7.2', 'ISO 27001 A.8.16', 'PCI-DSS Req. 10.6']
  };
}
```

### 4. Network Security Validation

#### Firewall and Network Segmentation

**Check: Security Group Configuration**
```javascript
// Verify security groups follow least privilege
async function checkSecurityGroups() {
  const securityGroups = await aws.ec2.describeSecurityGroups().promise();
  const violations = [];

  for (const sg of securityGroups.SecurityGroups) {
    // Check for overly permissive inbound rules
    for (const rule of sg.IpPermissions) {
      // Check for 0.0.0.0/0 (open to internet)
      const openToInternet = rule.IpRanges.some(
        range => range.CidrIp === '0.0.0.0/0'
      );

      if (openToInternet) {
        // High-risk ports
        const highRiskPorts = [22, 3306, 5432, 1433, 3389, 5984, 6379, 27017];

        if (rule.FromPort && highRiskPorts.includes(rule.FromPort)) {
          violations.push({
            security_group_id: sg.GroupId,
            group_name: sg.GroupName,
            port: rule.FromPort,
            protocol: rule.IpProtocol,
            issue: 'sensitive_port_open_to_internet',
            severity: 'critical',
            framework: ['SOC 2 CC6.6', 'PCI-DSS Req. 1.3', 'ISO 27001 A.8.20']
          });
        } else {
          violations.push({
            security_group_id: sg.GroupId,
            group_name: sg.GroupName,
            port: rule.FromPort || 'all',
            protocol: rule.IpProtocol,
            issue: 'port_open_to_internet',
            severity: 'medium',
            framework: ['SOC 2 CC6.6', 'PCI-DSS Req. 1.3']
          });
        }
      }
    }
  }

  return {
    check: 'security_group_configuration',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    total_security_groups: securityGroups.SecurityGroups.length
  };
}
```

**Check: Network ACL Configuration**
```javascript
// Verify network ACLs provide proper segmentation
async function checkNetworkACLs() {
  const networkAcls = await aws.ec2.describeNetworkAcls().promise();
  const violations = [];

  for (const acl of networkAcls.NetworkAcls) {
    // Check for overly permissive rules
    const inboundRules = acl.Entries.filter(e => !e.Egress);

    for (const rule of inboundRules) {
      if (rule.CidrBlock === '0.0.0.0/0' && rule.RuleAction === 'allow') {
        violations.push({
          network_acl_id: acl.NetworkAclId,
          rule_number: rule.RuleNumber,
          cidr_block: rule.CidrBlock,
          port_range: rule.PortRange,
          issue: 'overly_permissive_acl_rule',
          severity: 'medium',
          framework: ['PCI-DSS Req. 1.3', 'ISO 27001 A.8.20']
        });
      }
    }
  }

  return {
    check: 'network_acl_configuration',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    total_acls: networkAcls.NetworkAcls.length
  };
}
```

#### Public Exposure

**Check: Public S3 Buckets**
```javascript
// Verify no S3 buckets are publicly accessible (unless intentional)
async function checkPublicS3Buckets() {
  const buckets = await aws.s3.listBuckets().promise();
  const publicBuckets = [];

  for (const bucket of buckets.Buckets) {
    try {
      const acl = await aws.s3.getBucketAcl({ Bucket: bucket.Name }).promise();

      const publicRead = acl.Grants.some(grant =>
        grant.Grantee.Type === 'Group' &&
        grant.Grantee.URI === 'http://acs.amazonaws.com/groups/global/AllUsers' &&
        (grant.Permission === 'READ' || grant.Permission === 'FULL_CONTROL')
      );

      if (publicRead) {
        publicBuckets.push({
          bucket_name: bucket.Name,
          issue: 'publicly_readable',
          severity: 'critical',
          framework: ['GDPR Art. 32', 'HIPAA §164.312(a)(1)', 'SOC 2 CC6.6', 'PCI-DSS Req. 1.3']
        });
      }
    } catch (error) {
      console.error(`Error checking bucket ${bucket.Name}:`, error.message);
    }
  }

  return {
    check: 'public_s3_buckets',
    status: publicBuckets.length === 0 ? 'pass' : 'fail',
    violations: publicBuckets,
    total_buckets: buckets.Buckets.length
  };
}
```

### 5. Vulnerability Management

#### Patch Management

**Check: Operating System Patching**
```javascript
// Verify EC2 instances have recent patches
async function checkOSPatching() {
  const instances = await aws.ec2.describeInstances().promise();
  const violations = [];

  for (const reservation of instances.Reservations) {
    for (const instance of reservation.Instances) {
      // Check SSM patch compliance
      try {
        const patchInfo = await aws.ssm.describeInstancePatchStates({
          InstanceIds: [instance.InstanceId]
        }).promise();

        for (const patch of patchInfo.InstancePatchStates) {
          if (patch.MissingCount > 0 || patch.FailedCount > 0) {
            violations.push({
              instance_id: instance.InstanceId,
              missing_patches: patch.MissingCount,
              failed_patches: patch.FailedCount,
              last_patch_date: patch.OperationEndTime,
              severity: patch.MissingCount > 10 ? 'high' : 'medium',
              framework: ['SOC 2 CC7.1', 'PCI-DSS Req. 6.2', 'ISO 27001 A.8.8']
            });
          }
        }
      } catch (error) {
        // Instance not managed by SSM
        violations.push({
          instance_id: instance.InstanceId,
          issue: 'not_managed_by_ssm',
          severity: 'medium',
          framework: ['SOC 2 CC7.1']
        });
      }
    }
  }

  return {
    check: 'os_patching',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations
  };
}
```

**Check: Vulnerability Scanning**
```javascript
// Verify regular vulnerability scans are conducted
async function checkVulnerabilityScanning() {
  const lastScanDate = await db.vulnerability_scans.findOne({
    order: [['scan_date', 'DESC']],
    attributes: ['scan_date']
  });

  const daysSinceLastScan = lastScanDate
    ? Math.floor((Date.now() - lastScanDate.scan_date) / (24 * 60 * 60 * 1000))
    : 999;

  // PCI-DSS requires quarterly scans (90 days)
  const scanOverdue = daysSinceLastScan > 90;

  return {
    check: 'vulnerability_scanning',
    status: !scanOverdue ? 'pass' : 'fail',
    last_scan_date: lastScanDate?.scan_date,
    days_since_last_scan: daysSinceLastScan,
    required_frequency: 90, // days
    severity: scanOverdue ? 'high' : 'pass',
    framework: ['PCI-DSS Req. 11.2', 'SOC 2 CC7.1', 'ISO 27001 A.8.8']
  };
}
```

### 6. Incident Response and Business Continuity

#### Backup Validation

**Check: Database Backup Status**
```javascript
// Verify database backups are current
async function checkDatabaseBackups() {
  const rdsInstances = await aws.rds.describeDBInstances().promise();
  const violations = [];

  for (const instance of rdsInstances.DBInstances) {
    if (!instance.BackupRetentionPeriod || instance.BackupRetentionPeriod < 7) {
      violations.push({
        instance_id: instance.DBInstanceIdentifier,
        backup_retention_period: instance.BackupRetentionPeriod,
        issue: 'insufficient_backup_retention',
        severity: 'high',
        framework: ['HIPAA §164.308(a)(7)(ii)(A)', 'SOC 2 CC7.4', 'ISO 27001 A.8.13']
      });
    }

    // Check latest backup time
    if (instance.LatestRestorableTime) {
      const hoursSinceBackup = (Date.now() - instance.LatestRestorableTime) / (60 * 60 * 1000);

      if (hoursSinceBackup > 24) {
        violations.push({
          instance_id: instance.DBInstanceIdentifier,
          latest_restorable_time: instance.LatestRestorableTime,
          hours_since_backup: hoursSinceBackup,
          issue: 'backup_not_recent',
          severity: 'medium',
          framework: ['SOC 2 CC7.4', 'ISO 27001 A.8.13']
        });
      }
    }
  }

  return {
    check: 'database_backups',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    total_instances: rdsInstances.DBInstances.length
  };
}
```

**Check: Disaster Recovery Testing**
```javascript
// Verify DR testing is conducted regularly
async function checkDRTesting() {
  const lastDRTest = await db.dr_tests.findOne({
    order: [['test_date', 'DESC']],
    attributes: ['test_date', 'test_result']
  });

  const daysSinceLastTest = lastDRTest
    ? Math.floor((Date.now() - lastDRTest.test_date) / (24 * 60 * 60 * 1000))
    : 999;

  // SOC 2 recommends at least annual testing
  const testOverdue = daysSinceLastTest > 365;

  return {
    check: 'disaster_recovery_testing',
    status: !testOverdue && lastDRTest?.test_result === 'pass' ? 'pass' : 'fail',
    last_test_date: lastDRTest?.test_date,
    last_test_result: lastDRTest?.test_result,
    days_since_last_test: daysSinceLastTest,
    required_frequency: 365, // days (annual)
    severity: testOverdue ? 'medium' : 'pass',
    framework: ['HIPAA §164.308(a)(7)', 'SOC 2 CC7.4', 'ISO 27001 A.8.14']
  };
}
```

### 7. Training and Awareness

**Check: Security Training Completion**
```javascript
// Verify all employees have completed required training
async function checkSecurityTraining() {
  const employees = await db.employees.findAll({
    where: { status: 'active' }
  });

  const violations = [];

  for (const employee of employees) {
    // Check if training completed in last 12 months
    const recentTraining = await db.training_completions.findOne({
      where: {
        employee_id: employee.id,
        course_type: 'security_fundamentals',
        completion_date: {
          [Op.gte]: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000)
        }
      }
    });

    if (!recentTraining) {
      violations.push({
        employee_id: employee.id,
        email: employee.email,
        issue: 'security_training_overdue',
        severity: 'medium',
        framework: ['HIPAA §164.308(a)(5)', 'SOC 2 CC1.4', 'PCI-DSS Req. 12.6', 'ISO 27001 A.6.3']
      });
    }
  }

  return {
    check: 'security_training_completion',
    status: violations.length === 0 ? 'pass' : 'fail',
    violations: violations,
    completion_rate: ((employees.length - violations.length) / employees.length * 100).toFixed(2) + '%',
    total_employees: employees.length
  };
}
```

## Automated Compliance Monitoring Framework

### Continuous Monitoring Architecture

```javascript
// Centralized compliance monitoring orchestrator
class ComplianceMonitor {
  constructor() {
    this.checks = [
      checkLeasePrivilege,
      checkInactiveAccounts,
      checkMFAEnforcement,
      checkSharedAccounts,
      checkDatabaseEncryption,
      checkS3Encryption,
      checkTLSVersion,
      checkAuditLogCompleteness,
      checkLogRetention,
      checkSecurityGroups,
      checkPublicS3Buckets,
      checkOSPatching,
      checkVulnerabilityScanning,
      checkDatabaseBackups,
      checkSecurityTraining
      // ... add all checks
    ];
  }

  async runAllChecks() {
    const results = [];

    for (const check of this.checks) {
      try {
        const result = await check();
        results.push(result);

        // Store results
        await this.storeCheckResult(result);

        // Alert on failures
        if (result.status === 'fail') {
          await this.alertOnFailure(result);
        }
      } catch (error) {
        console.error(`Check ${check.name} failed:`, error);
        results.push({
          check: check.name,
          status: 'error',
          error: error.message
        });
      }
    }

    // Generate compliance dashboard
    await this.generateDashboard(results);

    return results;
  }

  async storeCheckResult(result) {
    await db.compliance_checks.create({
      check_name: result.check,
      status: result.status,
      violations: JSON.stringify(result.violations),
      timestamp: new Date(),
      framework: result.framework
    });
  }

  async alertOnFailure(result) {
    if (result.severity === 'critical') {
      // Page on-call engineer
      await this.sendPagerDutyAlert(result);
    } else if (result.severity === 'high') {
      // Slack alert to security channel
      await this.sendSlackAlert(result);
    } else {
      // Email digest
      await this.addToEmailDigest(result);
    }
  }

  async generateDashboard(results) {
    const dashboard = {
      timestamp: new Date(),
      total_checks: results.length,
      passed: results.filter(r => r.status === 'pass').length,
      failed: results.filter(r => r.status === 'fail').length,
      errors: results.filter(r => r.status === 'error').length,
      compliance_score: this.calculateComplianceScore(results),
      by_framework: this.groupByFramework(results)
    };

    // Store dashboard
    await db.compliance_dashboards.create(dashboard);

    // Update real-time dashboard
    await this.updateRealtimeDashboard(dashboard);

    return dashboard;
  }

  calculateComplianceScore(results) {
    const passed = results.filter(r => r.status === 'pass').length;
    const total = results.length;
    return ((passed / total) * 100).toFixed(2);
  }
}

// Schedule continuous monitoring
const monitor = new ComplianceMonitor();

// Run checks every hour
cron.schedule('0 * * * *', async () => {
  console.log('Running automated compliance checks...');
  await monitor.runAllChecks();
});

// Run full compliance audit daily
cron.schedule('0 2 * * *', async () => {
  console.log('Running daily compliance audit...');
  const results = await monitor.runAllChecks();

  // Generate daily compliance report
  await generateComplianceReport(results);
});
```

## Compliance Dashboard

### Real-Time Compliance Metrics

**Dashboard Components**:
1. **Compliance Score**: Overall percentage of passed checks
2. **Framework Status**: Compliance status per framework (GDPR, HIPAA, SOC 2, PCI-DSS, ISO 27001)
3. **Critical Violations**: High-priority issues requiring immediate attention
4. **Trend Analysis**: Compliance score over time
5. **Check Status**: Current status of each automated check
6. **Remediation Tracking**: Progress on addressing violations

### Evidence Collection for Audits

**Automated Evidence**:
- Screenshots of compliance dashboard
- Compliance check logs
- Configuration exports
- Audit logs
- Training completion reports
- Incident response logs
- Backup verification reports
- Vulnerability scan reports

**Evidence Storage**:
```javascript
// Store evidence for audit purposes
async function collectAuditEvidence() {
  const evidence = {
    timestamp: new Date(),
    compliance_checks: await db.compliance_checks.findAll({
      where: {
        timestamp: {
          [Op.gte]: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000) // Last year
        }
      }
    }),
    training_completions: await db.training_completions.findAll(),
    incident_response_logs: await db.incidents.findAll(),
    access_reviews: await db.access_reviews.findAll(),
    vulnerability_scans: await db.vulnerability_scans.findAll(),
    dr_tests: await db.dr_tests.findAll()
  };

  // Store evidence package
  const evidencePackage = await this.createEvidencePackage(evidence);

  // Upload to secure storage
  await this.uploadToAuditStorage(evidencePackage);

  return evidencePackage;
}
```

## Conclusion

Automated compliance checks are essential for maintaining continuous compliance, reducing manual effort, and ensuring consistent control validation. By implementing comprehensive automated checks across access control, encryption, logging, network security, vulnerability management, and business continuity, organizations can:

1. **Maintain Continuous Compliance**: Real-time monitoring instead of periodic manual checks
2. **Reduce Audit Costs**: Automated evidence collection and reporting
3. **Improve Security Posture**: Early detection of compliance drift
4. **Scale Efficiently**: Support growth without proportional compliance team growth
5. **Demonstrate Due Diligence**: Comprehensive audit trail of compliance efforts

**Key Takeaways**:
- Automate repetitive compliance checks
- Monitor compliance continuously, not just before audits
- Alert on critical violations immediately
- Collect evidence automatically for audit purposes
- Track compliance trends over time
- Integrate compliance checks into CI/CD pipelines
- Treat compliance as code (infrastructure as code principles)
