/**
 * Infrastructure Validation Tests
 * Purpose: Test infrastructure provisioning and validation using Terratest patterns
 * Framework: Jest with AWS SDK, Azure SDK
 * Version: 2.0.0
 */

const { exec } = require('child_process');
const util = require('util');
const AWS = require('aws-sdk');
const { ComputeManagementClient } = require('@azure/arm-compute');
const { DefaultAzureCredential } = require('@azure/identity');

const execPromise = util.promisify(exec);

describe('Infrastructure Validation Tests', () => {
  const environment = process.env.TEST_ENVIRONMENT || 'test';
  const awsRegion = process.env.AWS_REGION || 'us-east-1';

  // AWS clients
  let ec2, elb, rds;

  beforeAll(() => {
    AWS.config.update({ region: awsRegion });
    ec2 = new AWS.EC2();
    elb = new AWS.ELBv2();
    rds = new AWS.RDS();
  });

  describe('AWS Infrastructure Tests', () => {
    test('VPC should exist with correct CIDR block', async () => {
      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
        ],
      };

      const result = await ec2.describeVpcs(params).promise();
      expect(result.Vpcs).toHaveLength(1);

      const vpc = result.Vpcs[0];
      expect(vpc.CidrBlock).toBe('10.0.0.0/16');
      expect(vpc.State).toBe('available');
    });

    test('Public subnets should be in different availability zones', async () => {
      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
          {
            Name: 'tag:Type',
            Values: ['public'],
          },
        ],
      };

      const result = await ec2.describeSubnets(params).promise();
      expect(result.Subnets.length).toBeGreaterThanOrEqual(2);

      const azs = result.Subnets.map((subnet) => subnet.AvailabilityZone);
      const uniqueAzs = new Set(azs);
      expect(uniqueAzs.size).toBeGreaterThanOrEqual(2);
    });

    test('Private subnets should not have internet gateway routes', async () => {
      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
          {
            Name: 'tag:Type',
            Values: ['private'],
          },
        ],
      };

      const subnets = await ec2.describeSubnets(params).promise();

      for (const subnet of subnets.Subnets) {
        const routeParams = {
          Filters: [
            {
              Name: 'association.subnet-id',
              Values: [subnet.SubnetId],
            },
          ],
        };

        const routeTables = await ec2.describeRouteTables(routeParams).promise();

        for (const routeTable of routeTables.RouteTables) {
          const igwRoutes = routeTable.Routes.filter((route) =>
            route.GatewayId?.startsWith('igw-')
          );
          expect(igwRoutes).toHaveLength(0);
        }
      }
    });

    test('NAT gateways should exist in public subnets', async () => {
      const natGateways = await ec2
        .describeNatGateways({
          Filter: [
            {
              Name: 'tag:Environment',
              Values: [environment],
            },
            {
              Name: 'state',
              Values: ['available'],
            },
          ],
        })
        .promise();

      expect(natGateways.NatGateways.length).toBeGreaterThanOrEqual(1);

      for (const natGw of natGateways.NatGateways) {
        const subnet = await ec2
          .describeSubnets({ SubnetIds: [natGw.SubnetId] })
          .promise();

        const subnetTags = subnet.Subnets[0].Tags || [];
        const typeTag = subnetTags.find((tag) => tag.Key === 'Type');
        expect(typeTag?.Value).toBe('public');
      }
    });

    test('Security groups should have appropriate ingress rules', async () => {
      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
          {
            Name: 'group-name',
            Values: [`${environment}-app-sg`],
          },
        ],
      };

      const result = await ec2.describeSecurityGroups(params).promise();
      expect(result.SecurityGroups).toHaveLength(1);

      const sg = result.SecurityGroups[0];
      const httpRule = sg.IpPermissions.find(
        (rule) => rule.FromPort === 80 && rule.ToPort === 80
      );
      const httpsRule = sg.IpPermissions.find(
        (rule) => rule.FromPort === 443 && rule.ToPort === 443
      );

      expect(httpRule).toBeDefined();
      expect(httpsRule).toBeDefined();
    });
  });

  describe('Azure Infrastructure Tests', () => {
    test('Resource group should exist', async () => {
      const credential = new DefaultAzureCredential();
      const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID;

      if (!subscriptionId) {
        console.warn('Skipping Azure tests: AZURE_SUBSCRIPTION_ID not set');
        return;
      }

      const { ResourceManagementClient } = require('@azure/arm-resources');
      const client = new ResourceManagementClient(credential, subscriptionId);

      const resourceGroupName = `${environment}-rg`;
      const resourceGroup = await client.resourceGroups.get(resourceGroupName);

      expect(resourceGroup.name).toBe(resourceGroupName);
      expect(resourceGroup.properties.provisioningState).toBe('Succeeded');
    });
  });

  describe('Terraform State Tests', () => {
    test('Terraform state should be valid', async () => {
      const { stdout } = await execPromise('terraform show -json');
      const tfState = JSON.parse(stdout);

      expect(tfState.values).toBeDefined();
      expect(tfState.values.root_module).toBeDefined();
    });

    test('All outputs should be defined', async () => {
      const { stdout } = await execPromise('terraform output -json');
      const outputs = JSON.parse(stdout);

      expect(outputs).toHaveProperty('aws_vpc_id');
      expect(outputs).toHaveProperty('aws_public_subnet_ids');
      expect(outputs).toHaveProperty('aws_private_subnet_ids');
    });

    test('No resources should be in failed state', async () => {
      const { stdout } = await execPromise('terraform show -json');
      const tfState = JSON.parse(stdout);

      const resources = tfState.values.root_module.resources || [];
      const failedResources = resources.filter(
        (resource) => resource.values?.status === 'failed'
      );

      expect(failedResources).toHaveLength(0);
    });
  });

  describe('Compliance Tests', () => {
    test('All resources should have required tags', async () => {
      const requiredTags = ['Environment', 'ManagedBy'];

      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
        ],
      };

      // Check VPC tags
      const vpcs = await ec2.describeVpcs(params).promise();
      for (const vpc of vpcs.Vpcs) {
        const tagKeys = (vpc.Tags || []).map((tag) => tag.Key);
        for (const requiredTag of requiredTags) {
          expect(tagKeys).toContain(requiredTag);
        }
      }

      // Check Subnet tags
      const subnets = await ec2.describeSubnets(params).promise();
      for (const subnet of subnets.Subnets) {
        const tagKeys = (subnet.Tags || []).map((tag) => tag.Key);
        for (const requiredTag of requiredTags) {
          expect(tagKeys).toContain(requiredTag);
        }
      }
    });

    test('Encryption should be enabled for all storage', async () => {
      // Check RDS encryption
      const rdsInstances = await rds
        .describeDBInstances({
          Filters: [
            {
              Name: 'tag:Environment',
              Values: [environment],
            },
          ],
        })
        .promise();

      for (const instance of rdsInstances.DBInstances || []) {
        expect(instance.StorageEncrypted).toBe(true);
      }

      // Check EBS encryption
      const volumes = await ec2
        .describeVolumes({
          Filters: [
            {
              Name: 'tag:Environment',
              Values: [environment],
            },
          ],
        })
        .promise();

      for (const volume of volumes.Volumes || []) {
        expect(volume.Encrypted).toBe(true);
      }
    });
  });

  describe('High Availability Tests', () => {
    test('Multi-AZ deployment should be configured', async () => {
      const params = {
        Filters: [
          {
            Name: 'tag:Environment',
            Values: [environment],
          },
        ],
      };

      const subnets = await ec2.describeSubnets(params).promise();
      const azs = [...new Set(subnets.Subnets.map((s) => s.AvailabilityZone))];

      expect(azs.length).toBeGreaterThanOrEqual(2);
    });

    test('Load balancer should distribute across multiple AZs', async () => {
      const loadBalancers = await elb
        .describeLoadBalancers({
          Names: [`${environment}-alb`],
        })
        .promise();

      if (loadBalancers.LoadBalancers.length > 0) {
        const lb = loadBalancers.LoadBalancers[0];
        expect(lb.AvailabilityZones.length).toBeGreaterThanOrEqual(2);
      }
    });
  });
});

describe('Infrastructure Performance Tests', () => {
  test('Infrastructure provisioning should complete within time limit', async () => {
    const startTime = Date.now();

    // This test assumes Terraform is already initialized
    const { stdout } = await execPromise('terraform plan -detailed-exitcode', {
      timeout: 120000, // 2 minutes
    });

    const duration = Date.now() - startTime;
    expect(duration).toBeLessThan(120000); // Should complete within 2 minutes
  });
});
