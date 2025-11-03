#!/usr/bin/env python3
"""
AWS Deployment Automation Script
Handles Lambda, ECS, and EC2 deployments with CloudFormation/CDK
"""

import boto3
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

class AWSDeployer:
    """AWS deployment automation with multi-service support"""

    def __init__(self, region: str = 'us-east-1', profile: Optional[str] = None):
        """Initialize AWS clients with optional profile"""
        self.region = region
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()

        self.cloudformation = session.client('cloudformation', region_name=region)
        self.lambda_client = session.client('lambda', region_name=region)
        self.ecs = session.client('ecs', region_name=region)
        self.ec2 = session.client('ec2', region_name=region)
        self.s3 = session.client('s3', region_name=region)

    def deploy_lambda(self, function_name: str, zip_file: str, handler: str,
                     runtime: str = 'python3.11', memory: int = 512,
                     timeout: int = 60, env_vars: Optional[Dict] = None) -> Dict:
        """Deploy Lambda function with configuration"""
        try:
            with open(zip_file, 'rb') as f:
                zip_content = f.read()

            # Check if function exists
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                # Update existing function
                update_response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )

                # Update configuration
                config_response = self.lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Handler=handler,
                    MemorySize=memory,
                    Timeout=timeout,
                    Environment={'Variables': env_vars or {}}
                )

                return {
                    'status': 'updated',
                    'function_arn': config_response['FunctionArn'],
                    'version': config_response['Version']
                }

            except self.lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                create_response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Role=os.environ.get('AWS_LAMBDA_ROLE'),  # Must be set
                    Handler=handler,
                    Code={'ZipFile': zip_content},
                    MemorySize=memory,
                    Timeout=timeout,
                    Environment={'Variables': env_vars or {}}
                )

                return {
                    'status': 'created',
                    'function_arn': create_response['FunctionArn'],
                    'version': create_response['Version']
                }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def deploy_ecs_service(self, cluster_name: str, service_name: str,
                          task_definition: str, desired_count: int = 1,
                          subnets: Optional[List[str]] = None,
                          security_groups: Optional[List[str]] = None) -> Dict:
        """Deploy ECS Fargate service"""
        try:
            # Register task definition from file
            with open(task_definition, 'r') as f:
                task_def = json.load(f)

            register_response = self.ecs.register_task_definition(**task_def)
            task_def_arn = register_response['taskDefinition']['taskDefinitionArn']

            # Check if service exists
            try:
                describe_response = self.ecs.describe_services(
                    cluster=cluster_name,
                    services=[service_name]
                )

                if describe_response['services'] and \
                   describe_response['services'][0]['status'] == 'ACTIVE':
                    # Update existing service
                    update_response = self.ecs.update_service(
                        cluster=cluster_name,
                        service=service_name,
                        taskDefinition=task_def_arn,
                        desiredCount=desired_count,
                        forceNewDeployment=True
                    )

                    return {
                        'status': 'updated',
                        'service_arn': update_response['service']['serviceArn'],
                        'task_definition': task_def_arn
                    }

            except Exception:
                pass

            # Create new service
            network_config = {
                'awsvpcConfiguration': {
                    'subnets': subnets or [],
                    'securityGroups': security_groups or [],
                    'assignPublicIp': 'ENABLED'
                }
            }

            create_response = self.ecs.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=desired_count,
                launchType='FARGATE',
                networkConfiguration=network_config
            )

            return {
                'status': 'created',
                'service_arn': create_response['service']['serviceArn'],
                'task_definition': task_def_arn
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def deploy_cloudformation_stack(self, stack_name: str, template_file: str,
                                   parameters: Optional[List[Dict]] = None,
                                   capabilities: Optional[List[str]] = None) -> Dict:
        """Deploy CloudFormation stack with change set"""
        try:
            with open(template_file, 'r') as f:
                template_body = f.read()

            # Check if stack exists
            try:
                self.cloudformation.describe_stacks(StackName=stack_name)
                stack_exists = True
            except self.cloudformation.exceptions.ClientError:
                stack_exists = False

            change_set_name = f"{stack_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            change_set_type = 'UPDATE' if stack_exists else 'CREATE'

            # Create change set
            change_set_response = self.cloudformation.create_change_set(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters or [],
                Capabilities=capabilities or ['CAPABILITY_IAM'],
                ChangeSetName=change_set_name,
                ChangeSetType=change_set_type
            )

            # Wait for change set creation
            waiter = self.cloudformation.get_waiter('change_set_create_complete')
            waiter.wait(
                StackName=stack_name,
                ChangeSetName=change_set_name
            )

            # Execute change set
            self.cloudformation.execute_change_set(
                StackName=stack_name,
                ChangeSetName=change_set_name
            )

            # Wait for stack completion
            if stack_exists:
                waiter = self.cloudformation.get_waiter('stack_update_complete')
            else:
                waiter = self.cloudformation.get_waiter('stack_create_complete')

            waiter.wait(StackName=stack_name)

            # Get stack outputs
            stack_info = self.cloudformation.describe_stacks(StackName=stack_name)
            outputs = stack_info['Stacks'][0].get('Outputs', [])

            return {
                'status': 'success',
                'stack_id': change_set_response['StackId'],
                'outputs': {o['OutputKey']: o['OutputValue'] for o in outputs}
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def deploy_ec2_instance(self, instance_name: str, ami_id: str,
                           instance_type: str = 't3.micro',
                           key_name: Optional[str] = None,
                           security_group_ids: Optional[List[str]] = None,
                           user_data: Optional[str] = None) -> Dict:
        """Deploy EC2 instance with tags"""
        try:
            run_params = {
                'ImageId': ami_id,
                'InstanceType': instance_type,
                'MinCount': 1,
                'MaxCount': 1,
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': instance_name},
                        {'Key': 'DeployedBy', 'Value': 'AWSDeployer'},
                        {'Key': 'DeployedAt', 'Value': datetime.now().isoformat()}
                    ]
                }]
            }

            if key_name:
                run_params['KeyName'] = key_name
            if security_group_ids:
                run_params['SecurityGroupIds'] = security_group_ids
            if user_data:
                run_params['UserData'] = user_data

            response = self.ec2.run_instances(**run_params)
            instance = response['Instances'][0]

            return {
                'status': 'launched',
                'instance_id': instance['InstanceId'],
                'private_ip': instance.get('PrivateIpAddress'),
                'state': instance['State']['Name']
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


def main():
    """CLI interface for AWS deployment"""
    import argparse

    parser = argparse.ArgumentParser(description='AWS Deployment Automation')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--profile', help='AWS profile name')

    subparsers = parser.add_subparsers(dest='command', help='Deployment type')

    # Lambda deployment
    lambda_parser = subparsers.add_parser('lambda', help='Deploy Lambda function')
    lambda_parser.add_argument('--name', required=True, help='Function name')
    lambda_parser.add_argument('--zip', required=True, help='Zip file path')
    lambda_parser.add_argument('--handler', required=True, help='Handler (e.g., index.handler)')
    lambda_parser.add_argument('--runtime', default='python3.11', help='Runtime')
    lambda_parser.add_argument('--memory', type=int, default=512, help='Memory in MB')
    lambda_parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')

    # ECS deployment
    ecs_parser = subparsers.add_parser('ecs', help='Deploy ECS service')
    ecs_parser.add_argument('--cluster', required=True, help='Cluster name')
    ecs_parser.add_argument('--service', required=True, help='Service name')
    ecs_parser.add_argument('--task-def', required=True, help='Task definition file')
    ecs_parser.add_argument('--count', type=int, default=1, help='Desired count')

    # CloudFormation deployment
    cfn_parser = subparsers.add_parser('cfn', help='Deploy CloudFormation stack')
    cfn_parser.add_argument('--stack', required=True, help='Stack name')
    cfn_parser.add_argument('--template', required=True, help='Template file')

    # EC2 deployment
    ec2_parser = subparsers.add_parser('ec2', help='Deploy EC2 instance')
    ec2_parser.add_argument('--name', required=True, help='Instance name')
    ec2_parser.add_argument('--ami', required=True, help='AMI ID')
    ec2_parser.add_argument('--type', default='t3.micro', help='Instance type')

    args = parser.parse_args()

    deployer = AWSDeployer(region=args.region, profile=args.profile)

    if args.command == 'lambda':
        result = deployer.deploy_lambda(
            function_name=args.name,
            zip_file=args.zip,
            handler=args.handler,
            runtime=args.runtime,
            memory=args.memory,
            timeout=args.timeout
        )
    elif args.command == 'ecs':
        result = deployer.deploy_ecs_service(
            cluster_name=args.cluster,
            service_name=args.service,
            task_definition=args.task_def,
            desired_count=args.count
        )
    elif args.command == 'cfn':
        result = deployer.deploy_cloudformation_stack(
            stack_name=args.stack,
            template_file=args.template
        )
    elif args.command == 'ec2':
        result = deployer.deploy_ec2_instance(
            instance_name=args.name,
            ami_id=args.ami,
            instance_type=args.type
        )
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get('status') not in ['error'] else 1)


if __name__ == '__main__':
    main()
