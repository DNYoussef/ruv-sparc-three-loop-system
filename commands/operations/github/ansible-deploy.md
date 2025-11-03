# Ansible Deployment Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Medium

## Purpose

Automated configuration management and application deployment using Ansible playbooks with inventory management and role-based architecture.

## Usage

```bash
/ansible-deploy [action] [options]

Actions:
  init                Initialize Ansible project
  playbook            Run playbook
  role                Create/apply role
  inventory           Manage inventory
  vault               Manage secrets
  lint                Lint playbooks
  dry-run             Test without changes

Options:
  --playbook <file>   Playbook file
  --inventory <file>  Inventory file
  --limit <hosts>     Limit to specific hosts
  --tags <tags>       Run specific tags
  --check             Dry-run mode
  --diff              Show changes
  --vault-id <id>     Vault password ID
```

## Examples

```bash
# Initialize Ansible project structure
/ansible-deploy init --project myapp

# Run playbook
/ansible-deploy playbook --playbook deploy.yml --inventory prod

# Dry-run deployment
/ansible-deploy playbook --playbook deploy.yml --check --diff

# Deploy to specific servers
/ansible-deploy playbook --playbook deploy.yml --limit webservers

# Run specific tags
/ansible-deploy playbook --playbook deploy.yml --tags database,nginx

# Create new role
/ansible-deploy role --name webapp

# Manage secrets with Ansible Vault
/ansible-deploy vault --encrypt secrets.yml

# Lint playbooks
/ansible-deploy lint --playbook deploy.yml
```

## Project Structure

```
ansible/
├── ansible.cfg              # Ansible configuration
├── inventory/
│   ├── hosts                # Inventory file
│   ├── group_vars/
│   │   ├── all.yml         # Variables for all hosts
│   │   ├── webservers.yml  # Web server variables
│   │   └── databases.yml   # Database variables
│   └── host_vars/
│       └── server1.yml     # Host-specific variables
├── playbooks/
│   ├── deploy.yml          # Main deployment playbook
│   ├── setup.yml           # Initial server setup
│   └── rollback.yml        # Rollback playbook
├── roles/
│   ├── common/             # Common configuration
│   ├── nginx/              # Nginx web server
│   ├── nodejs/             # Node.js application
│   └── postgresql/         # PostgreSQL database
└── files/                  # Static files
```

## Inventory Configuration

### 1. Static Inventory

```ini
# inventory/hosts
[webservers]
web1.example.com ansible_host=192.168.1.10
web2.example.com ansible_host=192.168.1.11

[databases]
db1.example.com ansible_host=192.168.1.20
db2.example.com ansible_host=192.168.1.21

[loadbalancers]
lb1.example.com ansible_host=192.168.1.30

[production:children]
webservers
databases
loadbalancers

[webservers:vars]
ansible_user=deploy
ansible_port=22
ansible_python_interpreter=/usr/bin/python3
```

### 2. Dynamic Inventory (AWS)

```yaml
# inventory/aws_ec2.yml
plugin: amazon.aws.aws_ec2
regions:
  - us-east-1
  - us-west-2
filters:
  tag:Environment: production
keyed_groups:
  - key: tags.Role
    prefix: role
  - key: tags.Environment
    prefix: env
hostnames:
  - tag:Name
  - private-ip-address
```

## Deployment Playbook

### 1. Main Deployment Playbook

```yaml
# playbooks/deploy.yml
---
- name: Deploy Application
  hosts: webservers
  become: yes
  serial: 1  # Rolling deployment (one at a time)
  max_fail_percentage: 0

  vars:
    app_name: myapp
    app_version: "{{ version | default('latest') }}"
    deploy_user: deploy
    deploy_path: "/opt/{{ app_name }}"

  pre_tasks:
    - name: Check server health
      uri:
        url: "http://{{ ansible_host }}:3000/health"
        status_code: 200
      register: health_check
      ignore_errors: yes
      tags: [health]

    - name: Remove from load balancer
      community.aws.elb_instance:
        instance_id: "{{ ec2_id }}"
        state: absent
        wait: yes
      delegate_to: localhost
      tags: [loadbalancer]

  roles:
    - role: common
      tags: [common]
    - role: nodejs
      tags: [nodejs]
    - role: nginx
      tags: [nginx]

  tasks:
    - name: Stop application
      systemd:
        name: "{{ app_name }}"
        state: stopped
      tags: [deploy]

    - name: Pull latest code
      git:
        repo: "https://github.com/myorg/{{ app_name }}.git"
        dest: "{{ deploy_path }}"
        version: "{{ app_version }}"
      become_user: "{{ deploy_user }}"
      tags: [deploy]

    - name: Install dependencies
      npm:
        path: "{{ deploy_path }}"
        production: yes
      become_user: "{{ deploy_user }}"
      tags: [deploy]

    - name: Build application
      command: npm run build
      args:
        chdir: "{{ deploy_path }}"
      become_user: "{{ deploy_user }}"
      tags: [deploy]

    - name: Run database migrations
      command: npm run migrate
      args:
        chdir: "{{ deploy_path }}"
      environment:
        DATABASE_URL: "{{ database_url }}"
      become_user: "{{ deploy_user }}"
      run_once: yes
      tags: [database]

    - name: Start application
      systemd:
        name: "{{ app_name }}"
        state: started
        enabled: yes
      tags: [deploy]

    - name: Wait for application to be ready
      uri:
        url: "http://localhost:3000/health"
        status_code: 200
      retries: 30
      delay: 2
      tags: [health]

  post_tasks:
    - name: Add back to load balancer
      community.aws.elb_instance:
        instance_id: "{{ ec2_id }}"
        state: present
        wait: yes
      delegate_to: localhost
      tags: [loadbalancer]

    - name: Verify deployment
      uri:
        url: "http://{{ ansible_host }}:3000/health"
        status_code: 200
      tags: [health]
```

### 2. Nginx Role

```yaml
# roles/nginx/tasks/main.yml
---
- name: Install Nginx
  apt:
    name: nginx
    state: present
    update_cache: yes

- name: Configure Nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/sites-available/{{ app_name }}
  notify: reload nginx

- name: Enable site
  file:
    src: /etc/nginx/sites-available/{{ app_name }}
    dest: /etc/nginx/sites-enabled/{{ app_name }}
    state: link
  notify: reload nginx

- name: Start Nginx
  systemd:
    name: nginx
    state: started
    enabled: yes
```

```jinja2
# roles/nginx/templates/nginx.conf.j2
upstream {{ app_name }} {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name {{ domain_name }};

    location / {
        proxy_pass http://{{ app_name }};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 3. Node.js Role

```yaml
# roles/nodejs/tasks/main.yml
---
- name: Install Node.js repository
  shell: curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

- name: Install Node.js
  apt:
    name: nodejs
    state: present

- name: Create deploy user
  user:
    name: "{{ deploy_user }}"
    shell: /bin/bash
    create_home: yes

- name: Create app directory
  file:
    path: "{{ deploy_path }}"
    state: directory
    owner: "{{ deploy_user }}"
    group: "{{ deploy_user }}"
    mode: '0755'

- name: Create systemd service
  template:
    src: app.service.j2
    dest: "/etc/systemd/system/{{ app_name }}.service"
  notify: reload systemd
```

```jinja2
# roles/nodejs/templates/app.service.j2
[Unit]
Description={{ app_name }} Node.js Application
After=network.target

[Service]
Type=simple
User={{ deploy_user }}
WorkingDirectory={{ deploy_path }}
ExecStart=/usr/bin/node server.js
Restart=always
Environment=NODE_ENV=production
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
```

## Ansible Vault (Secrets Management)

```bash
# Create encrypted file
ansible-vault create secrets.yml

# Encrypt existing file
ansible-vault encrypt secrets.yml

# Edit encrypted file
ansible-vault edit secrets.yml

# Decrypt file
ansible-vault decrypt secrets.yml

# View encrypted file
ansible-vault view secrets.yml

# Use vault in playbook
ansible-playbook deploy.yml --ask-vault-pass

# Use vault password file
ansible-playbook deploy.yml --vault-password-file .vault_pass
```

```yaml
# secrets.yml (encrypted)
database_url: "postgresql://user:password@db.example.com/myapp"
jwt_secret: "supersecret123"
aws_access_key: "AKIAIOSFODNN7EXAMPLE"
aws_secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

## GitHub Actions Integration

```yaml
name: Ansible Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install Ansible
      run: |
        pip install ansible
        ansible-galaxy collection install community.general

    - name: Setup SSH key
      run: |
        mkdir -p ~/.ssh
        echo "${{ secrets.SSH_PRIVATE_KEY }}" > ~/.ssh/id_rsa
        chmod 600 ~/.ssh/id_rsa

    - name: Run Ansible Playbook
      env:
        ANSIBLE_VAULT_PASSWORD: ${{ secrets.ANSIBLE_VAULT_PASSWORD }}
      run: |
        cd ansible
        echo "$ANSIBLE_VAULT_PASSWORD" > .vault_pass
        ansible-playbook playbooks/deploy.yml \
          -i inventory/prod \
          --vault-password-file .vault_pass \
          --extra-vars "version=${{ github.sha }}"
```

## Rolling Deployment

```yaml
# Serial deployment (one at a time)
- hosts: webservers
  serial: 1
  max_fail_percentage: 0

# Batch deployment (50% at a time)
- hosts: webservers
  serial: "50%"

# Custom batch sizes
- hosts: webservers
  serial:
    - 1       # First server
    - 25%     # Then 25%
    - 100%    # Then all remaining
```

## Rollback Playbook

```yaml
# playbooks/rollback.yml
---
- name: Rollback Application
  hosts: webservers
  become: yes

  vars:
    previous_version: "{{ version }}"

  tasks:
    - name: Stop application
      systemd:
        name: "{{ app_name }}"
        state: stopped

    - name: Checkout previous version
      git:
        repo: "https://github.com/myorg/{{ app_name }}.git"
        dest: "{{ deploy_path }}"
        version: "{{ previous_version }}"

    - name: Rollback database
      command: npm run migrate:rollback
      args:
        chdir: "{{ deploy_path }}"

    - name: Start application
      systemd:
        name: "{{ app_name }}"
        state: started
```

## Best Practices

1. **Idempotency**: Write idempotent tasks (safe to run multiple times)
2. **Roles**: Use roles for reusable components
3. **Tags**: Tag tasks for selective execution
4. **Variables**: Use group_vars and host_vars for configuration
5. **Vault**: Encrypt sensitive data with Ansible Vault
6. **Handlers**: Use handlers for service restarts
7. **Check Mode**: Test with `--check` before applying
8. **Serial**: Use rolling deployments for zero downtime
9. **Delegation**: Delegate tasks to localhost when needed
10. **Facts**: Use Ansible facts for dynamic configuration

## Integration Points

- **/terraform-apply** - Infrastructure provisioning
- **/docker-compose** - Container deployment
- **/github-actions** - CI/CD automation
- **/aws-deploy** - Cloud deployment

## Agent Integration

```javascript
// Orchestrator: Coordinate deployment
mcp__ruv-swarm__task_orchestrate({
  task: "Deploy application with Ansible across 10 servers",
  strategy: "sequential"
})

// DevOps: Optimize playbooks
mcp__ruv-swarm__agent_spawn({
  type: "optimizer",
  capabilities: ["ansible", "configuration-management"]
})
```

## Related Commands

- `/terraform-apply` - Infrastructure as code
- `/docker-compose` - Container orchestration
- `/github-actions` - CI/CD workflows

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
