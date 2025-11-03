# Example 3: Azure Functions Event-Driven Processing

Build a scalable event-driven processing pipeline using Azure Functions, Cosmos DB, and Event Grid for real-time data processing.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│           Azure Event-Driven Processing Pipeline           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐                                     │
│  │  Event Sources   │                                     │
│  ├──────────────────┤                                     │
│  │ • Blob Storage   │                                     │
│  │ • Event Grid     │                                     │
│  │ • Service Bus    │                                     │
│  │ • HTTP Triggers  │                                     │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐     ┌──────────────────┐           │
│  │  Azure Functions │────▶│  Cosmos DB       │           │
│  │  (Serverless)    │     │  (NoSQL)         │           │
│  │  • Process data  │     │  • Global dist.  │           │
│  │  • Transform     │     │  • Low latency   │           │
│  │  • Enrich        │     └──────────────────┘           │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                     │
│  │  Output Bindings │                                     │
│  ├──────────────────┤                                     │
│  │ • Event Hub      │                                     │
│  │ • Storage Queue  │                                     │
│  │ • SendGrid       │                                     │
│  └──────────────────┘                                     │
└────────────────────────────────────────────────────────────┘
```

## Use Case

Real-time image processing pipeline for a social media application:

1. User uploads image to Blob Storage
2. Blob trigger invokes Azure Function
3. Function processes image (resize, thumbnail, metadata extraction)
4. Store metadata in Cosmos DB
5. Send notification via Event Grid
6. Generate analytics events to Event Hub

## Prerequisites

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Install Terraform
brew install terraform  # macOS
```

## Project Structure

```
azure-functions-pipeline/
├── terraform/
│   ├── main.tf              # Main Terraform configuration
│   ├── variables.tf         # Input variables
│   ├── functions.tf         # Azure Functions resources
│   ├── cosmos-db.tf         # Cosmos DB account
│   └── storage.tf           # Storage accounts
├── functions/
│   ├── image-processor/
│   │   ├── function.json    # Function binding configuration
│   │   ├── index.js         # Function code
│   │   └── package.json
│   ├── metadata-extractor/
│   │   ├── function.json
│   │   └── index.js
│   └── notification-sender/
│       ├── function.json
│       └── index.js
├── host.json                # Global function app settings
├── local.settings.json      # Local development settings
└── README.md
```

## Implementation

### Step 1: Terraform Infrastructure

**terraform/main.tf**:
```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstate"
    container_name       = "tfstate"
    key                  = "azure-functions.tfstate"
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "main" {
  name     = "image-pipeline-${var.environment}-rg"
  location = var.location

  tags = {
    Environment = var.environment
    Project     = "image-processing"
  }
}

# Application Insights for monitoring
resource "azurerm_application_insights" "main" {
  name                = "image-pipeline-${var.environment}-ai"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"

  retention_in_days = 90
}
```

**terraform/functions.tf**:
```hcl
# Storage Account for Azure Functions
resource "azurerm_storage_account" "functions" {
  name                     = "funcsa${var.environment}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  blob_properties {
    versioning_enabled = true
  }
}

# Storage Account for uploaded images
resource "azurerm_storage_account" "images" {
  name                     = "imgsa${var.environment}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "GRS"

  blob_properties {
    versioning_enabled       = true
    change_feed_enabled      = true
    last_access_time_enabled = true

    container_delete_retention_policy {
      days = 30
    }

    delete_retention_policy {
      days = 30
    }
  }
}

resource "azurerm_storage_container" "uploads" {
  name                  = "uploads"
  storage_account_name  = azurerm_storage_account.images.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "processed" {
  name                  = "processed"
  storage_account_name  = azurerm_storage_account.images.name
  container_access_type = "private"
}

# App Service Plan (Consumption tier for serverless)
resource "azurerm_service_plan" "main" {
  name                = "image-pipeline-${var.environment}-asp"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  sku_name            = "Y1"  # Consumption tier
}

# Function App
resource "azurerm_linux_function_app" "main" {
  name                = "image-pipeline-${var.environment}-fa"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  service_plan_id            = azurerm_service_plan.main.id
  storage_account_name       = azurerm_storage_account.functions.name
  storage_account_access_key = azurerm_storage_account.functions.primary_access_key

  app_settings = {
    FUNCTIONS_WORKER_RUNTIME       = "node"
    WEBSITE_NODE_DEFAULT_VERSION   = "~18"
    APPINSIGHTS_INSTRUMENTATIONKEY = azurerm_application_insights.main.instrumentation_key

    # Cosmos DB connection
    COSMOS_DB_ENDPOINT   = azurerm_cosmosdb_account.main.endpoint
    COSMOS_DB_KEY        = azurerm_cosmosdb_account.main.primary_key
    COSMOS_DB_DATABASE   = azurerm_cosmosdb_sql_database.main.name
    COSMOS_DB_CONTAINER  = azurerm_cosmosdb_sql_container.images.name

    # Storage connection
    STORAGE_CONNECTION_STRING = azurerm_storage_account.images.primary_connection_string
    UPLOADS_CONTAINER         = azurerm_storage_container.uploads.name
    PROCESSED_CONTAINER       = azurerm_storage_container.processed.name

    # Event Grid
    EVENT_GRID_TOPIC_ENDPOINT = azurerm_eventgrid_topic.main.endpoint
    EVENT_GRID_TOPIC_KEY      = azurerm_eventgrid_topic.main.primary_access_key
  }

  site_config {
    application_insights_key               = azurerm_application_insights.main.instrumentation_key
    application_insights_connection_string = azurerm_application_insights.main.connection_string

    application_stack {
      node_version = "18"
    }

    cors {
      allowed_origins = var.cors_origins
    }
  }

  identity {
    type = "SystemAssigned"
  }
}

# Event Grid Topic
resource "azurerm_eventgrid_topic" "main" {
  name                = "image-events-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}
```

**terraform/cosmos-db.tf**:
```hcl
resource "azurerm_cosmosdb_account" "main" {
  name                = "image-cosmos-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  consistency_policy {
    consistency_level       = "Session"
    max_interval_in_seconds = 5
    max_staleness_prefix    = 100
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  backup {
    type                = "Continuous"
    interval_in_minutes = 240
    retention_in_hours  = 720
  }

  capabilities {
    name = "EnableServerless"  # Serverless pricing model
  }
}

resource "azurerm_cosmosdb_sql_database" "main" {
  name                = "image-processing"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
}

resource "azurerm_cosmosdb_sql_container" "images" {
  name                = "images"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_path  = "/userId"

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/\"_etag\"/?"
    }
  }
}
```

### Step 2: Azure Function Code

**functions/image-processor/function.json**:
```json
{
  "bindings": [
    {
      "name": "myBlob",
      "type": "blobTrigger",
      "direction": "in",
      "path": "uploads/{name}",
      "connection": "STORAGE_CONNECTION_STRING"
    },
    {
      "name": "outputBlob",
      "type": "blob",
      "direction": "out",
      "path": "processed/{name}",
      "connection": "STORAGE_CONNECTION_STRING"
    },
    {
      "name": "cosmosDbDocument",
      "type": "cosmosDB",
      "direction": "out",
      "databaseName": "%COSMOS_DB_DATABASE%",
      "collectionName": "%COSMOS_DB_CONTAINER%",
      "connectionStringSetting": "COSMOS_DB_CONNECTION",
      "createIfNotExists": false
    },
    {
      "name": "eventGridEvent",
      "type": "eventGrid",
      "direction": "out"
    }
  ]
}
```

**functions/image-processor/index.js**:
```javascript
const sharp = require('sharp');
const crypto = require('crypto');

module.exports = async function (context, myBlob) {
  context.log('Processing blob:', context.bindingData.name);

  try {
    // Process image with sharp
    const processedImage = await sharp(myBlob)
      .resize(1200, 1200, {
        fit: 'inside',
        withoutEnlargement: true
      })
      .jpeg({ quality: 85 })
      .toBuffer();

    // Generate thumbnail
    const thumbnail = await sharp(myBlob)
      .resize(200, 200, {
        fit: 'cover'
      })
      .jpeg({ quality: 80 })
      .toBuffer();

    // Extract metadata
    const metadata = await sharp(myBlob).metadata();

    // Calculate hash for deduplication
    const hash = crypto.createHash('sha256').update(myBlob).digest('hex');

    const imageDocument = {
      id: context.bindingData.name,
      userId: extractUserId(context.bindingData.name),
      originalName: context.bindingData.name,
      processedName: `processed-${context.bindingData.name}`,
      thumbnailName: `thumb-${context.bindingData.name}`,
      hash: hash,
      metadata: {
        format: metadata.format,
        width: metadata.width,
        height: metadata.height,
        space: metadata.space,
        channels: metadata.channels,
        depth: metadata.depth,
        density: metadata.density,
        hasAlpha: metadata.hasAlpha
      },
      size: {
        original: myBlob.length,
        processed: processedImage.length,
        thumbnail: thumbnail.length
      },
      uploadedAt: new Date().toISOString(),
      status: 'processed'
    };

    // Output bindings
    context.bindings.outputBlob = processedImage;
    context.bindings.cosmosDbDocument = imageDocument;
    context.bindings.eventGridEvent = {
      subject: `image/processed/${imageDocument.id}`,
      eventType: 'ImageProcessing.ImageProcessed',
      data: {
        imageId: imageDocument.id,
        userId: imageDocument.userId,
        processedAt: imageDocument.uploadedAt
      },
      dataVersion: '1.0'
    };

    context.log('Image processed successfully:', imageDocument.id);

  } catch (error) {
    context.log.error('Error processing image:', error);

    // Send error event
    context.bindings.eventGridEvent = {
      subject: `image/error/${context.bindingData.name}`,
      eventType: 'ImageProcessing.ProcessingFailed',
      data: {
        imageName: context.bindingData.name,
        error: error.message,
        timestamp: new Date().toISOString()
      },
      dataVersion: '1.0'
    };

    throw error;
  }
};

function extractUserId(filename) {
  // Extract user ID from filename pattern: userId_timestamp_original.jpg
  const parts = filename.split('_');
  return parts[0] || 'unknown';
}
```

**functions/image-processor/package.json**:
```json
{
  "name": "image-processor",
  "version": "1.0.0",
  "dependencies": {
    "sharp": "^0.32.0",
    "@azure/cosmos": "^3.17.0"
  }
}
```

**functions/notification-sender/function.json**:
```json
{
  "bindings": [
    {
      "name": "eventGridEvent",
      "type": "eventGridTrigger",
      "direction": "in"
    },
    {
      "name": "message",
      "type": "sendGrid",
      "direction": "out",
      "apiKey": "SENDGRID_API_KEY",
      "from": "notifications@myapp.com"
    }
  ]
}
```

**functions/notification-sender/index.js**:
```javascript
module.exports = async function (context, eventGridEvent) {
  context.log('Event Grid Event:', eventGridEvent);

  if (eventGridEvent.eventType === 'ImageProcessing.ImageProcessed') {
    const { userId, imageId } = eventGridEvent.data;

    // Send email notification
    context.bindings.message = {
      to: await getUserEmail(userId),
      subject: 'Your image has been processed',
      text: `Your image (${imageId}) has been successfully processed and is ready to view.`,
      html: `
        <h1>Image Processing Complete</h1>
        <p>Your image has been successfully processed.</p>
        <p><strong>Image ID:</strong> ${imageId}</p>
        <p><a href="https://myapp.com/images/${imageId}">View Image</a></p>
      `
    };

    context.log('Notification sent to user:', userId);
  }
};

async function getUserEmail(userId) {
  // In production, query user database
  return `user-${userId}@example.com`;
}
```

### Step 3: Local Development

**local.settings.json**:
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "node",
    "COSMOS_DB_ENDPOINT": "https://localhost:8081",
    "COSMOS_DB_KEY": "YOUR_LOCAL_COSMOS_EMULATOR_KEY",
    "COSMOS_DB_DATABASE": "image-processing",
    "COSMOS_DB_CONTAINER": "images",
    "STORAGE_CONNECTION_STRING": "UseDevelopmentStorage=true",
    "UPLOADS_CONTAINER": "uploads",
    "PROCESSED_CONTAINER": "processed"
  }
}
```

**host.json**:
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "maxTelemetryItemsPerSecond": 20
      }
    }
  },
  "extensions": {
    "eventGrid": {
      "maxEventsPerBatch": 5,
      "preferredBatchSizeInKilobytes": 64
    },
    "cosmosDB": {
      "connectionMode": "Gateway"
    }
  }
}
```

### Step 4: Deployment

```bash
# Deploy infrastructure with Terraform
cd terraform
terraform init
terraform apply -var="environment=production"

# Deploy functions
cd ../functions
func azure functionapp publish image-pipeline-production-fa

# Or use Azure DevOps Pipeline
az pipelines create --name image-processing-pipeline --yml-path azure-pipelines.yml
```

**azure-pipelines.yml**:
```yaml
trigger:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  azureSubscription: 'Azure-Service-Connection'
  functionAppName: 'image-pipeline-production-fa'

stages:
  - stage: Build
    jobs:
      - job: BuildFunctions
        steps:
          - task: NodeTool@0
            inputs:
              versionSpec: '18.x'

          - script: |
              cd functions
              npm install
              npm run build
            displayName: 'Build Functions'

          - task: ArchiveFiles@2
            inputs:
              rootFolderOrFile: '$(System.DefaultWorkingDirectory)/functions'
              includeRootFolder: false
              archiveType: 'zip'
              archiveFile: '$(Build.ArtifactStagingDirectory)/functions.zip'

          - publish: $(Build.ArtifactStagingDirectory)/functions.zip
            artifact: functions

  - stage: Deploy
    dependsOn: Build
    jobs:
      - deployment: DeployFunctions
        environment: 'production'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureFunctionApp@1
                  inputs:
                    azureSubscription: $(azureSubscription)
                    appType: 'functionAppLinux'
                    appName: $(functionAppName)
                    package: '$(Pipeline.Workspace)/functions/functions.zip'
```

## Monitoring and Observability

### Application Insights Queries

```kusto
// Function execution times
requests
| where cloud_RoleName == "image-pipeline-production-fa"
| summarize avg(duration), percentile(duration, 95) by name
| order by avg_duration desc

// Error rate
requests
| where cloud_RoleName == "image-pipeline-production-fa"
| summarize ErrorRate = 100.0 * countif(success == false) / count() by bin(timestamp, 5m)
| render timechart

// Cosmos DB operations
dependencies
| where type == "Azure DocumentDB"
| summarize count(), avg(duration) by name
| order by count_ desc
```

## Cost Optimization

### Estimated Monthly Cost

| Service | Usage | Cost |
|---------|-------|------|
| Azure Functions | 2M executions, 256MB, 5s avg | $8.00 |
| Cosmos DB | Serverless, 1M RU/s | $0.25 |
| Storage | 100GB blob, 10M operations | $2.30 |
| Application Insights | 5GB ingestion | $0.00 (free tier) |
| Event Grid | 2M events | $0.60 |
| **Total** | | **~$11.15/month** |

### Cost-Saving Tips

1. **Use Consumption plan** for variable workloads
2. **Enable Cosmos DB serverless** for unpredictable traffic
3. **Set blob lifecycle policies** to move old files to Cool/Archive tiers
4. **Use Event Grid** instead of polling for events
5. **Optimize function execution time** (reduce cold starts)

## Production Readiness Checklist

- [ ] Durable Functions for long-running workflows
- [ ] Dead letter queues for failed messages
- [ ] Monitoring and alerting with Application Insights
- [ ] Load testing with Azure Load Testing
- [ ] Security scanning with Defender for Cloud
- [ ] Cost budget alerts
- [ ] Disaster recovery plan
- [ ] API rate limiting with API Management

## Related Resources

- [Azure Functions Documentation](https://docs.microsoft.com/azure/azure-functions/)
- [Cosmos DB Best Practices](https://docs.microsoft.com/azure/cosmos-db/best-practice-guide)
- [Event Grid Overview](https://docs.microsoft.com/azure/event-grid/overview)
