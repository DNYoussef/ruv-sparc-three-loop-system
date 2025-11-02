# Example 3: API Versioning Strategies & Documentation

**Scenario**: Implement and document multiple API versioning strategies (URL, header, query parameter) with deprecation notices, migration guides, and version-specific documentation.

**Difficulty**: Advanced
**Time**: 60-90 minutes
**Prerequisites**: Existing REST or GraphQL API, OpenAPI/Swagger setup

## 🎯 Objective

Create a comprehensive API versioning system with:
- Multiple versioning strategies (URL, header, query)
- Version-specific OpenAPI specs
- Deprecation notices and sunset dates
- Migration guides between versions
- Backward compatibility testing
- Version negotiation logic

## 📋 Versioning Strategies Overview

### Strategy 1: URL Versioning (Most Common)
```
GET /api/v1/users
GET /api/v2/users
GET /api/v3/users
```

**Pros**: Clear, cacheable, easy to test
**Cons**: URL pollution, harder to deprecate

### Strategy 2: Header Versioning (RESTful)
```
GET /api/users
Headers: Accept: application/vnd.api+json; version=1
```

**Pros**: Clean URLs, standard HTTP
**Cons**: Harder to test in browser, cache complexity

### Strategy 3: Query Parameter Versioning
```
GET /api/users?api_version=1
GET /api/users?api_version=2
```

**Pros**: Easy to test, backward compatible
**Cons**: Optional parameters can be ignored

## 🚀 Step-by-Step Implementation

### Step 1: Set Up URL-Based Versioning (Primary Strategy)

Create `config/versioning.js`:
```javascript
const VERSIONS = {
  v1: {
    version: '1.0.0',
    released: '2024-01-01',
    deprecated: '2025-06-01',
    sunset: '2025-12-31',
    status: 'deprecated',
    breaking_changes: [],
    notes: 'Legacy version, migrate to v2 by December 2025'
  },
  v2: {
    version: '2.0.0',
    released: '2024-06-01',
    deprecated: null,
    sunset: null,
    status: 'stable',
    breaking_changes: [
      'User.id changed from integer to UUID',
      'Pagination now uses cursor-based instead of offset',
      'Date fields now ISO 8601 strings instead of Unix timestamps'
    ],
    notes: 'Current stable version with enhanced features'
  },
  v3: {
    version: '3.0.0',
    released: '2025-01-01',
    deprecated: null,
    sunset: null,
    status: 'beta',
    breaking_changes: [
      'GraphQL-first design',
      'REST endpoints maintain compatibility',
      'New authentication flow (OAuth2 + JWT)'
    ],
    notes: 'Beta version, subject to changes'
  }
};

const DEFAULT_VERSION = 'v2';
const LATEST_VERSION = 'v3';

module.exports = {
  VERSIONS,
  DEFAULT_VERSION,
  LATEST_VERSION,
  isDeprecated: (version) => VERSIONS[version]?.status === 'deprecated',
  getSunsetDate: (version) => VERSIONS[version]?.sunset,
  getLatestStable: () => Object.keys(VERSIONS).find(
    v => VERSIONS[v].status === 'stable'
  )
};
```

### Step 2: Create Version Negotiation Middleware

Create `middleware/versionNegotiation.js`:
```javascript
const { VERSIONS, DEFAULT_VERSION, LATEST_VERSION } = require('../config/versioning');

/**
 * API Version Negotiation Middleware
 * Supports multiple versioning strategies:
 * 1. URL path: /api/v2/users
 * 2. Header: Accept: application/vnd.api.v2+json
 * 3. Query: ?api_version=2
 */
function versionNegotiation(req, res, next) {
  let version = DEFAULT_VERSION;
  let versionSource = 'default';

  // Strategy 1: URL path versioning (highest priority)
  const urlMatch = req.path.match(/\/v(\d+)\//);
  if (urlMatch) {
    version = `v${urlMatch[1]}`;
    versionSource = 'url';
  }

  // Strategy 2: Header versioning
  const acceptHeader = req.get('Accept');
  if (!urlMatch && acceptHeader) {
    const headerMatch = acceptHeader.match(/application\/vnd\.api\.v(\d+)\+json/);
    if (headerMatch) {
      version = `v${headerMatch[1]}`;
      versionSource = 'header';
    }
  }

  // Strategy 3: Query parameter versioning
  if (!urlMatch && !versionSource.includes('header') && req.query.api_version) {
    version = `v${req.query.api_version}`;
    versionSource = 'query';
  }

  // Validate version exists
  if (!VERSIONS[version]) {
    return res.status(400).json({
      error: 'Invalid API version',
      message: `Version '${version}' does not exist`,
      available_versions: Object.keys(VERSIONS),
      latest_stable: VERSIONS[DEFAULT_VERSION].version
    });
  }

  // Check if version is deprecated
  const versionInfo = VERSIONS[version];
  if (versionInfo.status === 'deprecated') {
    res.set({
      'X-API-Warn': `Version ${version} is deprecated`,
      'X-API-Sunset': versionInfo.sunset,
      'X-API-Migration-Guide': `/docs/migration/${version}-to-${DEFAULT_VERSION}`,
      'Deprecation': `version="${version}"`,
      'Sunset': versionInfo.sunset
    });
  }

  // Set version info in request
  req.apiVersion = version;
  req.apiVersionInfo = versionInfo;
  req.apiVersionSource = versionSource;

  // Add version headers to response
  res.set({
    'X-API-Version': versionInfo.version,
    'X-API-Version-Source': versionSource,
    'X-API-Latest-Version': VERSIONS[LATEST_VERSION].version
  });

  next();
}

module.exports = versionNegotiation;
```

### Step 3: Create Version-Specific Route Handlers

Create `routes/users/index.js`:
```javascript
const express = require('express');
const router = express.Router();
const v1Handler = require('./v1');
const v2Handler = require('./v2');
const v3Handler = require('./v3');

// Route to version-specific handlers
router.use((req, res, next) => {
  switch(req.apiVersion) {
    case 'v1':
      return v1Handler(req, res, next);
    case 'v2':
      return v2Handler(req, res, next);
    case 'v3':
      return v3Handler(req, res, next);
    default:
      return res.status(400).json({ error: 'Unsupported API version' });
  }
});

module.exports = router;
```

Create `routes/users/v1.js`:
```javascript
/**
 * Users API v1 (DEPRECATED)
 * Sunset date: 2025-12-31
 * Migration guide: /docs/migration/v1-to-v2
 */
module.exports = async (req, res, next) => {
  if (req.method === 'GET' && req.path === '/users') {
    // V1: Returns integer IDs, Unix timestamps
    const users = await User.find();
    return res.json({
      success: true,
      data: users.map(u => ({
        id: u.sequenceId, // Integer ID (v1 format)
        email: u.email,
        name: u.name,
        created_at: Math.floor(u.createdAt.getTime() / 1000) // Unix timestamp
      })),
      // V1: Offset pagination
      pagination: {
        page: parseInt(req.query.page) || 1,
        limit: parseInt(req.query.limit) || 10,
        total: await User.countDocuments()
      }
    });
  }
  next();
};
```

Create `routes/users/v2.js`:
```javascript
/**
 * Users API v2 (STABLE)
 * Current stable version
 */
module.exports = async (req, res, next) => {
  if (req.method === 'GET' && req.path === '/users') {
    // V2: Returns UUID IDs, ISO 8601 timestamps
    const { cursor, limit = 10 } = req.query;

    const query = cursor ? { _id: { $gt: cursor } } : {};
    const users = await User.find(query).limit(parseInt(limit));

    return res.json({
      success: true,
      data: users.map(u => ({
        id: u.id, // UUID (v2 format)
        email: u.email,
        name: u.name,
        role: u.role,
        createdAt: u.createdAt.toISOString() // ISO 8601
      })),
      // V2: Cursor-based pagination
      pagination: {
        cursor: users[users.length - 1]?.id,
        limit: parseInt(limit),
        hasMore: users.length === parseInt(limit)
      }
    });
  }
  next();
};
```

Create `routes/users/v3.js`:
```javascript
/**
 * Users API v3 (BETA)
 * Enhanced with GraphQL-style capabilities
 */
module.exports = async (req, res, next) => {
  if (req.method === 'GET' && req.path === '/users') {
    // V3: Adds field selection, filtering, sorting
    const {
      cursor,
      limit = 10,
      fields, // CSV: id,email,name
      filter, // JSON: {"role": "admin"}
      sort    // id:asc, createdAt:desc
    } = req.query;

    // Parse field selection
    const selectFields = fields ? fields.split(',') : null;

    // Parse filters
    const filterQuery = filter ? JSON.parse(filter) : {};
    if (cursor) filterQuery._id = { $gt: cursor };

    // Parse sorting
    const sortQuery = sort ?
      sort.split(',').reduce((acc, s) => {
        const [field, order] = s.split(':');
        acc[field] = order === 'desc' ? -1 : 1;
        return acc;
      }, {}) :
      { createdAt: -1 };

    const users = await User
      .find(filterQuery)
      .select(selectFields)
      .sort(sortQuery)
      .limit(parseInt(limit));

    return res.json({
      success: true,
      data: users.map(u => ({
        id: u.id,
        email: u.email,
        name: u.name,
        role: u.role,
        createdAt: u.createdAt.toISOString(),
        // V3: Add hypermedia links
        _links: {
          self: `/api/v3/users/${u.id}`,
          posts: `/api/v3/users/${u.id}/posts`
        }
      })),
      pagination: {
        cursor: users[users.length - 1]?.id,
        limit: parseInt(limit),
        hasMore: users.length === parseInt(limit)
      },
      // V3: Add metadata
      meta: {
        apiVersion: 'v3',
        responseTime: Date.now() - req.startTime,
        fieldsRequested: selectFields || 'all'
      }
    });
  }
  next();
};
```

### Step 4: Generate Version-Specific OpenAPI Documentation

Create `swagger/v1.config.js`:
```javascript
module.exports = {
  openapi: '3.0.0',
  info: {
    title: 'User Management API v1',
    version: '1.0.0',
    description: `
## ⚠️ DEPRECATED VERSION

**Sunset Date**: December 31, 2025
**Status**: Maintenance mode only - critical bugs will be fixed
**Migration Guide**: [v1 to v2 Migration](../migration/v1-to-v2.md)

### Breaking Changes in v2
- User IDs changed from integers to UUIDs
- Timestamps changed from Unix to ISO 8601
- Pagination changed from offset to cursor-based

### Why Upgrade?
- Better performance with cursor pagination
- UUID support for distributed systems
- Standard ISO 8601 date handling
- Enhanced error messages
    `,
    'x-deprecated': true,
    'x-sunset-date': '2025-12-31',
    'x-migration-guide': '/docs/migration/v1-to-v2'
  },
  servers: [
    {
      url: 'https://api.example.com/v1',
      description: 'Production v1 (Deprecated)'
    }
  ]
};
```

Create `swagger/v2.config.js`:
```javascript
module.exports = {
  openapi: '3.0.0',
  info: {
    title: 'User Management API v2',
    version: '2.0.0',
    description: `
## 📘 Current Stable Version

**Status**: Stable - recommended for all new integrations
**Released**: June 1, 2024

### Key Features
- UUID-based identifiers
- ISO 8601 timestamps
- Cursor-based pagination
- Enhanced error handling
- Rate limiting headers

### Upgrade from v1
See [Migration Guide](../migration/v1-to-v2.md) for detailed instructions.
    `
  },
  servers: [
    {
      url: 'https://api.example.com/v2',
      description: 'Production v2 (Stable)'
    }
  ]
};
```

Create `swagger/v3.config.js`:
```javascript
module.exports = {
  openapi: '3.0.0',
  info: {
    title: 'User Management API v3',
    version: '3.0.0-beta',
    description: `
## 🚀 Beta Version

**Status**: Beta - API subject to changes
**Released**: January 1, 2025

### New Capabilities
- GraphQL-style field selection
- Advanced filtering and sorting
- Hypermedia links (HATEOAS)
- Response metadata
- Enhanced performance

### What's Different from v2
- Field selection: \`?fields=id,email,name\`
- JSON filtering: \`?filter={"role":"admin"}\`
- Sorting: \`?sort=createdAt:desc,name:asc\`
- Hypermedia \`_links\` in responses
- Performance \`meta\` object

### Stability Notice
⚠️ This is a beta version. Production applications should use v2.
    `,
    'x-status': 'beta',
    'x-stability': 'experimental'
  },
  servers: [
    {
      url: 'https://api.example.com/v3',
      description: 'Production v3 (Beta)'
    }
  ]
};
```

### Step 5: Create Migration Guides

Create `docs/migration/v1-to-v2.md`:
```markdown
# Migration Guide: API v1 → v2

## Overview
This guide helps you migrate from deprecated v1 to stable v2.

**Estimated time**: 2-4 hours for typical integration
**Difficulty**: Moderate
**Backward compatibility**: None - v2 is breaking

## Breaking Changes

### 1. User ID Format
**v1**: Integer (e.g., 12345)
**v2**: UUID (e.g., "123e4567-e89b-12d3-a456-426614174000")

```javascript
// v1
const userId = 12345;
const url = `/api/v1/users/${userId}`;

// v2
const userId = "123e4567-e89b-12d3-a456-426614174000";
const url = `/api/v2/users/${userId}`;
```

**Migration strategy**:
1. Add UUID column to users table
2. Maintain integer ID for internal use
3. Update all API calls to use UUID
4. Remove integer ID references after migration

### 2. Timestamp Format
**v1**: Unix timestamp (seconds since epoch)
**v2**: ISO 8601 string

```javascript
// v1
{
  "created_at": 1641038400 // Unix timestamp
}

// v2
{
  "createdAt": "2022-01-01T12:00:00.000Z" // ISO 8601
}
```

**Migration strategy**:
```javascript
// Convert v1 to v2 format
const v1Timestamp = 1641038400;
const v2Timestamp = new Date(v1Timestamp * 1000).toISOString();

// Convert v2 to v1 format (if needed for legacy systems)
const v2Timestamp = "2022-01-01T12:00:00.000Z";
const v1Timestamp = Math.floor(new Date(v2Timestamp).getTime() / 1000);
```

### 3. Pagination Strategy
**v1**: Offset-based (`page` and `limit`)
**v2**: Cursor-based (`cursor` and `limit`)

```javascript
// v1
GET /api/v1/users?page=2&limit=10
Response: {
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 10,
    "total": 150
  }
}

// v2
GET /api/v2/users?cursor=user_123&limit=10
Response: {
  "data": [...],
  "pagination": {
    "cursor": "user_130",
    "limit": 10,
    "hasMore": true
  }
}
```

**Migration strategy**:
```javascript
// Implement cursor pagination wrapper
async function fetchAllUsers() {
  let allUsers = [];
  let cursor = null;

  while (true) {
    const params = cursor ? `?cursor=${cursor}&limit=50` : '?limit=50';
    const response = await fetch(`/api/v2/users${params}`);
    const data = await response.json();

    allUsers = allUsers.concat(data.data);

    if (!data.pagination.hasMore) break;
    cursor = data.pagination.cursor;
  }

  return allUsers;
}
```

## Migration Checklist

- [ ] Update base URL from `/v1` to `/v2`
- [ ] Change user ID references from integer to UUID
- [ ] Update timestamp parsing (Unix → ISO 8601)
- [ ] Implement cursor-based pagination
- [ ] Update error handling (new error codes)
- [ ] Test all API endpoints with v2
- [ ] Update documentation
- [ ] Deploy to staging
- [ ] Run integration tests
- [ ] Deploy to production
- [ ] Monitor for errors
- [ ] Remove v1 code after validation

## Testing Strategy

```javascript
// Parallel testing: Run both versions
async function testMigration() {
  const [v1Data, v2Data] = await Promise.all([
    fetch('/api/v1/users?page=1&limit=10'),
    fetch('/api/v2/users?limit=10')
  ]);

  // Verify data consistency
  const v1Users = await v1Data.json();
  const v2Users = await v2Data.json();

  console.assert(
    v1Users.data.length === v2Users.data.length,
    'User counts should match'
  );
}
```

## Support
- Questions: support@example.com
- Migration issues: [GitHub Issues](https://github.com/example/api/issues)
- Slack channel: #api-migration
```

## ✅ Success Criteria

Your API versioning is complete when:

1. **Multiple Strategies Work**: URL, header, and query parameter versioning
2. **Deprecation Clear**: Sunset headers present for deprecated versions
3. **Documentation Split**: Separate OpenAPI specs per version
4. **Migration Guides**: Complete guides for each version transition
5. **Version Negotiation**: Automatic version detection and routing
6. **Testing**: Version-specific test suites pass

## 📊 Generated Artifacts

```
project/
├── config/
│   └── versioning.js           # Version definitions
├── middleware/
│   └── versionNegotiation.js   # Multi-strategy negotiation
├── routes/
│   └── users/
│       ├── index.js            # Version router
│       ├── v1.js               # v1 handler
│       ├── v2.js               # v2 handler
│       └── v3.js               # v3 handler
├── swagger/
│   ├── v1.config.js            # v1 OpenAPI spec
│   ├── v2.config.js            # v2 OpenAPI spec
│   └── v3.config.js            # v3 OpenAPI spec
└── docs/
    └── migration/
        ├── v1-to-v2.md         # v1→v2 guide
        └── v2-to-v3.md         # v2→v3 guide
```

## 🎯 Key Takeaways

1. **Multi-Strategy Support**: Support multiple versioning methods
2. **Clear Deprecation**: Use sunset headers and notices
3. **Migration Guides**: Provide detailed upgrade instructions
4. **Version Negotiation**: Auto-detect and route to correct version
5. **Documentation Split**: Separate docs per version
6. **Backward Compatibility**: Test thoroughly before breaking

## 🚀 Next Steps

1. **Version Analytics**: Track version usage
2. **Auto-Migration**: Tools to auto-convert requests
3. **Client Libraries**: Version-specific SDKs
4. **Monitoring**: Alert on deprecated version spikes
5. **Sunset Automation**: Auto-disable after sunset date

## 🔗 Related Examples

- **Example 1**: REST API documentation with OpenAPI
- **Example 2**: GraphQL schema documentation
