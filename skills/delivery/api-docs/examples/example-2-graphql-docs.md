# Example 2: GraphQL Schema Documentation

**Scenario**: Generate comprehensive GraphQL schema documentation with type definitions, queries, mutations, subscriptions, and interactive GraphQL Playground.

**Difficulty**: Intermediate
**Time**: 45-60 minutes
**Prerequisites**: GraphQL server with schema and resolvers

## 🎯 Objective

Transform a GraphQL API into fully documented, self-describing schema with:
- Type definition documentation
- Query/Mutation descriptions
- Field-level documentation
- Directive documentation
- Interactive GraphQL Playground
- Schema introspection

## 📋 Starting Point

### Existing GraphQL Schema
```graphql
# schema.graphql
type User {
  id: ID!
  email: String!
  name: String!
  role: Role!
  posts: [Post!]!
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  published: Boolean!
  tags: [String!]!
  createdAt: DateTime!
}

enum Role {
  USER
  ADMIN
  MODERATOR
}

type Query {
  users: [User!]!
  user(id: ID!): User
  posts(published: Boolean): [Post!]!
  post(id: ID!): Post
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
  createPost(input: CreatePostInput!): Post!
}

input CreateUserInput {
  email: String!
  name: String!
  role: Role
}

input UpdateUserInput {
  email: String
  name: String
  role: Role
}

input CreatePostInput {
  title: String!
  content: String!
  tags: [String!]
}
```

## 🚀 Step-by-Step Process

### Step 1: Add Schema Documentation with Descriptions

Create `schema-documented.graphql`:

```graphql
"""
Represents a user account in the system.
Users can create posts, have roles, and participate in the platform.
"""
type User {
  """
  Unique identifier for the user (UUID v4)
  Example: "123e4567-e89b-12d3-a456-426614174000"
  """
  id: ID!

  """
  User's email address (must be unique)
  Format: valid email according to RFC 5322
  Example: "user@example.com"
  """
  email: String!

  """
  Full display name of the user
  Min length: 2 characters
  Max length: 100 characters
  Example: "John Doe"
  """
  name: String!

  """
  User's role in the system (default: USER)
  Determines access permissions and capabilities
  """
  role: Role!

  """
  Collection of posts authored by this user
  Returns empty array if user has no posts
  Sorted by creation date (newest first)
  """
  posts: [Post!]!

  """
  Timestamp when the user account was created
  Format: ISO 8601 date-time
  Example: "2025-01-15T10:30:00Z"
  """
  createdAt: DateTime!
}

"""
Represents a blog post created by a user.
Posts can be published or in draft state.
"""
type Post {
  """
  Unique identifier for the post (UUID v4)
  """
  id: ID!

  """
  Title of the post
  Min length: 5 characters
  Max length: 200 characters
  """
  title: String!

  """
  Full content of the post (supports Markdown)
  Max length: 50,000 characters
  """
  content: String!

  """
  Author of the post (always present)
  Cannot be null - every post must have an author
  """
  author: User!

  """
  Publication status of the post
  - true: Post is publicly visible
  - false: Post is in draft state
  Default: false
  """
  published: Boolean!

  """
  Array of tags for categorization
  Each tag: lowercase, alphanumeric, 2-30 characters
  Example: ["javascript", "web-development"]
  """
  tags: [String!]!

  """
  Timestamp when the post was created
  Format: ISO 8601 date-time
  """
  createdAt: DateTime!
}

"""
User role enumeration defining access levels
Roles are hierarchical: ADMIN > MODERATOR > USER
"""
enum Role {
  """
  Standard user role with basic permissions
  - Can create and edit own posts
  - Can view published posts
  - Cannot moderate content
  """
  USER

  """
  Administrative role with full permissions
  - All MODERATOR permissions
  - Can manage users
  - Can access admin panel
  - Can change system settings
  """
  ADMIN

  """
  Moderator role with content management permissions
  - All USER permissions
  - Can edit/delete any post
  - Can moderate comments
  - Cannot manage users
  """
  MODERATOR
}

"""
Root query type for read operations
All queries require authentication unless otherwise noted
"""
type Query {
  """
  Retrieve all users in the system

  Permissions: Requires ADMIN or MODERATOR role
  Pagination: Returns all users (consider adding pagination for production)
  Sorting: Ordered by creation date (newest first)

  Returns: Array of User objects (empty array if no users)

  Example query:
  ```
  query {
    users {
      id
      email
      name
      role
    }
  }
  ```
  """
  users: [User!]!

  """
  Retrieve a specific user by ID

  Permissions:
  - Any authenticated user can query own profile
  - ADMIN/MODERATOR can query any user

  Args:
  - id: User ID (UUID format)

  Returns: User object or null if not found

  Example query:
  ```
  query {
    user(id: "123e4567-e89b-12d3-a456-426614174000") {
      email
      name
      posts {
        title
      }
    }
  }
  ```
  """
  user(id: ID!): User

  """
  Retrieve posts with optional filtering

  Permissions: Public (no authentication required for published posts)
  Filtering:
  - published: true (only published posts)
  - published: false (only drafts - requires authentication)
  - published: null (all posts - requires ADMIN/MODERATOR)

  Args:
  - published: Optional boolean to filter by publication status

  Returns: Array of Post objects

  Example queries:
  ```
  # Get all published posts
  query {
    posts(published: true) {
      title
      content
      author {
        name
      }
    }
  }

  # Get all posts (admin only)
  query {
    posts {
      title
      published
    }
  }
  ```
  """
  posts(published: Boolean): [Post!]!

  """
  Retrieve a specific post by ID

  Permissions:
  - Published posts: Public access
  - Draft posts: Only author or ADMIN/MODERATOR

  Args:
  - id: Post ID (UUID format)

  Returns: Post object or null if not found/unauthorized
  """
  post(id: ID!): Post
}

"""
Root mutation type for write operations
All mutations require authentication
"""
type Mutation {
  """
  Create a new user account

  Permissions: Public (registration endpoint)
  Rate limit: 5 requests per hour per IP

  Input validation:
  - email: Must be unique, valid email format
  - name: 2-100 characters
  - role: Optional, defaults to USER (only ADMIN can set ADMIN role)

  Returns: Newly created User object

  Errors:
  - EMAIL_ALREADY_EXISTS: Email is already registered
  - INVALID_EMAIL_FORMAT: Email doesn't match RFC 5322
  - PERMISSION_DENIED: Non-admin trying to create admin user

  Example mutation:
  ```
  mutation {
    createUser(input: {
      email: "newuser@example.com"
      name: "Jane Smith"
      role: USER
    }) {
      id
      email
      name
      role
    }
  }
  ```
  """
  createUser(input: CreateUserInput!): User!

  """
  Update an existing user's information

  Permissions:
  - Users can update own profile (except role)
  - ADMIN can update any user

  Args:
  - id: User ID to update
  - input: Fields to update (all optional)

  Returns: Updated User object

  Errors:
  - USER_NOT_FOUND: User ID doesn't exist
  - PERMISSION_DENIED: Insufficient permissions
  - EMAIL_ALREADY_EXISTS: Email already taken by another user
  """
  updateUser(id: ID!, input: UpdateUserInput!): User!

  """
  Delete a user account permanently

  Permissions: ADMIN only
  Side effects:
  - All user's posts are reassigned to system user
  - User's sessions are invalidated
  - Deletion is permanent and cannot be undone

  Args:
  - id: User ID to delete

  Returns: true if successful, throws error otherwise

  Errors:
  - USER_NOT_FOUND: User doesn't exist
  - PERMISSION_DENIED: Only admins can delete users
  - CANNOT_DELETE_SELF: Admins cannot delete their own account
  """
  deleteUser(id: ID!): Boolean!

  """
  Create a new blog post

  Permissions: Authenticated users only

  Input validation:
  - title: 5-200 characters, required
  - content: 1-50,000 characters, required
  - tags: Optional, each tag 2-30 characters lowercase

  Returns: Newly created Post object (defaults to published: false)

  Example mutation:
  ```
  mutation {
    createPost(input: {
      title: "Getting Started with GraphQL"
      content: "GraphQL is a query language..."
      tags: ["graphql", "tutorial"]
    }) {
      id
      title
      published
      author {
        name
      }
    }
  }
  ```
  """
  createPost(input: CreatePostInput!): Post!
}

"""
Input type for creating a new user
All fields are validated before user creation
"""
input CreateUserInput {
  """Email address (must be unique)"""
  email: String!

  """Full name (2-100 characters)"""
  name: String!

  """User role (defaults to USER if not specified)"""
  role: Role
}

"""
Input type for updating user information
All fields are optional - only provided fields will be updated
"""
input UpdateUserInput {
  """New email address (must be unique if changed)"""
  email: String

  """New full name (2-100 characters)"""
  name: String

  """New role (ADMIN permission required to change roles)"""
  role: Role
}

"""
Input type for creating a new post
Title and content are required, tags are optional
"""
input CreatePostInput {
  """Post title (5-200 characters)"""
  title: String!

  """Post content (supports Markdown, 1-50,000 characters)"""
  content: String!

  """Optional tags for categorization"""
  tags: [String!]
}

"""
Custom scalar type for ISO 8601 date-time strings
Format: YYYY-MM-DDTHH:mm:ss.sssZ
Example: "2025-01-15T10:30:00.000Z"
"""
scalar DateTime
```

### Step 2: Set Up GraphQL Playground

Install dependencies:
```bash
npm install graphql-playground-middleware-express
```

Create `server.js`:
```javascript
const express = require('express');
const { ApolloServer } = require('apollo-server-express');
const expressPlayground = require('graphql-playground-middleware-express').default;
const { readFileSync } = require('fs');
const { join } = require('path');

const typeDefs = readFileSync(join(__dirname, 'schema-documented.graphql'), 'utf8');
const resolvers = require('./resolvers');

const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: true, // Enable introspection
  playground: false, // Disable built-in playground (we'll use custom)
  context: ({ req }) => ({
    user: req.user, // From authentication middleware
    headers: req.headers
  })
});

const app = express();

async function startServer() {
  await server.start();
  server.applyMiddleware({ app, path: '/graphql' });

  // Custom GraphQL Playground with documentation
  app.get('/playground', expressPlayground({
    endpoint: '/graphql',
    settings: {
      'editor.theme': 'dark',
      'editor.fontSize': 14,
      'editor.reuseHeaders': true,
      'tracing.hideTracingResponse': false,
      'queryPlan.hideQueryPlanResponse': false,
      'editor.cursorShape': 'line',
      'request.credentials': 'include'
    },
    tabs: [
      {
        endpoint: '/graphql',
        query: `# Welcome to GraphQL Playground
#
# Explore the schema using the "Docs" panel on the right →
# Try example queries below:

# Example 1: Get all published posts
query GetPublishedPosts {
  posts(published: true) {
    id
    title
    author {
      name
      email
    }
    tags
  }
}

# Example 2: Get user with their posts
query GetUserWithPosts {
  user(id: "your-user-id-here") {
    name
    email
    role
    posts {
      title
      published
    }
  }
}

# Example 3: Create a new post
mutation CreatePost {
  createPost(input: {
    title: "My First GraphQL Post"
    content: "This is the content of my post..."
    tags: ["graphql", "tutorial"]
  }) {
    id
    title
    published
    author {
      name
    }
  }
}
`
      }
    ]
  }));

  app.listen(4000, () => {
    console.log(`🚀 Server ready at http://localhost:4000${server.graphqlPath}`);
    console.log(`📖 GraphQL Playground at http://localhost:4000/playground`);
  });
}

startServer();
```

### Step 3: Generate Schema Documentation as Markdown

Create `generate-docs.js`:
```javascript
const { printSchema, getIntrospectionQuery, graphql } = require('graphql');
const { makeExecutableSchema } = require('@graphql-tools/schema');
const fs = require('fs');
const { readFileSync } = require('fs');

const typeDefs = readFileSync('./schema-documented.graphql', 'utf8');
const schema = makeExecutableSchema({ typeDefs });

// Generate schema SDL
const schemaSDL = printSchema(schema);
fs.writeFileSync('./docs/schema.graphql', schemaSDL);

// Generate introspection JSON
graphql(schema, getIntrospectionQuery()).then(result => {
  fs.writeFileSync('./docs/schema.json', JSON.stringify(result, null, 2));
  console.log('✅ Generated schema.json for tooling');
});

console.log('✅ Generated schema.graphql');
console.log('📖 Schema documentation complete!');
```

Run:
```bash
node generate-docs.js
```

### Step 4: Add Schema Directives for Additional Documentation

Extend schema with custom directives:

```graphql
"""
Marks a field as deprecated with migration instructions
"""
directive @deprecated(
  reason: String = "This field is deprecated"
  migrateWith: String
) on FIELD_DEFINITION | ENUM_VALUE

"""
Specifies authentication requirements for field access
"""
directive @auth(
  requires: Role = USER
) on OBJECT | FIELD_DEFINITION

"""
Documents example values for fields
"""
directive @example(
  value: String!
) on FIELD_DEFINITION

"""
Specifies validation constraints
"""
directive @constraint(
  minLength: Int
  maxLength: Int
  pattern: String
  format: String
) on INPUT_FIELD_DEFINITION | FIELD_DEFINITION

# Usage examples:
type User @auth(requires: ADMIN) {
  email: String! @constraint(format: "email")
  name: String! @constraint(minLength: 2, maxLength: 100)
  oldField: String @deprecated(
    reason: "Use newField instead"
    migrateWith: "user.newField"
  )
}

type Query {
  sensitiveData: String @auth(requires: ADMIN)
  users: [User!]! @auth(requires: MODERATOR)
}
```

## ✅ Success Criteria

Your GraphQL documentation is complete when:

1. **Schema Fully Documented**:
   - All types have descriptions
   - All fields have descriptions
   - All arguments have descriptions
   - Examples provided for complex fields

2. **Playground Functional**:
   - Accessible at /playground
   - Introspection enabled
   - Example queries work
   - Auto-complete functional

3. **Documentation Generated**:
   - schema.graphql (SDL)
   - schema.json (introspection)
   - Documentation accessible in Playground

4. **Directives Implemented**:
   - @deprecated for legacy fields
   - @auth for permissions
   - @constraint for validation
   - @example for clarity

## 📊 Generated Artifacts

```
project/
├── schema-documented.graphql   # Fully documented schema
├── server.js                   # Server with Playground
├── generate-docs.js            # Documentation generator
└── docs/
    ├── schema.graphql          # Generated SDL
    └── schema.json             # Introspection JSON
```

## 🎯 Key Takeaways

1. **Triple-Quoted Strings**: Use `"""` for multi-line descriptions
2. **Field-Level Docs**: Document every field with examples
3. **Permission Documentation**: Use @auth directive for clarity
4. **Deprecation Strategy**: Guide users to new fields
5. **Interactive Testing**: Playground enables exploration

## 🚀 Next Steps

1. **Add Subscriptions**: Document real-time events
2. **Federation**: Document federated schema
3. **Custom Scalars**: Document validation rules
4. **Error Codes**: Document error types
5. **Rate Limiting**: Document limits in descriptions

## 🔗 Related Examples

- **Example 1**: REST API documentation with OpenAPI
- **Example 3**: API versioning strategies
