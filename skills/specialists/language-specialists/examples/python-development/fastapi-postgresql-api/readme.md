# FastAPI + PostgreSQL API Example

Complete production-ready CRUD API with JWT authentication, database migrations, and OpenAPI documentation.

## Features

- ✅ FastAPI for high-performance async API
- ✅ SQLAlchemy 2.0 with async PostgreSQL
- ✅ Pydantic for data validation
- ✅ JWT authentication with bcrypt password hashing
- ✅ Health check endpoints
- ✅ OpenAPI/Swagger documentation

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install fastapi uvicorn[standard] sqlalchemy asyncpg pydantic-settings python-jose[cryptography] passlib[bcrypt] python-multipart
```

### 2. Setup PostgreSQL

```bash
# Using Docker
docker run --name api-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres

# Or install PostgreSQL locally
createdb api_db
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

API will be available at:
- http://localhost:8000
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## API Endpoints

### Health Check
```bash
GET /health
```

### Register User
```bash
POST /register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "testuser",
  "password": "securepassword123"
}
```

### Login
```bash
POST /token
Content-Type: application/x-www-form-urlencoded

username=testuser&password=securepassword123
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Get Current User
```bash
GET /users/me
Authorization: Bearer <access_token>
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

## Production Deployment

1. **Change SECRET_KEY**: Use a strong, random secret key
2. **Environment Variables**: Use pydantic-settings for config
3. **Database**: Use connection pooling and migrations with Alembic
4. **CORS**: Configure CORS middleware for frontend
5. **Rate Limiting**: Add slowapi or similar
6. **Logging**: Configure structured logging

## Project Structure

```
fastapi-postgresql-api/
├── main.py          # API implementation (~180 lines)
├── README.md        # This file
└── requirements.txt # Dependencies
```

## Key Patterns

- **Async/Await**: All database operations are async
- **Dependency Injection**: FastAPI's Depends() for clean code
- **Type Safety**: Full type hints with Pydantic
- **Security**: JWT tokens, bcrypt password hashing
- **Lifespan Events**: Proper startup/shutdown handling

---

**Lines**: ~180 (main.py)
**Language**: Python 3.10+
**Framework**: FastAPI
**Database**: PostgreSQL with SQLAlchemy 2.0
