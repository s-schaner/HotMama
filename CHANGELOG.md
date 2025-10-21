# Changelog

All notable changes to VolleySense will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Security & Validation
- **Input Validation**: Comprehensive file upload validation with configurable size and type restrictions
- **Rate Limiting**: Per-IP API rate limiting using slowapi (default: 10 requests/minute)
- **Secure Temp Files**: Cryptographically secure temporary file naming using secrets.token_hex()
- **Structured Error Handling**: Consistent error responses with proper HTTP status codes and error details
- **CORS Support**: Optional CORS middleware with configurable origins

#### Configuration & Settings
- **Centralized Settings**: New `app/settings.py` module using pydantic-settings for environment-based configuration
- **Environment Variables**: Support for `.env` files with comprehensive configuration options
- **Settings Documentation**: `.env.example` file documenting all configuration options
- **Validation**: Input validation module (`app/validators.py`) with reusable validators

#### API Improvements
- **Health Check Endpoint**: `/health` endpoint for monitoring and container orchestration
- **Async LLM Client**: New async HTTP client with retry logic and exponential backoff
- **Better Error Messages**: Error responses now include error type, detail, and machine-readable codes
- **HTMX-Aware Errors**: Returns HTML errors for HTMX requests, JSON for API requests

#### Developer Experience
- **CI/CD Pipeline**: GitHub Actions workflow with linting, testing, Docker build, and security scanning
- **Multiple Python Versions**: CI tests on Python 3.10 and 3.11
- **Security Scanning**: Automated bandit security scanning in CI
- **Code Coverage**: Codecov integration for test coverage reporting
- **API Documentation**: Enhanced OpenAPI docs (accessible at `/docs` in debug mode)

#### Dependencies
- **httpx**: Async HTTP client for LLM API calls
- **tenacity**: Retry logic with exponential backoff
- **slowapi**: API rate limiting middleware
- **pydantic-settings**: Environment-based configuration management
- **python-dotenv**: .env file support

### Changed

#### Breaking Changes
- **Pydantic v2**: Upgraded from Pydantic v1 (1.10.15) to v2 (2.6.3) for better performance and validation
  - This may require schema updates for projects using custom models

#### Improvements
- **Async Video Analysis**: The `/_api/analyze` endpoint now uses async/await throughout
- **Better Logging**: Enhanced logging with structured context throughout the application
- **Secure Cleanup**: Temporary files now cleaned up using Path.unlink() with proper error suppression
- **Settings Injection**: Endpoints use FastAPI's Depends() for settings injection
- **Error Context**: Exceptions now logged with full context using logger.exception()

#### Refactoring
- **Error Handling Module**: Centralized error handling in `app/error_handlers.py`
- **Validation Module**: Reusable validators in `app/validators.py`
- **Async Client**: Separated async LLM client into `llm/async_client.py`
- **Middleware Stack**: Properly ordered middleware (CORS, rate limiting, error handling)

### Fixed

- **Dependency Conflicts**: Resolved Pydantic v1/v2 version mismatch with FastAPI
- **Temp File Security**: Predictable temp file names replaced with secure random names
- **Error Leakage**: Sensitive error details no longer exposed in production (controlled by debug flag)
- **Resource Cleanup**: Improved temp file cleanup with contextlib.suppress()
- **Type Hints**: Added missing type hints in multiple modules

### Security

- **File Upload Validation**: Prevents upload of dangerous file types and oversized files
- **Rate Limiting**: Prevents API abuse and DoS attacks
- **Secure Temp Files**: Uses cryptographically secure random filenames
- **Error Sanitization**: Sensitive error details hidden unless debug mode enabled
- **Dependency Updates**: Updated dependencies to latest secure versions

### Documentation

- **README Updates**: Added sections on configuration, security, API endpoints, and CI/CD
- **Environment Variables**: Comprehensive `.env.example` with all configuration options
- **API Documentation**: Enhanced OpenAPI/Swagger documentation
- **Health Check**: Documented health check endpoint for monitoring
- **CI/CD**: Documented GitHub Actions pipeline

## Version History

### [1.0.0] - Previous Release
- Initial VolleySense release with:
  - FastAPI web interface
  - Video analysis with vision-language models
  - Heatmap generation (3 rendering strategies)
  - Plugin architecture
  - SQLite session storage
  - Docker containerization
  - Ball trajectory detection
  - Court calibration support

---

## Migration Guide

### Upgrading to Latest Version

#### 1. Update Dependencies

```bash
python -m tools.install --profile cpu  # or nvidia/amd
```

#### 2. Create Configuration File

```bash
cp .env.example .env
# Edit .env to customize settings
```

#### 3. Review Pydantic v2 Changes

If you've created custom Pydantic models, review the [Pydantic v2 migration guide](https://docs.pydantic.dev/2.0/migration/).

Key changes:
- `BaseModel.dict()` → `BaseModel.model_dump()`
- `BaseModel.parse_obj()` → `BaseModel.model_validate()`
- Config class → `model_config` class attribute

#### 4. Test Your Installation

```bash
# Run tests
pytest -v

# Check health endpoint
python -m app.main &
curl http://localhost:7860/health
```

### Breaking Changes

None for end users. API endpoints remain backward compatible.

For developers extending VolleySense:
- Custom models need updating for Pydantic v2
- Error handling may need adjustment to use new structured errors
