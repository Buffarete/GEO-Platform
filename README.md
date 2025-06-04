# GEO Platform

This repository contains a multi-tenant GEO analytics SaaS. The stack uses FastAPI for the backend and React for the frontend. Infrastructure is provisioned with Terraform and Docker.

## Directories
- `backend/` – FastAPI application
- `frontend/` – React application (placeholder)
- `infra/` – Infrastructure as code
- `ci/` – Continuous integration configuration

## Getting Started
```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Endpoints
- `GET /health` – service health check
- `POST /llm/openai` – query OpenAI chat model (requires OpenAI API key)
