# Chat Bot Front End - Project Spec

A single-user front end that allow taking prompt and perform RAG via written api

This stack is deliberately kept user-friendly:

## Tech stack

- Language: Python 3.12+
- Web framework: Streamlit

## API call

Use the api end points that written in api:
All endpoints are prefixed with /api. Request and response bodies are Pydantic
models. Endpoints are synchronous (def). Use appropriate HTTP status codes; do
not add auth. 5. POST /api/rag/prompt

- Accepts {query, top_k}.
- Runs retrieval (as above), then generates a grounded answer with the
  local Ollama model.
- Returns {answer, sources: [{document_id, filename, content}]}.
  The answer must be grounded only in the retrieved chunks; cite sources.

## Coding standards

1. Use latest mutually compatible library versions, pinned, with idiomatic
   modern Python. Keep everything synchronous.
2. Apply SOLID, but keep it simple. NEVER over-engineer, ALWAYS simplify, NO
   unnecessary defensive programming. Validate genuine contract violations
   (e.g. missing file, unknown document_id) and let everything else fail
   naturally. No extra features beyond this spec.
3. Be concise. Keep the README minimal. No emojis, ever.
4. Every class and function has a docstring with Args, Returns, and Raises (when
   it raises), following this form:

   def divide(dividend: float, divisor: float) -> float:
   """Divide a number by another number.
   Args:
   dividend: The number to be divided.
   divisor: The number to divide by.
   Returns:
   The resulting quotient.
   Raises:
   ValueError: If the divisor is zero.
   """
   if divisor == 0:
   raise ValueError("Cannot divide by zero.")
   return dividend / divisor

## Testing strategy

- Unit tests with pytest. Mock the api so that tests are fast and offline.

## Out of scope (non-goals)

Do not build any of the following: authentication or authorization, rate
limiting, multi-tenancy or per-user isolation, streaming responses, async
endpoints, a frontend, background job queues, caching layers, or hybrid/sparse
retrieval. Single user, synchronous request/response only.

## Delivery strategy

1. Write a phased plan with check-off success criteria per phase, covering:
   project scaffolding (pyproject.toml, .gitignore, .env.example, package
   layout), database schema and init, each endpoint, and rigorous unit tests.
2. Execute the plan, ensuring every criterion is met.
3. Carry out integration testing with pytest + TestClient, fixing all defects.
4. Define E2E tests and their success criteria (full flow: upload -> list ->
   retrieve -> prompt -> remove).
5. Only complete when the MVP is finished and tested, with the server running
   and ready for the user.

## Acceptance criteria (MVP done when all true)
