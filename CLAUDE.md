# Chat Bot with RAG Project

## Business Requirements

- An MVP of a chat bot with RAG
- A Back End that receive uploaded documents and pass to API for RAG
- An Api server that perform chunking uploaded documents from back end to generate embedding vectors and stores into database. The Api server also receive prompt from front end and perform RAG to retrieve
- A Front End that receive prompt to and query
- Implement an API that receive uploaded documents from back end, chunk

## Technical Details

- Implemented front end as a modern NextJS app, client rendered
- The front end app should be created in a subdirectory `frontend` to consume api from subdirectory 'api'
- Only logged in user to do exam at front end
- Use popular libraries
- As simple as possible but with an elegant UI

## Color Scheme

- Accent Yellow: `#ecad0a` - accent lines, highlights
- Blue Primary: `#209dd7` - links, key sections
- Purple Secondary: `#753991` - submit buttons, important actions
- Dark Navy: `#032147` - main headings
- Gray Text: `#888888` - supporting text, labels

## Strategy

1. Write plan with success criteria for each phase to be checked off. Include project scaffolding, including .gitignore, and rigorous unit testing.
2. Execute the plan ensuring all critiera are met
3. Carry out extensive integration testing with Playwright or similar, fixing defects
4. Only complete when the MVP is finished and tested, with the server running and ready for the user

## Coding standards

1. Use latest versions of libraries and idiomatic approaches as of today
2. Keep it simple - NEVER over-engineer, ALWAYS simplify, NO unnecessary defensive programming. No extra features - focus on simplicity.
3. Be concise. Keep README minimal. IMPORTANT: no emojis ever
