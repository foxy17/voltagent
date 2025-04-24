# @voltagent/google-ai

VoltAgent Google AI provider integration using the Google Generative AI SDK (`@google/genai`).

This package allows you to use Google's Generative AI models (like Gemini) within your VoltAgent agents.

## Installation

```bash
npm install @voltagent/google-ai
# or
yarn add @voltagent/google-ai
# or
pnpm add @voltagent/google-ai
```

## Usage

You need to provide your Google Generative AI API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

```typescript
import { Agent } from "@voltagent/core";
import { GoogleGenAIProvider } from "@voltagent/google-ai";

// Ensure your API key is stored securely, e.g., in environment variables
const googleApiKey = process.env.GOOGLE_API_KEY;

// Instantiate the provider
const googleProvider = new GoogleGenAIProvider({
  apiKey: googleApiKey,
});

// Create an agent using a Google model
const agent = new Agent({
  name: "Google Assistant",
  description: "A helpful and friendly assistant that can answer questions clearly and concisely.",
  llm: googleProvider,
  model: "gemini-1.5-pro-latest", // Specify the desired Google model
});

//With Vertex AI
const googleVertexProvider = new GoogleGenAIProvider({
  vertexai: true,
  project: "your-project-id",
  location: "your-project-location",
});
const agent = new Agent({
  name: "Google Assistant",
  description: "A helpful and friendly assistant that can answer questions clearly and concisely.",
  llm: googleProvider,
  model: "gemini-1.5-pro-latest", // Specify the desired Google model
});
```

## Configuration

The `GoogleGenAIProvider` accepts the following options in its constructor:

- `apiKey`: Your Google Generative AI API key (required).
- **(Advanced - Vertex AI)** `vertexai`: Set to `true` if using Vertex AI endpoints.
- **(Advanced - Vertex AI)** `project`: Your Google Cloud project ID (required if `vertexai` is `true`).
- **(Advanced - Vertex AI)** `location`: Your Google Cloud project location (required if `vertexai` is `true`).

## License

MIT
