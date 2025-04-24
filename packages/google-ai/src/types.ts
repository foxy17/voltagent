import type { GenerateContentResponse } from "@google/genai";

export type GoogleGenAIProviderOptions = {
  apiKey: string;
  vertexai?: boolean;
  project?: string;
  location?: string;
};

export type GoogleGenerateContentStreamResult = AsyncGenerator<
  GenerateContentResponse,
  any,
  unknown
>;
