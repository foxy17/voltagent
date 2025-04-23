import type {
    BaseMessage,
    GenerateTextOptions,
    LLMProvider,
    MessageRole,
    ProviderTextResponse,
    ProviderTextStreamResponse,
    StepWithContent,
    StreamTextOptions,
    UsageInfo,
    ProviderOptions,
  } from "@voltagent/core";
import {
    GoogleGenAI, 
    type Content,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GenerateContentConfig,
  } from "@google/genai";
import type { z } from "zod";
import { GoogleGenAIProviderOptions } from "./types";

// Define the structure returned by generateContentStream if needed elsewhere
type GoogleGenerateContentStreamResult = {
  stream: AsyncIterable<GenerateContentResponse>;
  response: Promise<GenerateContentResponse>;
};

export class GoogleGenAIProvider
    implements LLMProvider<string>
  {
    private ai: GoogleGenAI;
  
    constructor(options: GoogleGenAIProviderOptions) {
      const hasApiKey = !!options?.apiKey;
      const hasVertexAIConfig = options.vertexai && options.project && options.location;

      if (!hasApiKey && !hasVertexAIConfig) {
        throw new Error(
          "Google GenAI API key is required, or if using Vertex AI, both project and location must be specified."
        );
      }

      this.ai = new GoogleGenAI({apiKey: options.apiKey,vertexai: options.vertexai,project: options.project,location: options.location});  
  
      // Bind methods to preserve 'this' context
      this.generateText = this.generateText.bind(this);
      this.streamText = this.streamText.bind(this);
      this.toMessage = this.toMessage.bind(this);
      this._createStepFromChunk = this._createStepFromChunk.bind(this);
      this.getModelIdentifier = this.getModelIdentifier.bind(this);
    }

    getModelIdentifier = (model: string): string => {
      return model;
    };
  
    // Helper to convert BaseMessage role to Google Gen AI role
    private toGoogleRole(role: MessageRole): "user" | "model" {
      switch (role) {
        case "user":
          return "user";
        case "assistant":
          return "model";
        case "system":
          return "model";  
        case "tool":
          console.warn(
            `Tool role conversion not fully implemented yet. Defaulting to 'user'.`,
          );
          return "model";  
        default:
          console.warn(
            `Unsupported role conversion for: ${role}. Defaulting to 'user'.`,
          );
          return "user";
      }
    }
  
    toMessage = (message: BaseMessage): Content => {
      const role = this.toGoogleRole(message.role);
      if (typeof message.content === "string") {
        if (role !== "user" && role !== "model") {
          throw new Error(
            `Invalid role '${role}' passed to toMessage for non-system message.`,
          );
        }
        return { role, parts: [{ text: message.content }] };
      } else {
        const parts = message.content.map((part: any) => { 
          if (part.type === "text") {
            return { text: part.text };
          }
          return { text: `[Unsupported part type: ${part.type}]` };
        });
        // Ensure role is 'user' or 'model' before returning
        if (role !== "user" && role !== "model") {
          throw new Error(
            `Invalid role '${role}' passed to toMessage for non-system message.`,
          );
        }
        return { role, parts };
      }
    };
  
  
     private _createStepFromChunk(chunkResponse: GenerateContentResponse, role: MessageRole = "assistant", usage?: UsageInfo ): StepWithContent | null {
      const text = chunkResponse.text;
  
      if (text) {
          return {
              id: "",
              type: "text",
              content: text,
              role: role,
              usage: usage,
          };
      }
  
      return null;
  }
  
  
    generateText = async (
      options: GenerateTextOptions<string>,
    ): Promise<ProviderTextResponse<GenerateContentResponse>> => {
      const model = options.model;
      const contents = options.messages.map(this.toMessage);
      const providerOptions: ProviderOptions = options.provider || {};

  
      const config: GenerateContentConfig = {
        temperature: providerOptions.temperature,
        topP: providerOptions.topP,
        topK: providerOptions.topK,
        maxOutputTokens: providerOptions.maxOutputTokens,
        stopSequences: providerOptions.stopSequences,
        safetySettings: providerOptions.safetySettings,
        candidateCount: providerOptions.candidateCount,
        responseMimeType: providerOptions.responseMimeType,
        seed: providerOptions.seed,
        presencePenalty: providerOptions.presencePenalty,
        frequencyPenalty: providerOptions.frequencyPenalty,
        responseSchema: providerOptions.responseSchema,
        ...(providerOptions.extraOptions && providerOptions.extraOptions),
      };
  
      const generationParams: GenerateContentParameters = {
        model: model,
        contents: contents,
        ...(Object.keys(config).length > 0 && { config: config }),
      };
  
      const result = await this.ai.models.generateContent(generationParams);
  
      const responseText = result.text; 
      const usageInfo =  result?.usageMetadata;  
      const finishReason = result.candidates?.[0]?.finishReason?.toString(); 

      if (options.onStepFinish) {
        const step = this._createStepFromChunk(result, "assistant", usageInfo);
        if(step) await options.onStepFinish(step);
      }

      const providerResponse: ProviderTextResponse<GenerateContentResponse> = {
          provider: result,  
          text: responseText ?? "",  
          usage: usageInfo ? {
            promptTokens: usageInfo.promptTokenCount ,
            completionTokens: usageInfo.candidatesTokenCount ?? 0,
            totalTokens: usageInfo.totalTokenCount ?? 0,
          } : undefined, 
          finishReason: finishReason, 
          // toolCalls: undefined, 
          // toolResults: undefined, 
        };
  
        return providerResponse;
    };
  
    async streamText(
      options: StreamTextOptions<string>,
    ): Promise<ProviderTextStreamResponse<GoogleGenerateContentStreamResult>> {
      throw new Error("streamText is not fully implemented yet.");
    }
  
  
    async generateObject<TSchema extends z.ZodType>(
      options: any, 
    ): Promise<any> { 
      throw new Error("generateObject is not implemented for GoogleGenAIProvider yet.");
    }
  
    async streamObject<TSchema extends z.ZodType>(
      options: any, 
    ): Promise<any> {
      throw new Error("streamObject is not implemented for GoogleGenAIProvider yet.");
    }
  }