import {
  type Content,
  type GenerateContentConfig,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type GenerateContentResponseUsageMetadata,
  GoogleGenAI,
} from "@google/genai";
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
} from "@voltagent/core";
import type { GoogleGenAIProviderOptions, GoogleGenerateContentStreamResult } from "./types";

export class GoogleGenAIProvider implements LLMProvider<string> {
  private ai: GoogleGenAI;

  constructor(options: GoogleGenAIProviderOptions) {
    const hasApiKey = !!options?.apiKey;
    const hasVertexAIConfig = options.vertexai && options.project && options.location;

    if (!hasApiKey && !hasVertexAIConfig) {
      throw new Error(
        "Google GenAI API key is required, or if using Vertex AI, both project and location must be specified.",
      );
    }

    this.ai = new GoogleGenAI({
      apiKey: options.apiKey,
      vertexai: options.vertexai,
      project: options.project,
      location: options.location,
    });

    this.generateText = this.generateText.bind(this);
    this.streamText = this.streamText.bind(this);
    this.toMessage = this.toMessage.bind(this);
    this._createStepFromChunk = this._createStepFromChunk.bind(this);
    this.getModelIdentifier = this.getModelIdentifier.bind(this);
    this._getUsageInfo = this._getUsageInfo.bind(this);
    this._processStreamChunk = this._processStreamChunk.bind(this);
    this._finalizeStream = this._finalizeStream.bind(this);
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
        console.warn(`System role conversion might require specific handling. Mapping to 'model'.`);
        return "model";
      case "tool":
        // Google GenAI uses FunctionResponse parts for tool results, mapped to the 'function' role (internally represented).
        // When sending tool messages *back* to the model, they should be part of the 'user' or 'model' content array.
        // Mapping to 'model' as a placeholder, but requires proper FunctionResponse formatting.
        console.warn(
          `Tool role conversion to Google GenAI format is complex. Mapping to 'model' as placeholder.`,
        );
        return "model";
      default:
        console.warn(`Unsupported role conversion for: ${role}. Defaulting to 'user'.`);
        return "user";
    }
  }

  toMessage = (message: BaseMessage): Content => {
    const role = this.toGoogleRole(message.role);
    if (typeof message.content === "string") {
      if (role !== "user" && role !== "model") {
        throw new Error(
          `Invalid role '${role}' passed to toMessage for string content. Expected 'user' or 'model'.`,
        );
      }
      return { role, parts: [{ text: message.content }] };
    }
    const parts = message.content
      .map((part) => {
        if (part.type === "text") {
          return { text: part.text };
        }
        // TODO: Add mapping for image parts, etc.
        console.warn(`Unsupported part type: ${part.type}. Skipping.`);
        return null; // Filter out unsupported parts
      })
      .filter((part): part is { text: string } => part !== null);

    if (parts.length === 0) {
      // Handle cases where no supported parts are found
      console.warn(
        `No supported parts found for message with role ${role}. Creating empty text part.`,
      );
      return { role, parts: [{ text: "" }] };
    }

    if (role !== "user" && role !== "model") {
      throw new Error(
        `Invalid role '${role}' passed to toMessage for structured content. Expected 'user' or 'model'.`,
      );
    }
    return { role, parts };
  };

  private _createStepFromChunk(
    chunkResponse: GenerateContentResponse,
    role: MessageRole = "assistant",
    usage?: UsageInfo,
  ): StepWithContent | null {
    const text = chunkResponse.text; // Use the text getter which concatenates text parts
    // TODO: Potentially handle function calls or other parts from the chunk if needed for StepWithContent
    if (text !== undefined && text !== "") {
      // Check text is not undefined or empty
      return {
        id: chunkResponse.responseId || "", // Use responseId if available
        type: "text",
        content: text,
        role: role,
        usage: usage,
      };
    }
    return null;
  }

  private _getUsageInfo(
    usageInfo: GenerateContentResponseUsageMetadata | undefined,
  ): UsageInfo | undefined {
    if (!usageInfo) return undefined;

    // Ensure counts are numbers, default to 0 if undefined/null
    const promptTokens = usageInfo.promptTokenCount ?? 0;
    const completionTokens = usageInfo.candidatesTokenCount ?? 0;
    const totalTokens = usageInfo.totalTokenCount ?? 0;

    // Only return usage if there are actual tokens counted
    if (promptTokens > 0 || completionTokens > 0 || totalTokens > 0) {
      return { promptTokens, completionTokens, totalTokens };
    }

    return undefined;
  }

  generateText = async (
    options: GenerateTextOptions<string>,
  ): Promise<ProviderTextResponse<GenerateContentResponse>> => {
    const model = options.model;
    const contents = options.messages.map(this.toMessage);
    const providerOptions = options.provider || {};

    const config: GenerateContentConfig = {
      temperature: providerOptions.temperature,
      topP: providerOptions.topP,
      stopSequences: providerOptions.stopSequences,
      seed: providerOptions.seed,
      presencePenalty: providerOptions.presencePenalty,
      frequencyPenalty: providerOptions.frequencyPenalty,
      ...(providerOptions.extraOptions && providerOptions.extraOptions),
    };

    const generationParams: GenerateContentParameters = {
      model: model,
      contents: contents,
      ...(Object.keys(config).length > 0 && { config: config }),
    };

    const result = await this.ai.models.generateContent(generationParams);

    const responseText = result.text;
    const usageInfo = result?.usageMetadata;
    const finishReason = result.candidates?.[0]?.finishReason?.toString();
    const finalUsage = this._getUsageInfo(usageInfo);

    if (options.onStepFinish) {
      const step = this._createStepFromChunk(result, "assistant", finalUsage);
      if (step) await options.onStepFinish(step);
    }

    const providerResponse: ProviderTextResponse<GenerateContentResponse> = {
      provider: result,
      text: responseText ?? "", // Ensure text is a string
      usage: finalUsage,
      finishReason: finishReason,
      // toolCalls: undefined,
      // toolResults: undefined,
    };

    return providerResponse;
  };

  private async _processStreamChunk(
    chunkResponse: GenerateContentResponse,
    controller: ReadableStreamDefaultController<string>,
    state: { accumulatedText: string; finalUsage?: UsageInfo },
    options: StreamTextOptions<string>,
  ): Promise<void> {
    const textChunk = chunkResponse.text;
    const chunkUsage = this._getUsageInfo(chunkResponse.usageMetadata);

    // Capture usage if present
    if (chunkUsage) {
      state.finalUsage = chunkUsage;
    }

    // Enqueue text and call onChunk if applicable
    if (textChunk !== undefined && textChunk !== "") {
      controller.enqueue(textChunk);
      state.accumulatedText += textChunk;
      if (options.onChunk) {
        const step = this._createStepFromChunk(chunkResponse, "assistant", chunkUsage);
        if (step) await options.onChunk(step);
      }
    }

    // Handle prompt feedback
    if (chunkResponse.promptFeedback && options.onError) {
      console.warn("Prompt feedback received:", chunkResponse.promptFeedback);
      // Consider invoking onError or a specific feedback handler
    }

    // TODO: Handle function calls here if needed
  }

  // Helper to finalize the stream after processing all chunks
  private async _finalizeStream(
    state: { accumulatedText: string; finalUsage?: UsageInfo },
    options: StreamTextOptions<string>,
    controller: ReadableStreamDefaultController<string>,
  ): Promise<void> {
    if (options.onStepFinish) {
      const finalStep: StepWithContent = {
        id: "",
        type: "text",
        content: state.accumulatedText,
        role: "assistant",
        usage: state.finalUsage,
      };
      await options.onStepFinish(finalStep);
    }
    if (options.onFinish) {
      await options.onFinish({ text: state.accumulatedText });
    }
    controller.close();
  }

  async streamText(
    options: StreamTextOptions<string>,
  ): Promise<ProviderTextStreamResponse<GoogleGenerateContentStreamResult>> {
    const model = options.model;
    const contents = options.messages.map(this.toMessage);
    const providerOptions = options.provider || {};

    // Prepare config similar to generateText
    const config: GenerateContentConfig = Object.entries({
      temperature: providerOptions.temperature,
      topP: providerOptions.topP,
      stopSequences: providerOptions.stopSequences,
      seed: providerOptions.seed,
      presencePenalty: providerOptions.presencePenalty,
      frequencyPenalty: providerOptions.frequencyPenalty,
      ...(providerOptions.extraOptions && providerOptions.extraOptions),
    }).reduce((acc, [key, value]) => {
      if (value !== undefined) {
        (acc as any)[key] = value;
      }
      return acc;
    }, {} as GenerateContentConfig);

    const generationParams: GenerateContentParameters = {
      model: model,
      contents: contents,
      ...(Object.keys(config).length > 0 && { config: config }),
      // TODO: Add tools parameter mapping if options.tools is provided
    };

    // Get the async generator from the Google SDK
    const streamGenerator = await this.ai.models.generateContentStream(generationParams);

    const state = this; // Bind the class instance to the state
    const streamState = {
      accumulatedText: "",
      finalUsage: undefined as UsageInfo | undefined,
    };

    const readableStream = new ReadableStream<string>({
      async start(controller) {
        try {
          for await (const chunkResponse of streamGenerator) {
            await state._processStreamChunk(chunkResponse, controller, streamState, options);
          }
          await state._finalizeStream(streamState, options, controller);
        } catch (error) {
          console.error("Error during Google GenAI stream processing:", error);
          if (options.onError) {
            await options.onError(error);
          }
          controller.error(error);
        }
      },
      cancel(reason) {
        console.log("Google GenAI Stream cancelled:", reason);
        // Note: The @google/genai SDK's async generator doesn't have explicit cancellation.
      },
    });

    return {
      provider: streamGenerator, // Return the raw async generator
      textStream: readableStream,
    };
  }

  async generateObject(_options: any): Promise<any> {
    throw new Error("generateObject is not implemented for GoogleGenAIProvider yet.");
  }

  async streamObject(_options: any): Promise<any> {
    throw new Error("streamObject is not implemented for GoogleGenAIProvider yet.");
  }
}
