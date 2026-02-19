import type { LanguageModelV2 } from "@ai-sdk/provider"
import type { Provider } from "ai"
import { HarmonyLanguageModel, type HarmonyModelSettings } from "./harmony-language-model"

export interface HarmonyProviderSettings {
  /**
   * API key for authenticating requests.
   */
  apiKey?: string

  /**
   * Base URL for the vLLM completions endpoint.
   */
  baseURL?: string

  /**
   * Name of the provider (used for provider metadata keys).
   */
  name?: string

  /**
   * Custom headers to include in the requests.
   */
  headers?: Record<string, string>

  /**
   * Custom fetch implementation.
   */
  fetch?: typeof globalThis.fetch

  /**
   * Reasoning effort level for the model.
   */
  reasoningLevel?: "low" | "medium" | "high"
}

export function createHarmony(options: HarmonyProviderSettings = {}): Provider {
  const baseURL = options.baseURL ?? "http://localhost:8000"

  const createLanguageModel = (modelId: string): LanguageModelV2 => {
    const settings: HarmonyModelSettings = {
      baseURL,
      apiKey: options.apiKey,
      headers: options.headers,
      fetch: options.fetch,
      reasoningLevel: options.reasoningLevel,
    }
    const model = new HarmonyLanguageModel(modelId, settings)
    // Override the provider name if one was specified
    if (options.name) {
      Object.defineProperty(model, "provider", { value: options.name })
    }
    return model
  }

  return {
    languageModel: createLanguageModel,
    textEmbeddingModel: () => {
      throw new Error("Harmony provider does not support text embeddings")
    },
    imageModel: () => {
      throw new Error("Harmony provider does not support image generation")
    },
  }
}
