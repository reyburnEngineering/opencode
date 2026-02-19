import type {
  LanguageModelV2,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider"
import { renderTools, renderMessages, renderReasoningLevel } from "./harmony-renderer"
import { createHarmonyStreamParser, type HarmonyUsage } from "./harmony-parser"

export interface HarmonyModelSettings {
  baseURL: string
  apiKey?: string
  headers?: Record<string, string>
  fetch?: typeof globalThis.fetch
  reasoningLevel?: "low" | "medium" | "high"
}

export class HarmonyLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2" as const
  readonly provider: string
  readonly modelId: string

  readonly defaultObjectGenerationMode = undefined
  readonly supportedUrls = {}
  readonly supportsStructuredOutputs = false

  private readonly settings: HarmonyModelSettings
  private readonly fetchFn: typeof globalThis.fetch

  constructor(modelId: string, settings: HarmonyModelSettings) {
    this.modelId = modelId
    this.provider = "harmony"
    this.settings = settings
    this.fetchFn = settings.fetch ?? globalThis.fetch
  }

  async doGenerate(
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    // Implement doGenerate by internally calling doStream and buffering the result.
    // Single parser, no second code path.
    const { stream, ...rest } = await this.doStream(options)

    const content: LanguageModelV2Content[] = []
    let finishReason: LanguageModelV2FinishReason = "unknown"
    let usage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      reasoningTokens: 0,
      cachedInputTokens: 0,
    }

    // Accumulate text and tool call pieces from stream parts
    const textParts: Map<string, { text: string }> = new Map()
    const reasoningParts: Map<string, { text: string }> = new Map()
    const toolInputParts: Map<string, { toolName: string; input: string }> = new Map()
    const completedToolCalls: Set<string> = new Set()

    const reader = stream.getReader()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        switch (value.type) {
          case "text-start": {
            textParts.set(value.id, { text: "" })
            break
          }
          case "text-delta": {
            const part = textParts.get(value.id)
            if (part) {
              part.text += value.delta
            }
            break
          }
          case "reasoning-start": {
            reasoningParts.set(value.id, { text: "" })
            break
          }
          case "reasoning-delta": {
            const part = reasoningParts.get(value.id)
            if (part) {
              part.text += value.delta
            }
            break
          }
          case "tool-call": {
            if (!completedToolCalls.has(value.toolCallId)) {
              content.push({
                type: "tool-call",
                toolCallId: value.toolCallId,
                toolName: value.toolName,
                input: value.input,
              })
              completedToolCalls.add(value.toolCallId)
            }
            break
          }
          case "tool-input-start": {
            toolInputParts.set(value.id, { toolName: value.toolName, input: "" })
            break
          }
          case "tool-input-delta": {
            const part = toolInputParts.get(value.id)
            if (part) {
              part.input += value.delta
            }
            break
          }
          case "finish": {
            finishReason = value.finishReason
            usage = value.usage ?? {}
            break
          }
        }
      }
    } finally {
      reader.releaseLock()
    }

    // Convert accumulated reasoning parts to content
    for (const [, part] of reasoningParts) {
      if (part.text.length > 0) {
        content.push({
          type: "reasoning",
          text: part.text,
        })
      }
    }

    // Convert accumulated text parts to content
    for (const [, part] of textParts) {
      if (part.text.length > 0) {
        content.push({
          type: "text",
          text: part.text,
        })
      }
    }

    // Convert any incomplete tool input parts to tool-call content (if not already emitted)
    for (const [id, part] of toolInputParts) {
      if (!completedToolCalls.has(id)) {
        content.push({
          type: "tool-call",
          toolCallId: id,
          toolName: part.toolName,
          input: part.input,
        })
      }
    }

    return {
      content,
      finishReason,
      usage,
      providerMetadata: {},
      request: { body: rest.request?.body },
      response: {
        headers: rest.response?.headers,
      },
      warnings: [],
    }
  }

  async doStream(
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    // 1. Render tools to Harmony TypeScript namespace format
    const toolsText = renderTools(options.tools)

    // 2. Render messages to Harmony control token format
    const messagesText = renderMessages(options.prompt)

    // 3. Build the full prompt
    //    Tools and reasoning level go inside a system message block so
    //    the model recognises them as part of the Harmony structure.
    const reasoningLevel =
      (options.providerOptions?.harmony as any)?.reasoningLevel ??
      (options as any).reasoningLevel ??
      this.settings.reasoningLevel
    let preamble = ""
    if (reasoningLevel) {
      preamble += renderReasoningLevel(reasoningLevel) + "\n"
    }
    if (toolsText) {
      preamble += toolsText
    }

    let prompt = ""
    if (preamble) {
      prompt += `<|start|>system<|message|>${preamble}<|end|>\n`
    }
    prompt += messagesText

    // 4. Build the request body for the vLLM completions API
    const body = {
      model: this.modelId,
      prompt,
      max_tokens: options.maxOutputTokens ?? 4096,
      stream: true,
      stream_options: { include_usage: true },
      stop: ["<|call|>"],
      skip_special_tokens: false,
      include_stop_str_in_output: true,
    }

    const requestBody = JSON.stringify(body)

    // 5. HTTP POST to {baseURL}/v1/completions
    const url = `${this.settings.baseURL}/v1/completions`
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...this.settings.headers,
      ...(options.headers as Record<string, string>),
    }
    if (this.settings.apiKey) {
      headers["Authorization"] = `Bearer ${this.settings.apiKey}`
    }

    const response = await this.fetchFn(url, {
      method: "POST",
      headers,
      body: requestBody,
      signal: options.abortSignal ?? undefined,
    })

    if (!response.ok) {
      const errorBody = await response.text()
      throw new Error(
        `Harmony API request failed with status ${response.status}: ${errorBody}`,
      )
    }

    if (!response.body) {
      throw new Error("Harmony API response has no body")
    }

    const responseHeaders = Object.fromEntries(response.headers.entries())

    // 6. Parse SSE stream, extract text deltas, and pipe through Harmony stream parser
    const toolNames = options.tools
      ?.filter((t) => t.type === "function")
      .map((t) => t.name)
    const usage: HarmonyUsage = {}
    const sseStream = this.createSSETextStream(response.body, usage)
    const harmonyParser = createHarmonyStreamParser(toolNames, usage)
    const parsedStream = sseStream.pipeThrough(harmonyParser)

    return {
      stream: parsedStream,
      request: { body: requestBody },
      response: { headers: responseHeaders },
    }
  }

  /**
   * Transforms a raw SSE byte stream into a stream of text deltas
   * extracted from vLLM completions SSE chunks.
   *
   * Expects SSE format: `data: {"choices": [{"text": "..."}]}\n\n`
   * Emits the text content from each SSE chunk.
   */
  private createSSETextStream(
    body: ReadableStream<Uint8Array>,
    usage?: HarmonyUsage,
  ): ReadableStream<string> {
    const decoder = new TextDecoder()
    let buffer = ""

    return new ReadableStream<string>({
      async start(controller) {
        const reader = body.getReader()
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              // Process any remaining buffer
              if (buffer.trim().length > 0) {
                processLines(buffer, controller, usage)
              }
              controller.close()
              break
            }

            buffer += decoder.decode(value, { stream: true })

            // Process complete lines
            const lines = buffer.split("\n")
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() ?? ""

            for (const line of lines) {
              processLine(line.trim(), controller, usage)
            }
          }
        } catch (error) {
          controller.error(error)
        } finally {
          reader.releaseLock()
        }
      },
    })
  }
}

function processLines(text: string, controller: ReadableStreamDefaultController<string>, usage?: HarmonyUsage) {
  const lines = text.split("\n")
  for (const line of lines) {
    processLine(line.trim(), controller, usage)
  }
}

function processLine(line: string, controller: ReadableStreamDefaultController<string>, usage?: HarmonyUsage) {
  if (!line.startsWith("data: ")) return

  const data = line.slice(6) // Remove "data: " prefix

  // Handle end-of-stream signal
  if (data === "[DONE]") return

  try {
    const parsed = JSON.parse(data) as {
      choices?: Array<{ text?: string; finish_reason?: string | null }>
      usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
    }

    const text = parsed.choices?.[0]?.text
    if (text) {
      controller.enqueue(text)
    }

    // Extract usage from the SSE chunk (vLLM sends it on the final chunk)
    if (usage && parsed.usage) {
      if (parsed.usage.prompt_tokens != null) usage.inputTokens = parsed.usage.prompt_tokens
      if (parsed.usage.completion_tokens != null) usage.outputTokens = parsed.usage.completion_tokens
      if (parsed.usage.total_tokens != null) usage.totalTokens = parsed.usage.total_tokens
    }
  } catch {
    // Skip malformed JSON lines
  }
}
