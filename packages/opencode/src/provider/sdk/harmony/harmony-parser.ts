import type { LanguageModelV2FinishReason, LanguageModelV2StreamPart } from "@ai-sdk/provider"

/**
 * Harmony wire format stream parser for GPT-OSS models.
 *
 * Harmony uses control tokens to structure multi-channel output:
 *   - `<|start|>` / `<|end|>` — message boundaries
 *   - `<|message|>` — header-to-content transition
 *   - `<|channel|>` — channel switch (analysis, commentary, final)
 *   - `<|call|>` — tool call terminator (stop token)
 *   - `<|return|>` — model finished (stop token)
 *   - `<|constrain|>` — constrained generation marker (e.g., json)
 *
 * Channel routing:
 *   - analysis  → reasoning-delta events
 *   - final     → text-delta events
 *   - commentary → may contain tool calls
 *
 * @see https://developers.openai.com/cookbook/articles/openai-harmony
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HarmonyParseState {
  /** The currently active channel, or null if outside a channel. */
  currentChannel: "analysis" | "commentary" | "final" | null
  /** Accumulated text content within the current channel. */
  buffer: string
  /** Pending tool call being assembled from commentary channel content. */
  pendingToolCall: { name: string; arguments: string } | null
  /** Lookahead buffer that holds a possible partial control token at a chunk boundary. */
  tokenBuffer: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * All recognised control tokens in the Harmony wire format.
 * Order matters: longer prefixes should be checked first when disambiguating.
 */
const CONTROL_TOKENS = [
  "<|start|>",
  "<|end|>",
  "<|message|>",
  "<|channel|>",
  "<|call|>",
  "<|return|>",
  "<|constrain|>",
] as const

type ControlToken = (typeof CONTROL_TOKENS)[number]

/**
 * The longest control token length, used to bound the lookahead buffer.
 */
const MAX_TOKEN_LENGTH = Math.max(...CONTROL_TOKENS.map((t) => t.length))

/**
 * The common prefix shared by all control tokens.
 */
const TOKEN_PREFIX = "<|"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createInitialState(): HarmonyParseState {
  return {
    currentChannel: null,
    buffer: "",
    pendingToolCall: null,
    tokenBuffer: "",
  }
}

/**
 * Check whether `candidate` is a prefix of any known control token.
 * This is used for the lookahead buffer: when a chunk ends with a partial
 * match we hold the bytes until the next chunk confirms or denies.
 */
function isPrefixOfControlToken(candidate: string): boolean {
  if (candidate.length === 0) return false
  for (const token of CONTROL_TOKENS) {
    if (token.startsWith(candidate) && candidate.length < token.length) {
      return true
    }
  }
  return false
}

/**
 * Extract the tool name from a Harmony `to=functions.{name}` directive.
 * Returns the function name, or null if the format doesn't match.
 */
function extractToolName(headerText: string, knownTools?: string[]): string | null {
  const match = headerText.match(/to=functions\.(\S+)/)
  if (!match) return null
  const raw = match[1]
  if (!knownTools || knownTools.length === 0 || knownTools.includes(raw)) return raw
  // Fuzzy: if the model truncated the name, try prefix match
  const candidates = knownTools.filter((t) => t.startsWith(raw))
  if (candidates.length === 1) return candidates[0]
  return raw
}

/**
 * Try to parse a string as JSON. Returns true if valid.
 */
function isValidJson(text: string): boolean {
  try {
    JSON.parse(text)
    return true
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Creates a stateful `TransformStream` that processes Harmony wire-format
 * text chunks and emits AI SDK `LanguageModelV2StreamPart` events.
 *
 * The input side receives raw string chunks (the decoded text stream from
 * the model). The output side emits structured stream parts that the AI SDK
 * understands.
 *
 * Design principles:
 *   - **Lenient**: Never throw. Degrade gracefully on malformed input.
 *   - **Flush-safe**: Stream termination is treated as an implicit close.
 *   - **Chunk-boundary safe**: Partial control tokens are held in a
 *     lookahead buffer until the next chunk confirms or denies a match.
 */
export function createHarmonyStreamParser(knownTools?: string[]): TransformStream<string, LanguageModelV2StreamPart> {
  const state = createInitialState()

  // Track whether we have emitted start events so we can pair them with
  // end events in flush().
  let reasoningStarted = false
  let textStarted = false
  let toolCallCount = 0
  let hasToolCall = false

  // The finish reason to emit at stream end. Defaults to "stop" for a
  // normal completion; may be overridden by error conditions.
  let finishReason: LanguageModelV2FinishReason = "stop"

  // Track header text between <|start|> and <|message|> to extract
  // tool call metadata (to=functions.X, <|constrain|>json, etc.)
  let inHeader = false
  let headerText = ""

  /**
   * Flush the content buffer for the current channel, emitting the
   * appropriate delta events.
   */
  function flushBuffer(controller: TransformStreamDefaultController<LanguageModelV2StreamPart>) {
    if (state.buffer.length === 0) return

    const text = state.buffer
    state.buffer = ""

    switch (state.currentChannel) {
      case "analysis": {
        if (!reasoningStarted) {
          controller.enqueue({
            type: "reasoning-start",
            id: "harmony-reasoning-0",
          })
          reasoningStarted = true
        }
        controller.enqueue({
          type: "reasoning-delta",
          id: "harmony-reasoning-0",
          delta: text,
        })
        break
      }

      case "final": {
        if (!textStarted) {
          // If reasoning was open, close it first before starting text
          if (reasoningStarted) {
            controller.enqueue({
              type: "reasoning-end",
              id: "harmony-reasoning-0",
            })
            reasoningStarted = false
          }
          controller.enqueue({
            type: "text-start",
            id: "harmony-txt-0",
          })
          textStarted = true
        }
        controller.enqueue({
          type: "text-delta",
          id: "harmony-txt-0",
          delta: text,
        })
        break
      }

      case "commentary": {
        // Commentary content may be tool call arguments or preamble text.
        // If we have a pending tool call, accumulate into its arguments.
        if (state.pendingToolCall) {
          state.pendingToolCall.arguments += text
        }
        // Otherwise commentary text with no tool context is discarded
        // (it's multi-step preamble not shown to the user).
        break
      }

      default:
        // Content outside any channel: pass through as text for forward
        // compatibility with unknown formats.
        if (text.trim().length > 0) {
          if (!textStarted) {
            if (reasoningStarted) {
              controller.enqueue({
                type: "reasoning-end",
                id: "harmony-reasoning-0",
              })
              reasoningStarted = false
            }
            controller.enqueue({
              type: "text-start",
              id: "harmony-txt-0",
            })
            textStarted = true
          }
          controller.enqueue({
            type: "text-delta",
            id: "harmony-txt-0",
            delta: text,
          })
        }
        break
    }
  }

  /**
   * Emit the completed tool call events for the pending tool call,
   * then clear the pending state.
   */
  function emitToolCall(controller: TransformStreamDefaultController<LanguageModelV2StreamPart>) {
    if (!state.pendingToolCall) return

    const toolCallId = `harmony-tc-${toolCallCount++}`
    const { name, arguments: args } = state.pendingToolCall

    // Close reasoning if it was active before emitting tool events
    if (reasoningStarted) {
      controller.enqueue({
        type: "reasoning-end",
        id: "harmony-reasoning-0",
      })
      reasoningStarted = false
    }

    // Validate JSON arguments. If malformed, emit raw string and warn.
    if (args.length > 0 && !isValidJson(args)) {
      console.warn(
        `[harmony-parser] Malformed JSON in tool call arguments for "${name}": ${args.slice(0, 100)}${args.length > 100 ? "..." : ""}`,
      )
    }

    controller.enqueue({
      type: "tool-input-start",
      id: toolCallId,
      toolName: name,
    })

    if (args.length > 0) {
      controller.enqueue({
        type: "tool-input-delta",
        id: toolCallId,
        delta: args,
      })
    }

    controller.enqueue({
      type: "tool-input-end",
      id: toolCallId,
    })

    controller.enqueue({
      type: "tool-call",
      toolCallId,
      toolName: name,
      input: args,
    })

    hasToolCall = true
    finishReason = "tool-calls"
    state.pendingToolCall = null
  }

  /**
   * Handle implicit close of the current channel. This flushes any
   * buffered content and, for commentary channels with pending tool calls,
   * emits the tool call.
   */
  function implicitCloseChannel(controller: TransformStreamDefaultController<LanguageModelV2StreamPart>) {
    flushBuffer(controller)

    if (state.currentChannel === "commentary" && state.pendingToolCall) {
      emitToolCall(controller)
    }

    state.currentChannel = null
  }

  /**
   * Process a confirmed control token and update parser state accordingly.
   */
  function handleControlToken(
    token: ControlToken,
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    switch (token) {
      case "<|start|>": {
        // Beginning of a new message block. Implicitly close any open channel.
        if (state.currentChannel !== null) {
          implicitCloseChannel(controller)
        }
        // Enter header parsing mode to capture channel/to= directives
        inHeader = true
        headerText = ""
        break
      }

      case "<|message|>": {
        // Transition from header to content. Parse the accumulated header.
        inHeader = false

        // Parse the header for channel and tool call information.
        // Header format: "assistant<|channel|>commentary to=functions.name <|constrain|>json"
        // But by the time we see <|message|>, the channel token and its
        // content have already been processed as separate tokens/text.
        // So headerText should contain residual text like role names
        // that were between <|start|> and the first control token.
        break
      }

      case "<|channel|>": {
        // A channel switch. If a channel is already open, implicitly close it.
        if (state.currentChannel !== null) {
          implicitCloseChannel(controller)
        }
        // The channel name follows immediately as text (analysis/final/commentary).
        // It will be captured as text content; we set a flag to intercept it.
        // We handle this by putting ourselves in a state where the next text
        // chunk is interpreted as the channel name + optional header directives.
        state.currentChannel = null // will be set when we see the channel name
        // Mark that we need to parse channel name from the next text chunk
        inHeader = true
        headerText = ""
        break
      }

      case "<|constrain|>": {
        // Constrained generation marker (e.g., followed by "json").
        // The constraint type follows as text. We don't need to act on it
        // explicitly; the pending tool call structure already handles JSON args.
        break
      }

      case "<|call|>": {
        // Tool call stop token. This terminates the current message and
        // signals that the model wants to invoke a tool.
        flushBuffer(controller)
        if (state.pendingToolCall) {
          emitToolCall(controller)
        }
        state.currentChannel = null
        inHeader = false
        break
      }

      case "<|return|>": {
        // Model finished stop token. Flush and close.
        flushBuffer(controller)
        if (state.currentChannel === "commentary" && state.pendingToolCall) {
          emitToolCall(controller)
        }
        state.currentChannel = null
        inHeader = false
        finishReason = "stop"
        break
      }

      case "<|end|>": {
        // End of message block. Flush buffer and close channel.
        flushBuffer(controller)
        if (state.currentChannel === "commentary" && state.pendingToolCall) {
          emitToolCall(controller)
        }
        state.currentChannel = null
        inHeader = false
        break
      }
    }
  }

  /**
   * Handle text content that is not a control token. This routes the text
   * to the appropriate channel buffer or interprets it as a channel
   * name / header directive.
   */
  function handleText(text: string, controller: TransformStreamDefaultController<LanguageModelV2StreamPart>) {
    if (text.length === 0) return

    if (inHeader) {
      // We are inside a header (between <|start|>/<|channel|> and <|message|>).
      // Parse for channel names and tool call directives.
      headerText += text

      // Try to extract the channel name if not yet set.
      if (state.currentChannel === null) {
        const trimmed = headerText.trimStart()
        // Check for known channel names at the start of the header text
        for (const ch of ["analysis", "commentary", "final"] as const) {
          if (trimmed.startsWith(ch)) {
            state.currentChannel = ch
            // Check for tool call directives in the rest of the header
            const rest = trimmed.slice(ch.length)
            if (state.currentChannel === "commentary") {
              const toolName = extractToolName(rest, knownTools)
              if (toolName) {
                state.pendingToolCall = { name: toolName, arguments: "" }
              }
            }
            break
          }
        }
      } else if (state.currentChannel === "commentary") {
        // We may see "to=functions.X" after the channel name was already set
        if (!state.pendingToolCall) {
          const toolName = extractToolName(headerText, knownTools)
          if (toolName) {
            state.pendingToolCall = { name: toolName, arguments: "" }
          }
        }
      }

      return
    }

    // We are inside message content. Append to buffer and flush
    // immediately as deltas for low latency.
    state.buffer += text
    flushBuffer(controller)
  }

  /**
   * Core tokenizer: scans the input text for control tokens, handling
   * partial tokens at chunk boundaries via the lookahead buffer.
   *
   * Returns any remaining text that might be a partial control token
   * (held for the next chunk).
   */
  function processChunk(input: string, controller: TransformStreamDefaultController<LanguageModelV2StreamPart>) {
    // Prepend any leftover from the lookahead buffer
    let text = state.tokenBuffer + input
    state.tokenBuffer = ""

    let pos = 0

    while (pos < text.length) {
      // Search for the start of a potential control token
      const tokenStart = text.indexOf(TOKEN_PREFIX, pos)

      if (tokenStart === -1) {
        // No more potential tokens in the remaining text.
        // But the very end might be a partial "<" that starts TOKEN_PREFIX.
        if (text[text.length - 1] === "<" && pos <= text.length - 1) {
          handleText(text.slice(pos, text.length - 1), controller)
          state.tokenBuffer = "<"
        } else {
          handleText(text.slice(pos), controller)
        }
        return
      }

      // Emit any text before the potential token
      if (tokenStart > pos) {
        handleText(text.slice(pos, tokenStart), controller)
      }

      // Try to match a complete control token starting at tokenStart
      let matched = false
      for (const token of CONTROL_TOKENS) {
        if (text.startsWith(token, tokenStart)) {
          handleControlToken(token, controller)
          pos = tokenStart + token.length
          matched = true
          break
        }
      }

      if (matched) continue

      // No complete token matched. Check if the remaining text from
      // tokenStart could be a prefix of a control token (chunk boundary).
      const remaining = text.slice(tokenStart)
      if (remaining.length < MAX_TOKEN_LENGTH && isPrefixOfControlToken(remaining)) {
        // Hold in lookahead buffer for the next chunk
        state.tokenBuffer = remaining
        return
      }

      // Not a prefix of any known token. Treat "<|" as literal text.
      // Emit just the "<|" and continue scanning after it.
      handleText(TOKEN_PREFIX, controller)
      pos = tokenStart + TOKEN_PREFIX.length
    }
  }

  return new TransformStream<string, LanguageModelV2StreamPart>({
    start(controller) {
      controller.enqueue({ type: "stream-start", warnings: [] })
    },

    transform(chunk, controller) {
      try {
        processChunk(chunk, controller)
      } catch (error) {
        // Lenient: never crash. Emit error event and continue.
        controller.enqueue({
          type: "error",
          error: error instanceof Error ? error.message : String(error),
        })
      }
    },

    flush(controller) {
      try {
        // Flush any remaining lookahead buffer as literal text
        if (state.tokenBuffer.length > 0) {
          const remaining = state.tokenBuffer
          state.tokenBuffer = ""

          // If we were in the middle of a partial token like "<|cal",
          // check if it looked like a truncated tool call
          if (remaining.startsWith(TOKEN_PREFIX)) {
            // Partial control token at stream end. Check if it was a
            // truncated <|call|> with a pending tool call.
            if (state.pendingToolCall && remaining.startsWith("<|call")) {
              // Truncated <|call|> but we have a tool name and possibly
              // incomplete args. Discard the incomplete call.
              state.pendingToolCall = null
              finishReason = "length"
            } else if (state.pendingToolCall) {
              // Some other partial token while a tool call was pending.
              // Treat it as text content that got cut off.
              handleText(remaining, controller)
              finishReason = "length"
            } else {
              // Partial token with no tool context: treat as text
              handleText(remaining, controller)
            }
          } else {
            handleText(remaining, controller)
          }
        }

        // Implicitly close any open channel
        if (state.currentChannel !== null) {
          flushBuffer(controller)

          // If commentary had a pending tool call that was never completed
          // with <|call|>, emit it if we have valid name + args (the stop
          // token may have been consumed by the API without reaching the parser).
          if (state.currentChannel === "commentary" && state.pendingToolCall) {
            const { name, arguments: args } = state.pendingToolCall
            if (name && args.length > 0) {
              // Emit the tool call — stream likely ended because <|call|>
              // was a stop token consumed by the API.
              emitToolCall(controller)
            } else {
              // No args or no name: discard incomplete call
              state.pendingToolCall = null
              finishReason = "length"
            }
          }

          // If no <|end|> or <|return|> was seen, this is a truncated stream
          if (finishReason === "stop" && !hasToolCall) {
            finishReason = "length"
          }

          state.currentChannel = null
        }

        // Close any open reasoning part
        if (reasoningStarted) {
          controller.enqueue({
            type: "reasoning-end",
            id: "harmony-reasoning-0",
          })
          reasoningStarted = false
        }

        // Close any open text part
        if (textStarted) {
          controller.enqueue({
            type: "text-end",
            id: "harmony-txt-0",
          })
          textStarted = false
        }

        // Emit finish event
        controller.enqueue({
          type: "finish",
          finishReason,
          usage: {
            inputTokens: undefined,
            outputTokens: undefined,
            totalTokens: undefined,
          },
        })
      } catch (error) {
        // Even flush must not throw. Emit a minimal finish.
        controller.enqueue({
          type: "error",
          error: error instanceof Error ? error.message : String(error),
        })
        controller.enqueue({
          type: "finish",
          finishReason: "error",
          usage: {
            inputTokens: undefined,
            outputTokens: undefined,
            totalTokens: undefined,
          },
        })
      }
    },
  })
}
