import { describe, expect, test } from "bun:test"
import type { LanguageModelV2StreamPart } from "@ai-sdk/provider"
import { createHarmonyStreamParser } from "../../../src/provider/sdk/harmony/harmony-parser"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Feed chunks through the parser and collect all output events. */
async function parse(chunks: string[], knownTools?: string[]): Promise<LanguageModelV2StreamPart[]> {
  const parser = createHarmonyStreamParser(knownTools)
  const writer = parser.writable.getWriter()
  const reader = parser.readable.getReader()

  const writePromise = (async () => {
    for (const chunk of chunks) {
      await writer.write(chunk)
    }
    await writer.close()
  })()

  const events: LanguageModelV2StreamPart[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    events.push(value)
  }

  await writePromise
  return events
}

/** Feed a single string as one chunk. */
async function parseOne(text: string): Promise<LanguageModelV2StreamPart[]> {
  return parse([text])
}

/** Get events of a specific type. */
function eventsOfType<T extends LanguageModelV2StreamPart["type"]>(
  events: LanguageModelV2StreamPart[],
  type: T,
): Extract<LanguageModelV2StreamPart, { type: T }>[] {
  return events.filter((e) => e.type === type) as any
}

/** Concatenate all text-delta values. */
function collectText(events: LanguageModelV2StreamPart[]): string {
  return eventsOfType(events, "text-delta")
    .map((e) => e.delta)
    .join("")
}

/** Concatenate all reasoning-delta values. */
function collectReasoning(events: LanguageModelV2StreamPart[]): string {
  return eventsOfType(events, "reasoning-delta")
    .map((e) => e.delta)
    .join("")
}

/** Get the finish reason from the finish event. */
function finishReason(events: LanguageModelV2StreamPart[]): string | undefined {
  const finish = eventsOfType(events, "finish")
  return finish[0]?.finishReason
}

// ---------------------------------------------------------------------------
// Basic text output (final channel)
// ---------------------------------------------------------------------------

describe("text output (final channel)", () => {
  test("simple text response", async () => {
    const events = await parseOne(
      "<|channel|>final<|message|>Hello, world!<|end|>",
    )

    expect(collectText(events)).toBe("Hello, world!")
    expect(eventsOfType(events, "text-start")).toHaveLength(1)
    expect(eventsOfType(events, "text-end")).toHaveLength(1)
    expect(finishReason(events)).toBe("stop")
  })

  test("text across multiple chunks", async () => {
    const events = await parse([
      "<|channel|>final<|message|>Hello, ",
      "world!",
      "<|end|>",
    ])

    expect(collectText(events)).toBe("Hello, world!")
  })

  test("text with full message structure", async () => {
    const events = await parseOne(
      "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>",
    )

    expect(collectText(events)).toBe("The answer is 42.")
    expect(finishReason(events)).toBe("stop")
  })
})

// ---------------------------------------------------------------------------
// Reasoning output (analysis channel)
// ---------------------------------------------------------------------------

describe("reasoning output (analysis channel)", () => {
  test("analysis channel emits reasoning events", async () => {
    const events = await parseOne(
      "<|channel|>analysis<|message|>Let me think about this...<|end|>",
    )

    expect(collectReasoning(events)).toBe("Let me think about this...")
    expect(eventsOfType(events, "reasoning-start")).toHaveLength(1)
    expect(eventsOfType(events, "reasoning-end")).toHaveLength(1)
  })

  test("reasoning followed by text", async () => {
    const events = await parseOne(
      "<|channel|>analysis<|message|>Thinking...<|end|>" +
        "<|channel|>final<|message|>Here is my answer.<|end|>",
    )

    expect(collectReasoning(events)).toBe("Thinking...")
    expect(collectText(events)).toBe("Here is my answer.")

    // reasoning-end should come before text-start
    const reasoningEnd = events.findIndex((e) => e.type === "reasoning-end")
    const textStart = events.findIndex((e) => e.type === "text-start")
    expect(reasoningEnd).toBeLessThan(textStart)
  })
})

// ---------------------------------------------------------------------------
// Tool calls (commentary channel)
// ---------------------------------------------------------------------------

describe("tool calls (commentary channel)", () => {
  test("single tool call with <|call|> token", async () => {
    const events = await parseOne(
      '<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"filePath":"hello.txt","content":"Hello"}<|call|>',
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("write")

    const input = JSON.parse(toolCalls[0].input as string)
    expect(input.filePath).toBe("hello.txt")
    expect(input.content).toBe("Hello")
    expect(finishReason(events)).toBe("tool-calls")
  })

  test("tool call with full message structure", async () => {
    const events = await parseOne(
      '<|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json<|message|>{"path":"/tmp/test.txt"}<|call|>',
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("read")
  })

  test("tool call emits input streaming events", async () => {
    const events = await parseOne(
      '<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"a":"b"}<|call|>',
    )

    expect(eventsOfType(events, "tool-input-start")).toHaveLength(1)
    expect(eventsOfType(events, "tool-input-delta")).toHaveLength(1)
    expect(eventsOfType(events, "tool-input-end")).toHaveLength(1)

    const inputStart = eventsOfType(events, "tool-input-start")[0]
    expect(inputStart.toolName).toBe("write")
  })

  test("tool call with chunked JSON input", async () => {
    const events = await parse([
      "<|channel|>commentary to=functions.write <|constrain|>json<|message|>",
      '{"filePath":',
      '"test.txt",',
      '"content":',
      '"hello"}',
      "<|call|>",
    ])

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)

    const input = JSON.parse(toolCalls[0].input as string)
    expect(input.filePath).toBe("test.txt")
    expect(input.content).toBe("hello")
  })

  test("tool call flushed on stream end (stop token consumed by API)", async () => {
    // Simulates vLLM consuming <|call|> as a stop token so the parser
    // never sees it — the stream just ends with pending tool call data.
    const events = await parseOne(
      '<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"path":"a.txt"}',
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("write")
    expect(finishReason(events)).toBe("tool-calls")
  })
})

// ---------------------------------------------------------------------------
// Multi-channel: reasoning + tool call
// ---------------------------------------------------------------------------

describe("multi-channel output", () => {
  test("analysis then tool call", async () => {
    const events = await parseOne(
      "<|channel|>analysis<|message|>I need to create a file.<|end|>" +
        '<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"filePath":"hello.txt","content":"Hello World"}<|call|>',
    )

    expect(collectReasoning(events)).toBe("I need to create a file.")

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("write")
    expect(finishReason(events)).toBe("tool-calls")
  })

  test("analysis then text (no tool call)", async () => {
    const events = await parseOne(
      "<|channel|>analysis<|message|>Considering the question...<|end|>" +
        "<|channel|>final<|message|>The answer is 42.<|end|>",
    )

    expect(collectReasoning(events)).toBe("Considering the question...")
    expect(collectText(events)).toBe("The answer is 42.")
    expect(finishReason(events)).toBe("stop")
  })

  test("realistic full model output", async () => {
    // Simulates what gpt-oss actually produces
    const events = await parse([
      "<|channel|>",
      "analysis",
      "<|message|>",
      "We need to create a file.",
      "<|end|>",
      "<|start|>",
      "assistant",
      "<|channel|>",
      "commentary to=functions.write ",
      "<|constrain|>",
      "json",
      "<|message|>",
      '{"filePath":"hello.txt","content":"Hello World"}',
      "<|call|>",
    ])

    expect(collectReasoning(events)).toBe("We need to create a file.")

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("write")

    const input = JSON.parse(toolCalls[0].input as string)
    expect(input.filePath).toBe("hello.txt")
    expect(input.content).toBe("Hello World")
    expect(finishReason(events)).toBe("tool-calls")
  })
})

// ---------------------------------------------------------------------------
// Chunk boundary handling (lookahead buffer)
// ---------------------------------------------------------------------------

describe("chunk boundary handling", () => {
  test("control token split across chunks", async () => {
    const events = await parse([
      "<|channel|>final<|message|>Hello<|en",
      "d|>",
    ])

    expect(collectText(events)).toBe("Hello")
    expect(finishReason(events)).toBe("stop")
  })

  test("token prefix at end of chunk", async () => {
    const events = await parse([
      "<|channel|>final<|message|>Hello<",
      "|end|>",
    ])

    expect(collectText(events)).toBe("Hello")
  })

  test("single character '<' at chunk boundary", async () => {
    const events = await parse([
      "<|channel|>final<|message|>Hello<",
      "|end|>",
    ])

    expect(collectText(events)).toBe("Hello")
  })

  test("literal < in text content (not a control token)", async () => {
    const events = await parseOne(
      "<|channel|>final<|message|>x < y and a > b<|end|>",
    )

    expect(collectText(events)).toBe("x < y and a > b")
  })

  test("literal <| that doesn't form a control token", async () => {
    const events = await parseOne(
      "<|channel|>final<|message|>test <|unknown|> text<|end|>",
    )

    // Should pass through the unknown token as text
    const text = collectText(events)
    expect(text).toContain("test")
    expect(text).toContain("text")
  })
})

// ---------------------------------------------------------------------------
// Edge cases and error resilience
// ---------------------------------------------------------------------------

describe("edge cases", () => {
  test("empty stream produces finish event", async () => {
    const events = await parseOne("")

    const finish = eventsOfType(events, "finish")
    expect(finish).toHaveLength(1)
  })

  test("stream-start is first event", async () => {
    const events = await parseOne("<|channel|>final<|message|>Hi<|end|>")

    expect(events[0].type).toBe("stream-start")
  })

  test("finish is last event", async () => {
    const events = await parseOne("<|channel|>final<|message|>Hi<|end|>")

    expect(events[events.length - 1].type).toBe("finish")
  })

  test("truncated stream (no <|end|>) sets finish reason to length", async () => {
    const events = await parseOne(
      "<|channel|>final<|message|>This is truncated content",
    )

    expect(collectText(events)).toBe("This is truncated content")
    expect(finishReason(events)).toBe("length")
  })

  test("incomplete tool call (name but no args) on flush is discarded", async () => {
    const events = await parseOne(
      "<|channel|>commentary to=functions.write <|constrain|>json<|message|>",
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(0)
    expect(finishReason(events)).toBe("length")
  })

  test("text with <|return|> stop token", async () => {
    const events = await parseOne(
      "<|channel|>final<|message|>Done.<|return|>",
    )

    expect(collectText(events)).toBe("Done.")
    expect(finishReason(events)).toBe("stop")
  })

  test("parser never throws on malformed input", async () => {
    // Garbage input should not throw
    const events = await parseOne(
      "random text with no control tokens at all",
    )

    expect(eventsOfType(events, "finish")).toHaveLength(1)
  })

  test("many small chunks", async () => {
    const text = "<|channel|>final<|message|>Hello!<|end|>"
    // Split into single-character chunks
    const chunks = text.split("")
    const events = await parse(chunks)

    expect(collectText(events)).toBe("Hello!")
    expect(finishReason(events)).toBe("stop")
  })
})

// ---------------------------------------------------------------------------
// Tool name fuzzy matching
// ---------------------------------------------------------------------------

describe("tool name fuzzy matching", () => {
  test("truncated tool name resolved via prefix match", async () => {
    const events = await parse(
      ['<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"cmd":"ls"}<|call|>'],
      ["bash", "read", "write", "glob", "grep"],
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("bash")
  })

  test("ambiguous prefix keeps original name", async () => {
    // "g" matches both "glob" and "grep" — ambiguous, keep "g"
    const events = await parse(
      ['<|channel|>commentary to=functions.g <|constrain|>json<|message|>{"q":"test"}<|call|>'],
      ["bash", "read", "write", "glob", "grep"],
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("g")
  })

  test("exact match used even when prefix would also match", async () => {
    const events = await parse(
      ['<|channel|>commentary to=functions.read <|constrain|>json<|message|>{"path":"a.txt"}<|call|>'],
      ["read", "readline"],
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("read")
  })

  test("no known tools list skips fuzzy matching", async () => {
    const events = await parse(
      ['<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"cmd":"ls"}<|call|>'],
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("b")
  })

  test("multi-char prefix resolves correctly", async () => {
    const events = await parse(
      ['<|channel|>commentary to=functions.wr <|constrain|>json<|message|>{"path":"x"}<|call|>'],
      ["bash", "read", "write", "glob", "grep"],
    )

    const toolCalls = eventsOfType(events, "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0].toolName).toBe("write")
  })
})
