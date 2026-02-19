import { describe, expect, test } from "bun:test"
import {
  jsonSchemaToTypeScript,
  renderTools,
  renderMessages,
  renderReasoningLevel,
} from "../../../src/provider/sdk/harmony/harmony-renderer"

// ---------------------------------------------------------------------------
// jsonSchemaToTypeScript
// ---------------------------------------------------------------------------

describe("jsonSchemaToTypeScript", () => {
  test("primitives", () => {
    expect(jsonSchemaToTypeScript({ type: "string" })).toBe("string")
    expect(jsonSchemaToTypeScript({ type: "number" })).toBe("number")
    expect(jsonSchemaToTypeScript({ type: "integer" })).toBe("number")
    expect(jsonSchemaToTypeScript({ type: "boolean" })).toBe("boolean")
    expect(jsonSchemaToTypeScript({ type: "null" })).toBe("null")
  })

  test("array of strings", () => {
    expect(
      jsonSchemaToTypeScript({ type: "array", items: { type: "string" } }),
    ).toBe("string[]")
  })

  test("array without items", () => {
    expect(jsonSchemaToTypeScript({ type: "array" })).toBe("any[]")
  })

  test("enum values", () => {
    expect(
      jsonSchemaToTypeScript({ enum: ["a", "b", "c"] }),
    ).toBe('"a" | "b" | "c"')
  })

  test("object with required and optional fields", () => {
    const ts = jsonSchemaToTypeScript({
      type: "object",
      properties: {
        name: { type: "string" },
        age: { type: "number" },
      },
      required: ["name"],
    })
    expect(ts).toContain("name: string,")
    expect(ts).toContain("age?: number,")
  })

  test("nested object", () => {
    const ts = jsonSchemaToTypeScript({
      type: "object",
      properties: {
        address: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
          required: ["city"],
        },
      },
      required: ["address"],
    })
    expect(ts).toContain("address: {")
    expect(ts).toContain("city: string,")
  })

  test("anyOf union type", () => {
    const ts = jsonSchemaToTypeScript({
      anyOf: [{ type: "string" }, { type: "number" }],
    })
    expect(ts).toBe("string | number")
  })

  test("boolean schema returns any", () => {
    expect(jsonSchemaToTypeScript(true as any)).toBe("any")
  })

  test("empty object", () => {
    expect(
      jsonSchemaToTypeScript({ type: "object", properties: {} }),
    ).toBe("object")
  })

  test("field descriptions as comments", () => {
    const ts = jsonSchemaToTypeScript({
      type: "object",
      properties: {
        path: { type: "string", description: "The file path" },
      },
      required: ["path"],
    })
    expect(ts).toContain("// The file path")
    expect(ts).toContain("path: string,")
  })
})

// ---------------------------------------------------------------------------
// renderTools
// ---------------------------------------------------------------------------

describe("renderTools", () => {
  test("empty tools returns empty string", () => {
    expect(renderTools([])).toBe("")
    expect(renderTools(undefined)).toBe("")
  })

  test("single tool with parameters", () => {
    const result = renderTools([
      {
        type: "function",
        name: "write",
        description: "Write to a file",
        inputSchema: {
          type: "object",
          properties: {
            filePath: { type: "string", description: "Path" },
            content: { type: "string", description: "Content" },
          },
          required: ["filePath", "content"],
        },
      },
    ])

    expect(result).toContain("namespace functions {")
    expect(result).toContain("} // namespace functions")
    expect(result).toContain("// Write to a file")
    expect(result).toContain("type write = (_: {")
    expect(result).toContain("filePath: string,")
    expect(result).toContain("content: string,")
  })

  test("tool with no parameters", () => {
    const result = renderTools([
      {
        type: "function",
        name: "getTime",
        description: "Get current time",
        inputSchema: { type: "object", properties: {} },
      },
    ])

    expect(result).toContain("type getTime = () => any;")
  })

  test("multiple tools", () => {
    const result = renderTools([
      {
        type: "function",
        name: "read",
        description: "Read a file",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
      {
        type: "function",
        name: "write",
        description: "Write a file",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" }, content: { type: "string" } },
          required: ["path", "content"],
        },
      },
    ])

    expect(result).toContain("type read = (_: {")
    expect(result).toContain("type write = (_: {")
  })

  test("skips provider-defined tools", () => {
    const result = renderTools([
      { type: "provider-defined", name: "internal", id: "x" } as any,
      {
        type: "function",
        name: "hello",
        description: "Say hi",
        inputSchema: { type: "object", properties: {} },
      },
    ])

    expect(result).not.toContain("internal")
    expect(result).toContain("type hello = () => any;")
  })
})

// ---------------------------------------------------------------------------
// renderMessages
// ---------------------------------------------------------------------------

describe("renderMessages", () => {
  test("system message", () => {
    const result = renderMessages([
      { role: "system", content: "You are helpful." },
    ])

    expect(result).toContain("<|start|>system<|message|>You are helpful.<|end|>")
    // Should end with assistant priming
    expect(result).toEndWith("<|start|>assistant")
  })

  test("user message with text array", () => {
    const result = renderMessages([
      {
        role: "user",
        content: [{ type: "text", text: "Hello there" }],
      },
    ])

    expect(result).toContain("<|start|>user<|message|>Hello there<|end|>")
  })

  test("assistant text goes to final channel", () => {
    const result = renderMessages([
      {
        role: "assistant",
        content: [{ type: "text", text: "Sure thing." }],
      },
    ])

    expect(result).toContain("<|start|>assistant<|channel|>final<|message|>Sure thing.<|end|>")
  })

  test("assistant reasoning goes to analysis channel", () => {
    const result = renderMessages([
      {
        role: "assistant",
        content: [{ type: "reasoning", text: "Let me think..." }],
      },
    ])

    expect(result).toContain("<|start|>assistant<|channel|>analysis<|message|>Let me think...<|end|>")
  })

  test("assistant tool call goes to commentary channel", () => {
    const result = renderMessages([
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "tc-1",
            toolName: "write",
            input: { filePath: "a.txt", content: "hi" },
          },
        ],
      },
    ])

    expect(result).toContain("<|start|>assistant<|channel|>commentary to=functions.write")
    expect(result).toContain("<|constrain|>json<|message|>")
    expect(result).toContain('"filePath":"a.txt"')
    expect(result).toContain("<|call|>")
  })

  test("tool result message", () => {
    const result = renderMessages([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc-1",
            toolName: "write",
            output: { type: "text", value: "File written." },
          },
        ],
      },
    ])

    expect(result).toContain("<|start|>functions.write to=assistant<|channel|>commentary<|message|>File written.<|end|>")
  })

  test("multi-turn conversation ordering", () => {
    const result = renderMessages([
      { role: "system", content: "Be helpful." },
      { role: "user", content: [{ type: "text", text: "Hi" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
      },
      { role: "user", content: [{ type: "text", text: "Bye" }] },
    ])

    const systemIdx = result.indexOf("<|start|>system")
    const user1Idx = result.indexOf("<|start|>user<|message|>Hi")
    const assistantIdx = result.indexOf("<|start|>assistant<|channel|>final")
    const user2Idx = result.indexOf("<|start|>user<|message|>Bye")
    const primingIdx = result.lastIndexOf("<|start|>assistant")

    expect(systemIdx).toBeLessThan(user1Idx)
    expect(user1Idx).toBeLessThan(assistantIdx)
    expect(assistantIdx).toBeLessThan(user2Idx)
    expect(user2Idx).toBeLessThan(primingIdx)
  })

  test("always ends with assistant priming", () => {
    const result = renderMessages([
      { role: "user", content: [{ type: "text", text: "Test" }] },
    ])

    expect(result).toEndWith("<|start|>assistant")
  })
})

// ---------------------------------------------------------------------------
// renderReasoningLevel
// ---------------------------------------------------------------------------

describe("renderReasoningLevel", () => {
  test("formats level correctly", () => {
    expect(renderReasoningLevel("low")).toBe("Reasoning: low")
    expect(renderReasoningLevel("medium")).toBe("Reasoning: medium")
    expect(renderReasoningLevel("high")).toBe("Reasoning: high")
  })
})
