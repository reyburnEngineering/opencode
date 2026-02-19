import type { LanguageModelV2CallOptions } from "@ai-sdk/provider"
import type { JSONSchema7 } from "@ai-sdk/provider"
import type { ModelMessage } from "ai"

// Harmony control tokens
const START = "<|start|>"
const END = "<|end|>"
const MESSAGE = "<|message|>"
const CHANNEL = "<|channel|>"
const CONSTRAIN = "<|constrain|>"
const CALL = "<|call|>"
const RETURN = "<|return|>"

/**
 * Convert a JSON Schema type definition into TypeScript type syntax.
 *
 * Handles primitives, objects (with required vs optional fields), arrays,
 * enums (as unions of literals), union types (anyOf/oneOf), and nested objects.
 */
export function jsonSchemaToTypeScript(schema: JSONSchema7, name?: string): string {
  if (typeof schema === "boolean") {
    return "any"
  }

  // Enum values -> union of literals
  if (schema.enum) {
    return schema.enum.map((v) => JSON.stringify(v)).join(" | ")
  }

  // anyOf / oneOf -> union types
  const unionSchemas = schema.anyOf ?? schema.oneOf
  if (unionSchemas) {
    const parts = unionSchemas
      .filter((s): s is JSONSchema7 => typeof s !== "boolean")
      .map((s) => jsonSchemaToTypeScript(s))
    return parts.join(" | ")
  }

  // Handle by type
  const type = schema.type

  if (type === "string") return "string"
  if (type === "number" || type === "integer") return "number"
  if (type === "boolean") return "boolean"
  if (type === "null") return "null"

  if (type === "array") {
    const items = schema.items
    if (items && typeof items !== "boolean") {
      return `${jsonSchemaToTypeScript(items as JSONSchema7)}[]`
    }
    return "any[]"
  }

  if (type === "object" || schema.properties) {
    const props = schema.properties ?? {}
    const required = new Set(schema.required ?? [])
    const lines: string[] = []

    for (const [key, value] of Object.entries(props)) {
      if (typeof value === "boolean") continue
      const propSchema = value as JSONSchema7
      const isRequired = required.has(key)
      const description = propSchema.description
      const typeDef = jsonSchemaToTypeScript(propSchema)

      if (description) {
        lines.push(`// ${description}`)
      }
      lines.push(`${key}${isRequired ? "" : "?"}: ${typeDef},`)
    }

    if (lines.length === 0) {
      return "object"
    }

    return `{\n${lines.join("\n")}\n}`
  }

  // Fallback for unrecognized or missing type
  return "any"
}

/**
 * Convert AI SDK tool definitions into Harmony's TypeScript-style namespace format.
 *
 * Output format:
 * ```
 * namespace functions {
 *
 * // Description
 * type tool_name = (_: { param: type }) => any;
 *
 * } // namespace functions
 * ```
 */
export function renderTools(tools: LanguageModelV2CallOptions["tools"]): string {
  if (!tools || tools.length === 0) return ""

  const declarations: string[] = []

  for (const tool of tools) {
    if (tool.type === "provider-defined") continue

    const lines: string[] = []

    // Description comment
    if (tool.description) {
      lines.push(`// ${tool.description}`)
    }

    const schema = tool.inputSchema as JSONSchema7 | undefined
    const hasProperties = schema && typeof schema !== "boolean" && schema.properties && Object.keys(schema.properties).length > 0

    if (!hasProperties) {
      // No parameters
      lines.push(`type ${tool.name} = () => any;`)
    } else {
      // Build parameter object type inline
      const paramType = renderParamObject(schema as JSONSchema7)
      lines.push(`type ${tool.name} = (_: ${paramType}) => any;`)
    }

    declarations.push(lines.join("\n"))
  }

  if (declarations.length === 0) return ""

  return [
    "namespace functions {",
    "",
    declarations.join("\n\n"),
    "",
    "} // namespace functions",
  ].join("\n")
}

/**
 * Render a JSON Schema object as a Harmony-style inline parameter type.
 * Uses inline comments for field descriptions.
 */
function renderParamObject(schema: JSONSchema7): string {
  if (typeof schema === "boolean") return "object"

  const props = schema.properties ?? {}
  const required = new Set(schema.required ?? [])
  const entries = Object.entries(props)

  if (entries.length === 0) return "object"

  const lines: string[] = []

  for (const [key, value] of entries) {
    if (typeof value === "boolean") continue
    const propSchema = value as JSONSchema7
    const isRequired = required.has(key)
    const description = propSchema.description
    const typeDef = jsonSchemaToTypeScript(propSchema)

    if (description) {
      lines.push(`// ${description}`)
    }
    lines.push(`${key}${isRequired ? "" : "?"}: ${typeDef},`)
  }

  return `{\n${lines.join("\n")}\n}`
}

/**
 * Convert AI SDK ModelMessage[] into a Harmony control token sequence.
 *
 * Rules:
 * - System messages:    <|start|>system<|message|>{content}<|end|>
 * - User messages:      <|start|>user<|message|>{content}<|end|>
 * - Assistant messages:  <|start|>assistant<|channel|>{channel}<|message|>{content}<|end|>
 *   - ReasoningParts  -> analysis channel
 *   - TextParts       -> final channel
 *   - ToolCallParts   -> commentary channel with <|call|> token
 * - Tool messages:      <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{output}<|end|>
 *
 * When storing history, <|return|> is replaced with <|end|>.
 */
export function renderMessages(messages: ModelMessage[]): string {
  const parts: string[] = []

  for (const msg of messages) {
    switch (msg.role) {
      case "system": {
        const content = typeof msg.content === "string" ? msg.content : ""
        parts.push(`${START}system${MESSAGE}${content}${END}\n`)
        break
      }

      case "user": {
        const content = extractUserContent(msg)
        parts.push(`${START}user${MESSAGE}${content}${END}\n`)
        break
      }

      case "assistant": {
        if (!Array.isArray(msg.content)) {
          // Simple string content -> final channel
          const content = typeof msg.content === "string" ? msg.content : ""
          parts.push(`${START}assistant${CHANNEL}final${MESSAGE}${content}${END}\n`)
          break
        }

        // Collect parts by type for channel grouping
        const reasoningParts: string[] = []
        const textParts: string[] = []
        const toolCallParts: Array<{
          toolCallId: string
          toolName: string
          input: unknown
        }> = []

        for (const part of msg.content) {
          switch (part.type) {
            case "reasoning": {
              if (part.text) {
                reasoningParts.push(part.text)
              }
              break
            }
            case "text": {
              textParts.push(part.text)
              break
            }
            case "tool-call": {
              toolCallParts.push({
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                input: part.input,
              })
              break
            }
          }
        }

        // Emit analysis channel for reasoning (chain-of-thought)
        if (reasoningParts.length > 0) {
          const reasoning = reasoningParts.join("")
          parts.push(`${START}assistant${CHANNEL}analysis${MESSAGE}${reasoning}${END}\n`)
        }

        // Emit commentary channel for each tool call
        for (const tc of toolCallParts) {
          const args = JSON.stringify(tc.input ?? {})
          parts.push(
            `${START}assistant${CHANNEL}commentary to=functions.${tc.toolName} ${CONSTRAIN}json${MESSAGE}${args}${CALL}\n`,
          )
        }

        // Emit final channel for text content
        if (textParts.length > 0) {
          const text = textParts.join("")
          parts.push(`${START}assistant${CHANNEL}final${MESSAGE}${text}${END}\n`)
        }

        // If there were no parts at all, emit an empty final message
        if (reasoningParts.length === 0 && toolCallParts.length === 0 && textParts.length === 0) {
          parts.push(`${START}assistant${CHANNEL}final${MESSAGE}${END}\n`)
        }

        break
      }

      case "tool": {
        if (!Array.isArray(msg.content)) break

        for (const part of msg.content) {
          if (part.type !== "tool-result") continue

          const output = renderToolOutput(part.output)
          const toolName = part.toolName

          // Tool result format: <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{output}<|end|>
          parts.push(
            `${START}functions.${toolName} to=assistant${CHANNEL}commentary${MESSAGE}${output}${END}\n`,
          )
        }
        break
      }
    }
  }

  // Prime the model to generate as assistant. With raw completions API
  // the model needs an explicit start token to know it should continue
  // as the assistant in Harmony format.
  parts.push(`${START}assistant`)

  return parts.join("")
}

/**
 * Extract text content from a user message, handling both string and array formats.
 */
function extractUserContent(msg: ModelMessage): string {
  if (typeof msg.content === "string") return msg.content

  if (Array.isArray(msg.content)) {
    const textParts: string[] = []
    for (const part of msg.content) {
      if (part.type === "text") {
        textParts.push(part.text)
      }
    }
    return textParts.join("")
  }

  return ""
}

/**
 * Render a tool output value to a string suitable for the Harmony format.
 */
function renderToolOutput(output: { type: string; value: unknown }): string {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value as string
    case "json":
    case "error-json":
    case "content":
      return JSON.stringify(output.value)
    default:
      return String(output.value)
  }
}

/**
 * Return a string to inject into the system message configuring reasoning effort.
 *
 * Valid levels: "low", "medium", "high"
 *
 * In Harmony format this appears in the system message as:
 *   Reasoning: {level}
 */
export function renderReasoningLevel(level: "low" | "medium" | "high"): string {
  return `Reasoning: ${level}`
}
