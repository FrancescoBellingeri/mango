<script setup lang="ts">
import { marked } from 'marked'

interface ToolStep {
  name: string
  args: Record<string, unknown>
  success?: boolean
  preview?: string
  done: boolean
}

interface MessageMeta {
  iterations: number
  inputTokens: number
  outputTokens: number
  memoryHits: number
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  toolSteps: ToolStep[]
  streaming: boolean
  meta?: MessageMeta
}

marked.setOptions({ breaks: true })

function renderMarkdown(text: string): string {
  return marked.parse(text) as string
}

const sessionId = ref<string | null>(null)
const messages = ref<Message[]>([])
const input = ref('')
const isLoading = ref(false)
const messagesEnd = ref<HTMLElement | null>(null)
const textarea = ref<HTMLTextAreaElement | null>(null)

function scrollToBottom() {
  nextTick(() => messagesEnd.value?.scrollIntoView({ behavior: 'smooth' }))
}

function autoResize() {
  if (!textarea.value) return
  textarea.value.style.height = 'auto'
  textarea.value.style.height = Math.min(textarea.value.scrollHeight, 160) + 'px'
}

async function sendMessage() {
  const question = input.value.trim()
  if (!question || isLoading.value) return

  input.value = ''
  nextTick(() => autoResize())
  isLoading.value = true

  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    text: question,
    toolSteps: [],
    streaming: false,
  })

  const assistantMsg: Message = reactive({
    id: crypto.randomUUID(),
    role: 'assistant',
    text: '',
    toolSteps: [],
    streaming: true,
  })
  messages.value.push(assistantMsg)
  scrollToBottom()

  try {
    const res = await fetch('/api/v1/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId.value }),
    })

    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const event = JSON.parse(line.slice(6))
          handleEvent(assistantMsg, event)
          scrollToBottom()
        } catch { /* malformed chunk, skip */ }
      }
    }
  } catch (err: unknown) {
    assistantMsg.text = `Error: ${err instanceof Error ? err.message : String(err)}`
  } finally {
    assistantMsg.streaming = false
    isLoading.value = false
    scrollToBottom()
  }
}

function handleEvent(msg: Message, event: Record<string, unknown>) {
  switch (event.type) {
    case 'session':
      sessionId.value = event.session_id as string
      break

    case 'tool_call':
      msg.toolSteps.push({
        name: event.tool_name as string,
        args: event.tool_args as Record<string, unknown>,
        done: false,
      })
      break

    case 'tool_result': {
      const step = [...msg.toolSteps].reverse().find(s => s.name === event.tool_name && !s.done)
      if (step) {
        step.success = event.success as boolean
        step.preview = event.preview as string
        step.done = true
      }
      break
    }

    case 'answer':
      msg.text = event.text as string
      break

    case 'done':
      msg.meta = {
        iterations: event.iterations as number,
        inputTokens: event.input_tokens as number,
        outputTokens: event.output_tokens as number,
        memoryHits: event.memory_hits as number,
      }
      break

    case 'error':
      msg.text = `⚠ ${event.message as string}`
      break
  }
}

function resetSession() {
  sessionId.value = null
  messages.value = []
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

const SUGGESTIONS = [
  'How many documents are in each collection?',
  'Show me the most recent records',
  'What fields does the main collection have?',
]
</script>

<template>
  <div class="flex flex-col h-screen font-sans" style="background: #111111; color: #ececec;">

    <!-- Header -->
    <header class="shrink-0 flex items-center justify-between px-6 py-3.5" style="background: #161616; border-bottom: 1px solid #2a2a2a;">
      <div class="flex items-center gap-3">
        <div class="flex items-center justify-center w-8 h-8 rounded-lg text-sm select-none" style="background: #f97316;">
          🥭
        </div>
        <span class="font-semibold text-base tracking-tight" style="color: #f0f0f0;">Mango</span>
        <span class="text-xs font-mono hidden sm:inline" style="color: #555;">MongoDB AI</span>
      </div>
      <button
        v-if="messages.length > 0"
        @click="resetSession"
        class="new-chat-btn text-xs font-medium px-3.5 py-1.5 rounded-lg transition-all cursor-pointer"
      >
        + New chat
      </button>
    </header>

    <!-- Messages -->
    <main class="flex-1 overflow-y-auto">
      <div class="max-w-2xl mx-auto px-4 py-10 space-y-6">

        <!-- Empty state -->
        <div v-if="messages.length === 0" class="flex flex-col items-center justify-center gap-8 text-center pt-24">
          <div>
            <div class="w-12 h-12 rounded-xl flex items-center justify-center text-2xl mx-auto mb-5" style="background: #f97316;">🥭</div>
            <p class="text-2xl font-semibold mb-2" style="color: #f0f0f0;">Ask your database</p>
            <p class="text-sm" style="color: #666;">Natural language → MongoDB query → answer</p>
          </div>
          <div class="flex flex-col gap-2 w-full max-w-md">
            <button
              v-for="s in SUGGESTIONS"
              :key="s"
              @click="input = s; nextTick(() => textarea?.focus())"
              class="suggestion-btn text-left text-sm px-4 py-3 rounded-xl transition-all cursor-pointer"
            >
              {{ s }}
            </button>
          </div>
        </div>

        <!-- Message list -->
        <template v-for="msg in messages" :key="msg.id">

          <!-- User -->
          <div v-if="msg.role === 'user'" class="flex justify-end">
            <div class="max-w-xl rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed user-bubble">
              {{ msg.text }}
            </div>
          </div>

          <!-- Assistant -->
          <div v-else class="flex flex-col gap-2">

            <!-- Tool steps -->
            <div v-if="msg.toolSteps.length > 0" class="flex flex-col gap-1.5 pl-1 mb-1">
              <div v-for="(step, i) in msg.toolSteps" :key="i" class="flex items-start gap-2 text-xs font-mono" style="color: #555;">
                <span class="mt-0.5 shrink-0">
                  <span v-if="!step.done" class="inline-block w-2.5 h-2.5 rounded-full border animate-pulse" style="border-color: #555;" />
                  <span v-else-if="step.success" style="color: #4ade80;">✓</span>
                  <span v-else style="color: #f87171;">✗</span>
                </span>
                <div class="min-w-0">
                  <span style="color: #888;">{{ step.name }}</span>
                  <span v-if="step.preview" class="block truncate mt-0.5" style="color: #444;">{{ step.preview }}</span>
                </div>
              </div>
            </div>

            <!-- Answer bubble -->
            <div class="assistant-bubble rounded-2xl rounded-tl-sm px-5 py-4 text-sm leading-relaxed">
              <span v-if="msg.streaming && !msg.text" class="flex gap-1.5 items-center h-5">
                <span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#f97316; animation-delay:0ms" />
                <span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#f97316; animation-delay:160ms" />
                <span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#f97316; animation-delay:320ms" />
              </span>
              <div v-else class="prose prose-sm max-w-none" v-html="renderMarkdown(msg.text)" />
            </div>

            <!-- Metadata -->
            <div v-if="msg.meta" class="flex gap-4 pl-1 text-xs font-mono" style="color: #444;">
              <span>{{ msg.meta.iterations }} {{ msg.meta.iterations === 1 ? 'step' : 'steps' }}</span>
              <span>{{ msg.meta.inputTokens + msg.meta.outputTokens }} tok</span>
              <span v-if="msg.meta.memoryHits > 0" style="color: #f97316;">{{ msg.meta.memoryHits }} from memory</span>
            </div>

          </div>
        </template>

        <div ref="messagesEnd" />
      </div>
    </main>

    <!-- Input -->
    <footer class="shrink-0 px-4 pb-5 pt-3" style="background: #161616; border-top: 1px solid #2a2a2a;">
      <form @submit.prevent="sendMessage" class="max-w-2xl mx-auto flex gap-2.5 items-center">
        <div class="flex-1 input-wrapper rounded-2xl">
          <textarea
            ref="textarea"
            v-model="input"
            @keydown="onKeydown"
            @input="autoResize"
            :disabled="isLoading"
            placeholder="Ask your database…"
            rows="1"
            class="input-textarea w-full resize-none px-4 py-3.5 text-sm leading-relaxed disabled:opacity-40 bg-transparent outline-none"
          />
        </div>
        <button
          type="submit"
          :disabled="isLoading || !input.trim()"
          class="send-btn shrink-0 w-11 h-11 rounded-xl flex items-center justify-center transition-all cursor-pointer disabled:cursor-default"
        >
          <span v-if="isLoading">
            <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          </span>
          <span v-else>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
            </svg>
          </span>
        </button>
      </form>
      <p class="text-center text-xs mt-2" style="color: #3a3a3a;">Enter to send · Shift+Enter for newline</p>
    </footer>

  </div>
</template>

<style scoped>
/* New chat button */
.new-chat-btn {
  color: #888;
  border: 1px solid #2a2a2a;
  background: transparent;
}
.new-chat-btn:hover {
  color: #f97316;
  border-color: #f97316;
  background: rgba(249, 115, 22, 0.08);
}

/* User bubble — orange accent */
.user-bubble {
  background: #f97316;
  color: #fff;
}

/* Assistant bubble */
.assistant-bubble {
  background: #1e1e1e;
  border: 1px solid #2a2a2a;
  color: #e0e0e0;
}

/* Suggestion buttons */
.suggestion-btn {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  color: #888;
}
.suggestion-btn:hover {
  background: #222;
  border-color: #444;
  color: #e0e0e0;
}

/* Input */
.input-wrapper {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.input-wrapper:focus-within {
  border-color: #f97316;
  box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1);
}
.input-textarea {
  color: #e0e0e0;
}
.input-textarea::placeholder {
  color: #555;
}

/* Send button — orange accent */
.send-btn {
  background: #f97316;
  color: #fff;
}
.send-btn:hover:not(:disabled) {
  background: #fb923c;
  transform: translateY(-1px);
}
.send-btn:disabled {
  background: #222;
  color: #444;
}

/* Markdown prose */
.prose :deep(p) { margin-bottom: 0.75rem; color: #e0e0e0; line-height: 1.7; }
.prose :deep(p:last-child) { margin-bottom: 0; }
.prose :deep(strong) { color: #f5f5f5; font-weight: 600; }
.prose :deep(em) { color: #ccc; }
.prose :deep(code) {
  background: #111;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 0.12em 0.4em;
  font-size: 0.83em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #e0e0e0;
}
.prose :deep(pre) {
  background: #111;
  border: 1px solid #2a2a2a;
  border-radius: 10px;
  padding: 1rem 1.25rem;
  overflow-x: auto;
  margin: 0.75rem 0;
}
.prose :deep(pre code) { background: none; border: none; padding: 0; font-size: 0.8rem; color: #ccc; }
.prose :deep(ul), .prose :deep(ol) { padding-left: 1.4rem; margin: 0.5rem 0 0.75rem; color: #e0e0e0; }
.prose :deep(ul) { list-style-type: disc; }
.prose :deep(ol) { list-style-type: decimal; }
.prose :deep(li) { margin-bottom: 0.25rem; }
.prose :deep(h1), .prose :deep(h2), .prose :deep(h3) { color: #f5f5f5; font-weight: 600; margin-top: 1rem; margin-bottom: 0.4rem; }
.prose :deep(h1) { font-size: 1.1rem; }
.prose :deep(h2) { font-size: 1rem; }
.prose :deep(h3) { font-size: 0.95rem; }
.prose :deep(blockquote) { border-left: 3px solid #333; padding-left: 0.75rem; color: #888; margin: 0.5rem 0; }
.prose :deep(hr) { border-color: #2a2a2a; margin: 0.75rem 0; }
.prose :deep(a) { color: #f97316; text-decoration: underline; }
.prose :deep(a:hover) { color: #fb923c; }
.prose :deep(table) { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.85rem; }
.prose :deep(th) { background: #1a1a1a; color: #e0e0e0; padding: 0.5rem 0.75rem; text-align: left; border: 1px solid #2a2a2a; font-weight: 600; }
.prose :deep(td) { padding: 0.4rem 0.75rem; border: 1px solid #222; color: #ccc; }
.prose :deep(tr:nth-child(even) td) { background: rgba(255,255,255,0.02); }
</style>
