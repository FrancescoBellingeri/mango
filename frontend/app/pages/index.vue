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
  <div class="flex flex-col h-screen bg-gray-950 text-gray-100 font-sans">

    <!-- Header -->
    <header class="shrink-0 flex items-center justify-between px-6 py-4 border-b border-gray-800/60">
      <div class="flex items-center gap-2.5">
        <span class="text-xl leading-none">🥭</span>
        <span class="font-semibold tracking-tight">Mango</span>
        <span class="text-xs text-gray-600 font-mono hidden sm:inline">MongoDB AI</span>
      </div>
      <button
        v-if="messages.length > 0"
        @click="resetSession"
        class="text-xs text-gray-600 hover:text-gray-400 transition-colors cursor-pointer"
      >
        New chat
      </button>
    </header>

    <!-- Messages -->
    <main class="flex-1 overflow-y-auto">
      <div class="max-w-3xl mx-auto px-4 py-8 space-y-8">

        <!-- Empty state -->
        <div v-if="messages.length === 0" class="flex flex-col items-center justify-center min-h-64 gap-6 text-center">
          <div>
            <p class="text-2xl font-semibold text-gray-300 mb-1">Ask your database</p>
            <p class="text-sm text-gray-600">Natural language → MongoDB query → answer</p>
          </div>
          <div class="flex flex-col gap-2 w-full max-w-md">
            <button
              v-for="s in SUGGESTIONS"
              :key="s"
              @click="input = s; nextTick(() => textarea?.focus())"
              class="text-left text-sm px-4 py-2.5 rounded-xl border border-gray-800 hover:border-gray-600 hover:bg-gray-800/50 text-gray-400 hover:text-gray-200 transition-all cursor-pointer"
            >
              {{ s }}
            </button>
          </div>
        </div>

        <!-- Message list -->
        <template v-for="msg in messages" :key="msg.id">

          <!-- User -->
          <div v-if="msg.role === 'user'" class="flex justify-end">
            <div class="max-w-xl bg-indigo-600 rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed">
              {{ msg.text }}
            </div>
          </div>

          <!-- Assistant -->
          <div v-else class="flex flex-col gap-2">

            <!-- Tool steps -->
            <div v-if="msg.toolSteps.length > 0" class="flex flex-col gap-1.5 pl-0.5">
              <div
                v-for="(step, i) in msg.toolSteps"
                :key="i"
                class="flex items-start gap-2.5 text-xs font-mono text-gray-500"
              >
                <!-- Status icon -->
                <span class="mt-0.5 shrink-0">
                  <span v-if="!step.done" class="inline-block w-3.5 h-3.5 rounded-full border border-gray-600 animate-pulse" />
                  <span v-else-if="step.success" class="text-emerald-500">✓</span>
                  <span v-else class="text-red-500">✗</span>
                </span>
                <div class="min-w-0">
                  <span class="text-gray-400 font-medium">{{ step.name }}</span>
                  <span v-if="step.preview" class="block text-gray-600 truncate mt-0.5">{{ step.preview }}</span>
                </div>
              </div>
            </div>

            <!-- Answer bubble -->
            <div class="bg-gray-800/70 border border-gray-700/50 rounded-2xl rounded-tl-sm px-5 py-4 text-sm leading-relaxed">
              <!-- Thinking indicator -->
              <span v-if="msg.streaming && !msg.text" class="flex gap-1 items-center text-gray-600">
                <span class="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style="animation-delay:0ms" />
                <span class="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style="animation-delay:150ms" />
                <span class="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style="animation-delay:300ms" />
              </span>
              <div v-else class="prose prose-invert prose-sm max-w-none" v-html="renderMarkdown(msg.text)" />
            </div>

            <!-- Metadata -->
            <div v-if="msg.meta" class="flex gap-4 pl-0.5 text-xs text-gray-700 font-mono">
              <span>{{ msg.meta.iterations }} {{ msg.meta.iterations === 1 ? 'step' : 'steps' }}</span>
              <span>{{ msg.meta.inputTokens + msg.meta.outputTokens }} tok</span>
              <span v-if="msg.meta.memoryHits > 0" class="text-indigo-800">{{ msg.meta.memoryHits }} from memory</span>
            </div>

          </div>
        </template>

        <div ref="messagesEnd" />
      </div>
    </main>

    <!-- Input -->
    <footer class="shrink-0 border-t border-gray-800/60 px-4 py-4">
      <form @submit.prevent="sendMessage" class="max-w-3xl mx-auto flex gap-3 items-end">
        <textarea
          ref="textarea"
          v-model="input"
          @keydown="onKeydown"
          @input="autoResize"
          :disabled="isLoading"
          placeholder="Ask your database…"
          rows="1"
          class="flex-1 resize-none bg-gray-800/80 border border-gray-700/60 rounded-xl px-4 py-3 text-sm placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-40 transition-all leading-relaxed"
        />
        <button
          type="submit"
          :disabled="isLoading || !input.trim()"
          class="shrink-0 px-5 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-600 rounded-xl text-sm font-medium transition-colors cursor-pointer disabled:cursor-default"
        >
          <span v-if="isLoading">
            <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          </span>
          <span v-else>Send</span>
        </button>
      </form>
      <p class="text-center text-xs text-gray-800 mt-2">Enter to send · Shift+Enter for newline</p>
    </footer>

  </div>
</template>

<style scoped>
/* Markdown prose styles for assistant messages */
.prose :deep(p) {
  margin-bottom: 0.75rem;
  color: #f3f4f6;
}
.prose :deep(p:last-child) {
  margin-bottom: 0;
}
.prose :deep(strong) {
  color: #fff;
  font-weight: 600;
}
.prose :deep(em) {
  color: #e5e7eb;
}
.prose :deep(code) {
  background-color: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 4px;
  padding: 0.1em 0.4em;
  font-size: 0.85em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #a5b4fc;
}
.prose :deep(pre) {
  background-color: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  padding: 1rem;
  overflow-x: auto;
  margin: 0.75rem 0;
}
.prose :deep(pre code) {
  background: none;
  border: none;
  padding: 0;
  font-size: 0.8rem;
  color: #d1d5db;
}
.prose :deep(ul) {
  list-style-type: disc;
  padding-left: 1.25rem;
  margin: 0.5rem 0 0.75rem;
  color: #f3f4f6;
}
.prose :deep(ol) {
  list-style-type: decimal;
  padding-left: 1.25rem;
  margin: 0.5rem 0 0.75rem;
  color: #f3f4f6;
}
.prose :deep(li) {
  margin-bottom: 0.2rem;
}
.prose :deep(h1),
.prose :deep(h2),
.prose :deep(h3) {
  color: #fff;
  font-weight: 600;
  margin-top: 1rem;
  margin-bottom: 0.4rem;
}
.prose :deep(h1) { font-size: 1.1rem; }
.prose :deep(h2) { font-size: 1rem; }
.prose :deep(h3) { font-size: 0.95rem; }
.prose :deep(blockquote) {
  border-left: 3px solid #4f46e5;
  padding-left: 0.75rem;
  color: #9ca3af;
  margin: 0.5rem 0;
}
.prose :deep(hr) {
  border-color: rgba(255,255,255,0.1);
  margin: 0.75rem 0;
}
.prose :deep(a) {
  color: #818cf8;
  text-decoration: underline;
}
</style>
