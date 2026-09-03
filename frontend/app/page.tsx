"use client";

import Image from "next/image";
import { AnimatePresence, motion } from "motion/react";
import { startTransition, useEffect, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Conversation,
  ConversationContent,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { PromptInput, PromptInputSubmit, PromptInputTextarea } from "@/components/ai-elements/prompt-input";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { AlertCircle, Menu, MessageSquarePlus, PanelLeftClose, Plus, Settings2 } from "lucide-react";
import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

type ChatSession = {
  id: string;
  title: string;
  messages: ChatMessage[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const getErrorMessage = (response: Response, detail?: string) => {
  if (response.status === 429) {
    return "You are moving faster than dumbGPT can think. Please wait one second before sending another message.";
  }
  if (response.status === 401) {
    return "Your session is not authenticated. Sign in before chatting with dumbGPT.";
  }
  if (response.status === 503) {
    return "The chat service is not configured yet. Check the backend connection and try again.";
  }
  return detail || "Something went wrong while reaching dumbGPT. Please try again.";
};

export default function Home() {
  const [chats, setChats] = useState<ChatSession[]>([
    { id: "dumbgpt-chat", title: "New chat", messages: [] },
  ]);
  const [activeChatId, setActiveChatId] = useState("dumbgpt-chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [userId, setUserId] = useState("anonymous");
  const [hydrated, setHydrated] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [status, setStatus] = useState<"ready" | "submitted" | "error">("ready");
  const [error, setError] = useState<string | null>(null);
  const abortController = useRef<AbortController | null>(null);
  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? chats[0];
  const messages = activeChat.messages;
  const hasConversation = messages.length > 0 || status === "submitted";

  useEffect(() => {
    const storedUserId = localStorage.getItem("dumbgpt-user-id") ?? crypto.randomUUID();
    const storedChats = localStorage.getItem("dumbgpt-chats");
    startTransition(() => setUserId(storedUserId));
    localStorage.setItem("dumbgpt-user-id", storedUserId);
    if (storedChats) {
      try {
        const parsedChats = JSON.parse(storedChats) as ChatSession[];
        if (parsedChats.length > 0) {
          startTransition(() => {
            setChats(parsedChats);
            setActiveChatId(parsedChats[0].id);
          });
        }
      } catch {
        localStorage.removeItem("dumbgpt-chats");
      }
    }
    startTransition(() => setHydrated(true));
  }, []);

  useEffect(() => {
    if (hydrated) {
      localStorage.setItem("dumbgpt-chats", JSON.stringify(chats));
    }
  }, [chats, hydrated]);

  const updateActiveChat = (update: (chat: ChatSession) => ChatSession) => {
    setChats((current) =>
      current.map((chat) => (chat.id === activeChatId ? update(chat) : chat))
    );
  };

  const createNewChat = () => {
    abortController.current?.abort();
    const id = crypto.randomUUID();
    setChats((current) => [
      ...current,
      { id, title: "New chat", messages: [] },
    ]);
    setActiveChatId(id);
    setStatus("ready");
    setError(null);
    setSidebarOpen(false);
  };

  const selectChat = (id: string) => {
    abortController.current?.abort();
    setActiveChatId(id);
    setStatus("ready");
    setError(null);
    setSidebarOpen(false);
  };

  const handleSubmit = async ({ text }: PromptInputMessage) => {
    const message = text.trim();
    if (!message || status === "submitted") return;

    setError(null);
    setStatus("submitted");
    abortController.current = new AbortController();
    updateActiveChat((chat) => ({
      ...chat,
      title: chat.title === "New chat" ? message.slice(0, 36) : chat.title,
      messages: [
        ...chat.messages,
        { id: crypto.randomUUID(), role: "user", text: message },
      ],
    }));

    try {
      const response = await fetch(`${API_URL}/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-ID": userId },
        signal: abortController.current.signal,
        body: JSON.stringify({
          message,
          chat_id: activeChatId,
          max_new_tokens: maxNewTokens,
          temperature,
          top_k: topK,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(getErrorMessage(response, payload.detail));
      }
      updateActiveChat((chat) => ({
        ...chat,
        messages: [
          ...chat.messages,
          { id: crypto.randomUUID(), role: "assistant", text: payload.message },
        ],
      }));
      setStatus("ready");
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") {
        return;
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Something went wrong while reaching dumbGPT. Please try again."
      );
      setStatus("error");
    } finally {
      abortController.current = null;
    }
  };

  const handleStop = () => {
    abortController.current?.abort();
    setStatus("ready");
    setError("Generation stopped. Your message is still in the conversation.");
  };

  return (
    <main className="relative flex h-dvh max-h-dvh min-h-0 flex-1 flex-col overflow-hidden bg-[#101112] text-zinc-100">
      <header className="relative z-10 flex h-16 shrink-0 items-center border-b border-white/[0.07] px-5 sm:px-8" />

      <motion.div
        layout
        transition={{ type: "spring", stiffness: 320, damping: 30 }}
        className={hasConversation
          ? "absolute left-5 top-4 z-10 flex items-center gap-2.5 sm:left-8"
          : "absolute left-1/2 top-1/2 z-10 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-4"}
      >
        <Image className={hasConversation ? "size-7 rounded-full object-cover ring-1 ring-white/15" : "size-20 rounded-full object-cover ring-1 ring-white/15 shadow-2xl shadow-black/30"} src="/logo.jpeg" alt="dumbGPT" width={80} height={80} priority />
        <motion.span layout className={hasConversation ? "text-sm font-medium tracking-tight text-zinc-300" : "text-xl font-medium tracking-tight text-zinc-100"}>dumbGPT</motion.span>
      </motion.div>

      {!sidebarOpen && (
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="absolute left-4 top-20 z-30 border-white/10 bg-[#191b1d]/90 shadow-lg backdrop-blur hover:bg-[#242628]"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open chat history"
        >
          <Menu />
        </Button>
      )}

      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className="absolute inset-y-0 left-0 z-20 flex w-72 flex-col border-r border-white/[0.07] bg-[#151718] px-3 pb-5 pt-5 shadow-2xl shadow-black/30"
          >
            <div className="flex items-center justify-between px-2 pb-5 pt-1">
              <div>
                <p className="text-sm font-medium text-zinc-100">Your chats</p>
                <p className="mt-1 text-xs text-zinc-500">A quiet place for old thoughts</p>
              </div>
              <Button type="button" variant="ghost" size="icon-sm" onClick={() => setSidebarOpen(false)} aria-label="Collapse chat history">
                <PanelLeftClose />
              </Button>
            </div>
            <Button type="button" variant="secondary" className="mb-4 w-full justify-start gap-2" onClick={createNewChat}>
              <Plus />
              New chat
            </Button>
            <div className="flex flex-col gap-1 overflow-y-auto">
              {chats.map((chat) => (
                <button
                  type="button"
                  key={chat.id}
                  onClick={() => selectChat(chat.id)}
                  className={`flex items-center gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${chat.id === activeChatId ? "bg-primary/15 text-zinc-100" : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"}`}
                >
                  <MessageSquarePlus className="size-4 shrink-0" />
                  <span className="truncate">{chat.title}</span>
                </button>
              ))}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <Conversation className="mx-auto min-h-0 w-full max-w-3xl">
        <ConversationContent className="gap-7 px-5 py-8 sm:px-8">
          {messages.length > 0 && (
            messages.map((message) => (
              <Message key={message.id} from={message.role}>
                <MessageContent className={message.role === "assistant" ? "max-w-2xl text-[15px] leading-7 text-zinc-300" : "max-w-[85%] bg-[#242628] text-zinc-100"}>
                  {message.role === "assistant" ? <MessageResponse>{message.text}</MessageResponse> : message.text}
                </MessageContent>
              </Message>
            ))
          )}
          {status === "submitted" && (
            <Message from="assistant">
              <MessageContent className="text-sm text-zinc-400">
                <Shimmer className="[--color-background:#93c5fd]" duration={1.7}>Lemme think I&apos;m not that smart...</Shimmer>
              </MessageContent>
            </Message>
          )}
        </ConversationContent>
      </Conversation>

      <div className="mx-auto w-full max-w-3xl shrink-0 px-5 pb-6 sm:px-8 sm:pb-8">
        {error && (
          <Alert variant="destructive" className="mb-3 border-red-400/20 bg-red-950/30 text-red-200">
            <AlertCircle />
            <AlertTitle>Couldn&apos;t send that</AlertTitle>
            <AlertDescription className="text-red-200/70">{error}</AlertDescription>
          </Alert>
        )}
        {settingsOpen && (
          <div className="mb-3 grid grid-cols-1 gap-3 rounded-xl border border-white/10 bg-[#191b1d] p-4 text-xs text-zinc-400 sm:grid-cols-3">
            <label className="flex items-center justify-between gap-3 sm:flex-col sm:items-stretch">
              <span>Response length</span>
              <input className="w-32 accent-primary sm:w-full" type="range" min="1" max="512" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} />
              <output className="w-8 text-right text-zinc-200 sm:w-auto sm:text-left">{maxNewTokens}</output>
            </label>
            <label className="flex items-center justify-between gap-3 sm:flex-col sm:items-stretch">
              <span>Temperature</span>
              <input className="w-32 accent-primary sm:w-full" type="range" min="0.1" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
              <output className="w-8 text-right text-zinc-200 sm:w-auto sm:text-left">{temperature.toFixed(1)}</output>
            </label>
            <label className="flex items-center justify-between gap-3 sm:flex-col sm:items-stretch">
              <span>Top K</span>
              <input className="w-32 accent-primary sm:w-full" type="range" min="1" max="200" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
              <output className="w-8 text-right text-zinc-200 sm:w-auto sm:text-left">{topK}</output>
            </label>
          </div>
        )}
        <PromptInput onSubmit={handleSubmit} className="rounded-xl border-white/10 bg-[#191b1d] shadow-2xl shadow-black/20">
          <PromptInputTextarea placeholder="Don&apos;t ask me anything difficult" disabled={status === "submitted"} />
          <div className="flex items-center justify-between px-3 pb-3">
            <Button type="button" variant="ghost" size="icon-sm" onClick={() => setSettingsOpen((open) => !open)} aria-label={settingsOpen ? "Hide generation settings" : "Show generation settings"}>
              <Settings2 />
            </Button>
            <PromptInputSubmit status={status} onStop={handleStop} className="bg-primary text-primary-foreground hover:bg-primary/85" aria-label={status === "submitted" ? "Stop generating" : "Send message"} />
          </div>
        </PromptInput>
        <p className="mt-3 text-center text-[11px] text-zinc-600">dumbGPT can make mistakes. Keep prompts kind and curious.</p>
      </div>
    </main>
  );
}
