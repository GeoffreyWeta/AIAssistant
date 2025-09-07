import { useEffect, useRef, useState } from "react";
import { Upload, Send, Trash2, X, Bot, User, ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";

type LLMName = "openai" | "groq" | "hf-bart";
type Tone = "human" | "strict";
type DraftMode = null | "book_outline" | "book_chapter" | "long_report";
type ReturnMode = "chat" | "pdf";

interface Citation {
  filename: string;
  page?: number;
  snippet?: string;
  doc_id?: string;
  score?: number;
}

interface WebSource {
  title?: string;
  url?: string;
}

interface WebImage {
  title?: string;
  url?: string;
  image_url: string;
  thumbnail?: string;
  width?: number;
  height?: number;
  score?: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  attachedDocs?: string;
  loading?: boolean;
  citations?: Citation[];
  web_sources?: WebSource[];
  web_images?: WebImage[];
  mode?: "chat" | "pdf";
  report_url?: string;
  latency_ms?: number;
}

type BackendDocument = { id: string; filename: string };

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function hostFromUrl(u?: string) {
  try {
    if (!u) return "";
    const h = new URL(u).hostname;
    return h.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function TypingBubble() {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-800 rounded-2xl shadow-sm">
      <span className="sr-only">AI is typing</span>
      <div className="flex gap-1">
        <span className="dot" />
        <span className="dot dot-2" />
        <span className="dot dot-3" />
      </div>
    </div>
  );
}

function UploadToast({ text = "Uploading" }: { text?: string }) {
  return (
    <div className="fixed bottom-4 right-4 z-40 w-64 rounded-2xl border border-gray-200 bg-white shadow-lg p-3">
      <div className="flex items-center gap-3">
        <div className="eq" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className="text-sm">
          <div className="font-medium">{text}</div>
          <div className="mt-1 h-1.5 w-full bg-gray-100 rounded overflow-hidden">
            <div className="shimmer h-full w-1/2" />
          </div>
        </div>
      </div>
    </div>
  );
}

// Lightbox viewer
function ImageLightbox({
  images,
  index,
  onClose,
  onPrev,
  onNext,
}: {
  images: WebImage[];
  index: number;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}) {
  const img = images[index];

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") onPrev();
      if (e.key === "ArrowRight") onNext();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onPrev, onNext]);

  if (!img) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <button
        className="absolute top-4 right-4 h-9 w-9 rounded-full bg-white/90 hover:bg-white grid place-items-center shadow"
        onClick={onClose}
        aria-label="Close"
        title="Close"
      >
        <X size={18} />
      </button>

      <button
        className="absolute left-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-white/90 hover:bg-white grid place-items-center shadow"
        onClick={onPrev}
        aria-label="Previous image"
        title="Previous"
      >
        <ChevronLeft size={18} />
      </button>

      <figure className="max-w-[92vw] max-h-[82vh] text-center">
        <img
          src={img.image_url}
          alt={img.title || "image"}
          className="mx-auto max-h-[70vh] max-w-[92vw] object-contain rounded-lg shadow-lg"
        />
        <figcaption className="mt-3 text-sm text-white/90">
          {img.title || "image"}{" "}
          {img.url ? (
            <>
              <span className="opacity-75">from</span>{" "}
              <a
                href={img.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 underline underline-offset-2"
                title="Open source"
              >
                {hostFromUrl(img.url)}
                <ExternalLink size={14} />
              </a>
            </>
          ) : null}
        </figcaption>
      </figure>

      <button
        className="absolute right-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-white/90 hover:bg-white grid place-items-center shadow"
        onClick={onNext}
        aria-label="Next image"
        title="Next"
      >
        <ChevronRight size={18} />
      </button>
    </div>
  );
}

export default function App() {
  // chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");

  // documents
  const [documents, setDocuments] = useState<BackendDocument[]>([]);
  const [attachedDocs, setAttachedDocs] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // options mapping to backend
  const [model, setModel] = useState<LLMName>("openai");
  const [returnMode, setReturnMode] = useState<ReturnMode>("chat");
  const [webEnrich, setWebEnrich] = useState(false);
  const [tone, setTone] = useState<Tone>("human");
  const [draftMode, setDraftMode] = useState<DraftMode>(null);
  const [rerank, setRerank] = useState<boolean>(false);
  const [citationsOn, setCitationsOn] = useState<boolean>(true);

  // image lightbox state
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxImages, setLightboxImages] = useState<WebImage[]>([]);
  const [lightboxIndex, setLightboxIndex] = useState(0);

  // ui
  const [showDocs, setShowDocs] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatAttachRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/documents`);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      setDocuments(Array.isArray(data.documents) ? data.documents : []);
    } catch {
      setError("Could not load documents");
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const valid = Array.from(files).every((f) => /\.(pdf|docx|txt)$/i.test(f.name));
    if (!valid) {
      alert("Only PDF, DOCX, or TXT files are allowed.");
      return;
    }

    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));

    try {
      setIsUploading(true);
      const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      if (data.error) {
        alert(`Upload error: ${data.error}`);
      } else {
        await fetchDocuments();
      }
    } catch {
      alert("Upload failed");
    } finally {
      setIsUploading(false);
      if (e.target) e.target.value = "";
    }
  };

  const removeDocument = async (filename: string) => {
    if (!confirm(`Delete "${filename}" from the server?`)) return;
    try {
      const res = await fetch(`${BASE_URL}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(String(res.status));
      await fetchDocuments();
      setAttachedDocs((prev) => prev.filter((n) => n !== filename));
    } catch {
      alert("Delete failed");
    }
  };

  const attachToChat = (fileName: string) => {
    if (attachedDocs.includes(fileName)) return;
    if (attachedDocs.length >= 3) {
      alert("Attach up to 3 documents only.");
      return;
    }
    setAttachedDocs((prev) => [...prev, fileName]);
  };

  const sendMessage = async () => {
    const activeDocs = attachedDocs;
    if (activeDocs.length === 0) {
      alert("Attach at least one document first.");
      return;
    }
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input, attachedDocs: activeDocs.join(", ") };
    setMessages((prev) => [...prev, userMessage, { role: "assistant", content: "", loading: true }]);

    const payload = {
      q: input,
      documents: activeDocs,
      llm_name: model,
      web_enrich: webEnrich,
      return_mode: returnMode,
      tone: tone,
      draft_mode: draftMode,
      rerank: rerank,
      citations: citationsOn,
    };

    try {
      const res = await fetch(`${BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();

      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer || (data.mode === "pdf" ? "report ready" : "no response"),
        loading: false,
        citations: data.citations,
        web_sources: data.web_sources,
        web_images: data.web_images,
        mode: data.mode,
        report_url: data.url ? `${BASE_URL}${data.url}` : undefined,
        latency_ms: data.latency_ms,
      };

      setMessages((prev) => prev.filter((m) => !m.loading).concat(assistantMsg));
      setInput("");
    } catch {
      setMessages((prev) =>
        prev.filter((m) => !m.loading).concat({ role: "assistant", content: "request failed", loading: false })
      );
    }
  };

  // open viewer for a message images array
  function openLightbox(images: WebImage[], startIndex: number) {
    setLightboxImages(images);
    setLightboxIndex(startIndex);
    setLightboxOpen(true);
  }
  function closeLightbox() {
    setLightboxOpen(false);
  }
  function prevLightbox() {
    setLightboxIndex((i) => (i - 1 + lightboxImages.length) % lightboxImages.length);
  }
  function nextLightbox() {
    setLightboxIndex((i) => (i + 1) % lightboxImages.length);
  }

  return (
    <div className="h-screen w-screen bg-neutral-50 text-gray-900 flex flex-col">
      <style>{`
        .dot { width: 6px; height: 6px; background: #111827; display: inline-block; border-radius: 9999px; animation: bounceDot 1.2s infinite; }
        .dot-2 { animation-delay: .15s; }
        .dot-3 { animation-delay: .3s; }
        @keyframes bounceDot { 0% { transform: translateY(0); opacity: .6; } 30% { transform: translateY(-4px); opacity: 1; } 60% { transform: translateY(0); opacity: .6; } 100% { transform: translateY(0); opacity: .6; } }
        .eq { display: inline-flex; gap: 3px; height: 22px; align-items: flex-end; }
        .eq span { width: 3px; background: #10b981; display: inline-block; border-radius: 2px; animation: eqBar 1s infinite ease-in-out; }
        .eq span:nth-child(1) { height: 6px; animation-delay: 0s; }
        .eq span:nth-child(2) { height: 10px; animation-delay: .1s; }
        .eq span:nth-child(3) { height: 16px; animation-delay: .2s; }
        .eq span:nth-child(4) { height: 10px; animation-delay: .3s; }
        .eq span:nth-child(5) { height: 6px; animation-delay: .4s; }
        @keyframes eqBar { 0%, 100% { transform: scaleY(.6); } 50% { transform: scaleY(1.4); } }
        .shimmer { background: linear-gradient(90deg, rgba(16,185,129,.0) 0%, rgba(16,185,129,.35) 50%, rgba(16,185,129,.0) 100%); animation: shimmer 1.5s infinite; }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
      `}</style>

      {/* header */}
      <header className="border-b border-neutral-200 px-4 py-3 flex items-center justify-between bg-white">
        <div className="font-medium flex items-center gap-2">
          <Bot size={18} />
          AI agent
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDocs((v) => !v)}
            className="text-sm px-3 py-1 border border-gray-300 rounded bg-white"
            aria-expanded={showDocs}
          >
            documents
          </button>
          <button
            onClick={() => setShowOptions((v) => !v)}
            className="text-sm px-3 py-1 border border-gray-300 rounded bg-white"
            aria-expanded={showOptions}
          >
            options
          </button>
        </div>
      </header>

      {/* documents drawer */}
      {showDocs && (
        <section className="border-b border-neutral-200 px-4 py-3 bg-white">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium">library</div>
            <button onClick={() => fileInputRef.current?.click()} className="text-sm inline-flex items-center gap-2">
              <Upload size={14} />
              {isUploading ? "uploading" : "upload"}
            </button>
            <input ref={fileInputRef} type="file" multiple accept=".pdf,.docx,.txt" onChange={handleUpload} hidden />
          </div>

          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
            {loadingDocs ? (
              <div className="text-sm text-gray-500">loading</div>
            ) : error ? (
              <div className="text-sm text-red-600">{error}</div>
            ) : documents.length === 0 ? (
              <div className="text-sm text-gray-500">no documents yet</div>
            ) : (
              documents.map((d) => (
                <div key={d.id} className="border border-gray-200 rounded-xl px-3 py-2 flex items-center justify-between">
                  <span className="truncate pr-2" title={d.filename}>
                    {d.filename}
                  </span>
                  <div className="flex items-center gap-2">
                    <button className="text-xs underline" onClick={() => attachToChat(d.filename)} title="attach to chat">
                      attach
                    </button>
                    <button
                      className="text-gray-500 hover:text-red-600"
                      onClick={() => removeDocument(d.filename)}
                      title="delete"
                      aria-label={`delete ${d.filename}`}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {attachedDocs.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {attachedDocs.map((name, i) => (
                <span key={`${name}-${i}`} className="text-xs border px-2 py-1 rounded-xl inline-flex items-center gap-1">
                  {name}
                  <button
                    className="text-gray-500 hover:text-gray-800"
                    onClick={() => setAttachedDocs((prev) => prev.filter((_, idx) => idx !== i))}
                    title="remove"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          )}
        </section>
      )}

      {/* options drawer */}
      {showOptions && (
        <section className="border-b border-neutral-200 px-4 py-3 bg-white">
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as LLMName)}
              className="border border-gray-300 rounded-xl px-2 py-1 text-sm bg-white"
              title="model"
            >
              <option value="openai">openai</option>
              <option value="groq">groq</option>
              <option value="hf-bart">hf-bart</option>
            </select>

            <select
              value={returnMode}
              onChange={(e) => setReturnMode(e.target.value as ReturnMode)}
              className="border border-gray-300 rounded-xl px-2 py-1 text-sm bg-white"
              title="return mode"
            >
              <option value="chat">chat</option>
              <option value="pdf">pdf</option>
            </select>

            <select
              value={tone}
              onChange={(e) => setTone(e.target.value as Tone)}
              className="border border-gray-300 rounded-xl px-2 py-1 text-sm bg-white"
              title="tone"
            >
              <option value="human">human</option>
              <option value="strict">strict</option>
            </select>

            <select
              value={draftMode === null ? "null" : draftMode}
              onChange={(e) =>
                setDraftMode(e.target.value === "null" ? null : (e.target.value as Exclude<DraftMode, null>))
              }
              className="border border-gray-300 rounded-xl px-2 py-1 text-sm bg-white"
              title="draft mode"
            >
              <option value="null">no draft mode</option>
              <option value="book_outline">book_outline</option>
              <option value="book_chapter">book_chapter</option>
              <option value="long_report">long_report</option>
            </select>

            <label className="text-sm inline-flex items-center gap-2 select-none">
              <input type="checkbox" checked={webEnrich} onChange={(e) => setWebEnrich(e.target.checked)} />
              web enrich
            </label>

            <label className="text-sm inline-flex items-center gap-2 select-none">
              <input type="checkbox" checked={citationsOn} onChange={(e) => setCitationsOn(e.target.checked)} />
              citations
            </label>

            {/* keep rerank off in UI if backend does not use it
            <label className="text-sm inline-flex items-center gap-2 select-none">
              <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
              rerank
            </label> */}
          </div>
        </section>
      )}

      {/* conversation */}
      <main className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="h-full grid place-items-center">
            <div className="text-center">
              <div className="text-sm text-gray-500">upload documents, attach up to 3, ask your question</div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((m, i) => {
              const isUser = m.role === "user";
              return (
                <div key={i} className={cx("flex gap-2", isUser ? "justify-end" : "justify-start")}>
                  {!isUser && (
                    <div className="flex-shrink-0 self-end">
                      <div className="h-8 w-8 rounded-full bg-gray-900 text-white grid place-items-center">
                        <Bot size={16} />
                      </div>
                    </div>
                  )}

                  <div
                    className={cx(
                      "max-w-[80%] rounded-2xl px-4 py-3 shadow-sm",
                      isUser ? "bg-gray-900 text-white rounded-br-sm" : "bg-white text-gray-900 border border-gray-200 rounded-bl-sm"
                    )}
                  >
                    <div className="text-[11px] mb-1 opacity-70">{isUser ? "you" : "ai"}</div>

                    {m.loading ? (
                      <TypingBubble />
                    ) : (
                      <>
                        <div className="text-sm whitespace-pre-wrap">{m.content}</div>

                        {m.mode === "pdf" && m.report_url && (
                          <div className="mt-2">
                            <a
                              href={m.report_url}
                              target="_blank"
                              rel="noreferrer"
                              className={cx(
                                "text-sm underline",
                                isUser ? "text-emerald-300 hover:text-emerald-200" : "text-emerald-600 hover:text-emerald-500"
                              )}
                            >
                              open generated pdf
                            </a>
                          </div>
                        )}

                        {m.attachedDocs && (
                          <div className="text-[11px] opacity-70 mt-2">attached: {m.attachedDocs}</div>
                        )}

                        {m.web_images && m.web_images.length > 0 && (
                          <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2">
                            {m.web_images.map((img, idx) => (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => openLightbox(m.web_images || [], idx)}
                                className="group block overflow-hidden rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                                title={img.title || "image"}
                              >
                                <img
                                  src={img.thumbnail || img.image_url}
                                  alt={img.title || "image"}
                                  loading="lazy"
                                  className="w-full h-28 object-cover transition-transform duration-200 group-hover:scale-105"
                                />
                                <div className="px-2 py-1 text-[10px] text-gray-600 bg-white/80 truncate">
                                  {hostFromUrl(img.url) || "source"}
                                </div>
                              </button>
                            ))}
                          </div>
                        )}

                        {m.citations && m.citations.length > 0 && (
                          <details className="mt-3">
                            <summary className="text-xs font-medium cursor-pointer select-none">citations</summary>
                            <ul className="text-xs space-y-1 list-disc pl-4 mt-1">
                              {m.citations.map((c, idx) => (
                                <li key={idx}>
                                  <span className="font-medium">{c.filename}</span>
                                  {typeof c.page === "number" ? `, page ${c.page}` : ""}
                                  {c.snippet ? `: "${c.snippet}"` : ""}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}

                        {m.web_sources && m.web_sources.length > 0 && (
                          <details className="mt-2">
                            <summary className="text-xs font-medium cursor-pointer select-one">web sources</summary>
                            <ul className="text-xs space-y-1 list-disc pl-4 mt-1">
                              {m.web_sources.map((s, idx) => (
                                <li key={idx}>
                                  {s.title ? `${s.title}: ` : ""}
                                  {s.url ? (
                                    <a
                                      className={cx("underline break-all", isUser ? "text-emerald-200" : "text-emerald-600")}
                                      href={s.url}
                                      target="_blank"
                                      rel="noreferrer"
                                    >
                                      {s.url}
                                    </a>
                                  ) : null}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}

                        {typeof m.latency_ms === "number" && (
                          <div className="mt-2 text-[10px] opacity-60">latency {m.latency_ms} ms</div>
                        )}
                      </>
                    )}
                  </div>

                  {isUser && (
                    <div className="flex-shrink-0 self-end">
                      <div className="h-8 w-8 rounded-full bg-gray-200 text-gray-700 grid place-items-center">
                        <User size={16} />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* composer */}
      <footer className="border-t border-neutral-200 p-3 bg-white">
        <div className="max-w-3xl mx-auto flex flex-col gap-2">
          {attachedDocs.length > 0 && (
            <div className="text-xs text-gray-600">using documents: {attachedDocs.join(", ")}</div>
          )}
          <div className="flex items-end gap-2">
            <button
              onClick={() => chatAttachRef.current?.click()}
              className="text-sm px-3 py-2 border border-gray-300 rounded-xl bg-white"
              title="upload then attach from drawer"
            >
              upload
            </button>
            <input ref={chatAttachRef} type="file" multiple accept=".pdf,.docx,.txt" onChange={handleUpload} hidden />

            <div className="flex-1">
              <input
                className="w-full px-3 py-3 border border-gray-300 rounded-2xl text-sm outline-none"
                placeholder="message ai"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              />
            </div>

            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="inline-flex items-center justify-center h-[42px] w-[42px] rounded-full bg-gray-900 text-white disabled:opacity-50"
              title="send"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </footer>

      {isUploading && <UploadToast text="Uploading" />}

      {lightboxOpen && (
        <ImageLightbox
          images={lightboxImages}
          index={lightboxIndex}
          onClose={closeLightbox}
          onPrev={prevLightbox}
          onNext={nextLightbox}
        />
      )}
    </div>
  );
}
