Payload Your App Sends

When you hit send, this is what goes to /search:

{
  "q": "your user question",
  "documents": ["doc1.pdf", "doc2.txt"],
  "llm_name": "openai",       // or "groq", "hf-bart"
  "web_enrich": false,        // enrich with web search
  "return_mode": "chat",      // "chat" or "pdf"
  "tone": "human",            // "human" or "strict"
  "draft_mode": null,         // "book_outline", "book_chapter", "long_report"
  "rerank": false,            // (optional, if backend supports)
  "citations": true           // attach source citations
}


This payload supports varying options:

llm_name → choose the model backend (openai, groq, hf-bart)

return_mode → decide whether you get a chat-style response or a generated PDF

tone → controls strict/human style of answer

draft_mode → allows long-form outputs (e.g., reports, book writing)

web_enrich → whether to fetch extra info from the web

citations → toggle citations

rerank → enable doc reranking (if you implement it later)





Build
# Build image docker build -t AIAssistant .
