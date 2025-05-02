import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAYIHIziCeg8JqR7Qbbu6nDx_gyLlx-kzA")
def generate_answer(query, top_chunks):
    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    prompt = f"""
You are an insurance assistant. Answer the question using ONLY the context below.

Context:
{context}

Question: {query}

Respond in structured JSON:
{{
  "answer": "<short answer here>",
  "chunks_used": [<chunk1>, <chunk2>, <chunk3>]
}}
"""
    model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
    response = model.generate_content(prompt)
    return response.text
