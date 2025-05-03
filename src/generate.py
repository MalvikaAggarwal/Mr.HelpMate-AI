import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

api_key = os.getenv("GENERATIVE_AI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please ensure it is set in the .env file.")

genai.configure(api_key=api_key)


def generate_answer(query, top_chunks):
    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    prompt = f"""
You are Mr. HelpMate AI, a smart assistant built to help people understand complex insurance policy documents. These documents are often long and full of hidden clauses that buyers usually miss.

Your job is to carefully read the provided context and answer the user’s specific question clearly and accurately.

You must:
- Provide a concise and clear answer to the question.
- Mention the section heading (e.g., "PART IV - BENEFITS, Article 2 - Death Benefits Payable") where the information was found.
- Include the page number(s) where this content appears in the policy.
- If the answer spans multiple sections, mention all relevant headings and pages.
- Use simple, professional language that helps the user understand their policy without legal jargon.

---

Context:
{context}

Question: {query}

---

Example 1:
Question: Does this policy provide coverage for death due to an accident?

Answer: Yes, the policy includes coverage for accidental death.

Found in:
Section: PART IV - BENEFITS, Section B - Member Accidental Death and Dismemberment Insurance  
Article: Article 3 - Benefits Payable  
Pages: 49–51

---

Example 2:
Question: Is there a suicide exclusion clause in this insurance policy?

Answer: Yes, the policy does not pay a benefit if the insured dies by suicide within the first two years.

Found in:
Section: PART V - GENERAL EXCLUSIONS  
Article: Article 1 - Suicide Exclusion  
Pages: 52

---

Example 3:
Question: Are pre-existing conditions covered in this plan?

Answer: No, the policy excludes coverage for pre-existing conditions during the first 12 months.

Found in:
Section: PART III - LIMITATIONS  
Article: Article 2 - Pre-existing Condition Limitation  
Pages: 35–36

---

Now, based on the given context and user question, follow the above format to respond appropriately.

"""
    model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
    response = model.generate_content(prompt)
    return response.text
