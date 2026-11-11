# GLAB 927.6.1: Mastering LangChain for Advanced Language Model Applications

An official Per Scholas assignment demonstrating automated customer support workflows, sequential multi-stage prompt chains, and prompt optimization using **LangChain** and **Groq** inside a `uv`-managed Python package.

## Core Concepts Implemented

* **LangChain Expression Language (LCEL):** Composing modular prompt templates, chat models (`ChatGroq`), and output parsers (`StrOutputParser`) into clean, runnable pipelines.
* **Basic FAQ Prompt Chain (Task 2):** Automating responses to standard customer inquiries (business hours and return policies) using system role instructions.
* **Sequential Multi-Stage Workflow (Task 3):** Chaining two distinct LLM steps together—first extracting structured order fields from raw database records, then passing that output into a second chain to draft a customer update.
* **Prompt Optimization & Guardrails (Task 4):** Grounding responses in verified company policy context, enforcing concise formatting limits, and instructing the model to escalate out-of-scope questions rather than hallucinating.

## Setup & Execution Instructions

1. **Sync project dependencies:**
	```bash
	uv sync
	```

2. **Configure environment variables:**
	Create a `.env` file in the project root and add your Groq API key:
	```env
	GROQ_API_KEY=your_groq_api_key_here
	```

3. **Run the connected package entry point:**
	```bash
	uv run glab-927-6-1
	```

## Summary of Findings & Optimization Insights

Comparing Task 2 against Task 4 demonstrated how prompt engineering directly impacts reliability in business applications. While basic prompts answer general questions well, adding explicit company policy context, tone constraints, and fallback rules in Task 4 prevented the model from guessing answers on unsupported topics (such as shipping to Mars) and kept customer replies concise and accurate.
