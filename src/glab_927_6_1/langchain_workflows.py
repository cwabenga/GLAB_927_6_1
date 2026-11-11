"""
=============================================================================
GLAB 927.6.1: MASTERING LANGCHAIN FOR ADVANCED LANGUAGE MODEL APPLICATIONS
=============================================================================
Demonstrates:
- Task 2: Basic Prompt Chain for Customer Support FAQs
- Task 3: Multi-Stage Sequential Workflow (Data Extraction + Response Generation)
- Task 4: Prompt Optimization & Workflow Evaluation
=============================================================================
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load the GROQ_API_KEY safely from the local .env file.
load_dotenv()


def get_llm(temperature: float = 0.2) -> ChatGroq:
	"""Initialize and return the Groq Chat LLM instance."""
	if not os.getenv("GROQ_API_KEY"):
		raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

	return ChatGroq(model="openai/gpt-oss-20b", temperature=temperature)


def task_2_basic_faq_chain(llm: ChatGroq) -> None:
	"""Configure LangChain with a basic prompt chain for FAQs."""
	print("=================================================================")
	print("TASK 2: BASIC CUSTOMER SUPPORT FAQ CHAIN")
	print("=================================================================")

	faq_prompt = ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"You are a helpful customer support assistant. Answer common questions clearly.",
			),
			("human", "{customer_query}"),
		]
	)
	faq_chain = faq_prompt | llm | StrOutputParser()

	sample_queries = [
		"What are your standard business hours?",
		"What is your return policy for electronics?",
	]

	for query in sample_queries:
		print(f"\nCustomer Query: {query}")
		response = faq_chain.invoke({"customer_query": query})
		print(f"AI Response   : {response}")


def task_3_sequential_workflow(llm: ChatGroq) -> None:
	"""Build a multi-stage sequential workflow for complex queries."""
	print("\n=================================================================")
	print("TASK 3: MULTI-STAGE SEQUENTIAL PROMPT CHAIN")
	print("=================================================================")

	mock_order_database = (
		"Order #4092 | Customer: Alex Rivera | Item: Wireless Noise-Canceling Headphones | "
		"Status: Delayed in Transit | Expected Delivery: Friday by 5:00 PM | "
		"Carrier: FastShip Express"
	)

	extraction_prompt = ChatPromptTemplate.from_template(
		"Extract the order number, item name, current status, and new delivery date "
		"from the following database record:\n\n{database_record}"
	)
	extraction_chain = extraction_prompt | llm | StrOutputParser()

	response_prompt = ChatPromptTemplate.from_template(
		"Using these extracted order details, write a brief, polite customer support "
		"update answering the customer's question.\n\n"
		"Extracted Details:\n{extracted_info}\n\n"
		"Customer Question: {customer_question}"
	)
	response_chain = response_prompt | llm | StrOutputParser()

	customer_question = (
		"Hi, I ordered headphones (Order #4092) and they haven't arrived yet. Where are they?"
	)
	print(f"Customer Question: {customer_question}\n")

	extracted_data = extraction_chain.invoke({"database_record": mock_order_database})
	print(f"[Stage 1 - Extracted Data]:\n{extracted_data}\n")

	final_response = response_chain.invoke(
		{
			"extracted_info": extracted_data,
			"customer_question": customer_question,
		}
	)
	print(f"[Stage 2 - Generated Customer Response]:\n{final_response}")


def task_4_optimized_workflow(llm: ChatGroq) -> None:
	"""Optimize prompts with explicit context, guardrails, and formatting."""
	print("\n=================================================================")
	print("TASK 4: OPTIMIZED & EVALUATED WORKFLOW")
	print("=================================================================")

	company_policy_context = (
		"Company Name: TechNova Retail. "
		"Business Hours: Monday-Friday, 9:00 AM to 6:00 PM CST. Closed weekends. "
		"Return Policy: 30-day full refund with original receipt; 15% restocking fee if opened. "
		"Shipping SLA: Standard shipping takes 3-5 business days."
	)

	optimized_prompt = ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"You are a senior customer support specialist at TechNova Retail. "
				"Answer the customer's question using ONLY the verified company policies provided below. "
				"If the answer is not in the policy, politely offer to escalate to a human agent. "
				"Keep responses concise (under 3 sentences) and friendly.\n\n"
				"Verified Policies:\n{policy_context}",
			),
			("human", "{customer_query}"),
		]
	)
	optimized_chain = optimized_prompt | llm | StrOutputParser()

	test_queries = [
		"Are you open on Saturdays?",
		"Can I return a laptop I opened two weeks ago?",
		"Do you offer international shipping to Mars?",
	]

	for query in test_queries:
		print(f"\nTest Query        : {query}")
		response = optimized_chain.invoke(
			{
				"policy_context": company_policy_context,
				"customer_query": query,
			}
		)
		print(f"Optimized Response: {response}")
	print("=================================================================")


def run_all_tasks() -> None:
	"""Run all LangChain lab workflows."""
	llm = get_llm()
	task_2_basic_faq_chain(llm)
	task_3_sequential_workflow(llm)
	task_4_optimized_workflow(llm)


if __name__ == "__main__":
	run_all_tasks()
