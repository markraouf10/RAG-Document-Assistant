import os
import json
import time
from google import genai
from dotenv import load_dotenv
from eval_dataset import QA_PAIRS
from rag import ask

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def judge(question, ground_truth, generated_answer):
    prompt = f"""You are an expert evaluator of AI-generated answers.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Score the generated answer on two criteria, each from 1 to 5:

1. Faithfulness (1-5): Is the generated answer factually consistent with the ground truth?
   1 = completely wrong or fabricated
   3 = partially correct
   5 = fully accurate and grounded

2. Relevance (1-5): Does the generated answer actually address the question asked?
   1 = completely off topic
   3 = partially addresses the question
   5 = directly and completely answers the question

Respond ONLY in this exact JSON format with no markdown:
{{"faithfulness": <score>, "relevance": <score>, "reasoning": "<one sentence explanation>"}}"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text)


def evaluate():
    results = []
    total_faithfulness = 0
    total_relevance = 0

    print(f"Evaluating {len(QA_PAIRS)} questions...\n")
    print("-" * 60)

    for i, pair in enumerate(QA_PAIRS):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        print(f"[{i+1}/{len(QA_PAIRS)}] {question}")

        # Step 1 — get RAG answer
        answer, chunks = ask(question)

        # Step 2 — judge with Gemini
        scores = judge(question, ground_truth, answer)

        faithfulness = scores["faithfulness"]
        relevance = scores["relevance"]
        reasoning = scores["reasoning"]

        total_faithfulness += faithfulness
        total_relevance += relevance

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "reasoning": reasoning,
            "sources": [f"{c['source']} p{c['page']}" for c in chunks]
        }
        results.append(result)

        print(f"  Faithfulness: {faithfulness}/5 | Relevance: {relevance}/5")
        print(f"  Reasoning: {reasoning}\n")

        # Save results after every question in case of crash
        os.makedirs("eval_logs", exist_ok=True)
        with open("eval_logs/eval_results.json", "w") as f:
            json.dump({
                "avg_faithfulness": total_faithfulness / (i + 1),
                "avg_relevance": total_relevance / (i + 1),
                "questions_evaluated": i + 1,
                "results": results
            }, f, indent=2)

        # Wait to respect Gemini free tier rate limit (5 requests per minute)
        if i < len(QA_PAIRS) - 1:
            print(f"  Waiting 13 seconds for rate limit...\n")
            time.sleep(13)

    # Final scores
    avg_faithfulness = total_faithfulness / len(QA_PAIRS)
    avg_relevance = total_relevance / len(QA_PAIRS)

    print("-" * 60)
    print(f"FINAL RESULTS")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}/5")
    print(f"Average Relevance:    {avg_relevance:.2f}/5")
    print("-" * 60)
    print(f"\nFull results saved to eval_logs/eval_results.json")


if __name__ == "__main__":
    evaluate()