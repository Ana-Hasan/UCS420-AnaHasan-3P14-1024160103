
import pandas as pd
import re

# ============================================================
# Q1. BUILD YOUR PERSONALIZED KNOWLEDGE BASE
# ============================================================

# Replace this with your actual college roll number
roll_number = input("Enter your college roll number: ").strip()

# Extract digits from roll number
digits = re.sub(r"\D", "", roll_number)

if len(digits) < 2:
    raise ValueError("Roll number must contain at least two digits.")

# Take the last two digits
last_two = digits[-2:]
digit2 = int(last_two[0])
digit3 = int(last_two[1])

# Category mapping
categories = ["billing", "account", "general"]

category2 = categories[digit2 % 3]
category3 = categories[digit3 % 3]

# Six FAQ entries: four fixed and two personalized
fixed_entries = [
    {
        "question": "what is the annual fee",
        "answer": "The annual fee is Rs 500.",
        "keywords": ["fee", "price", "charge"],
        "category": "billing"
    },
    {
        "question": "how to reset password",
        "answer": "Go to Reset Password and follow the instructions.",
        "keywords": ["password", "reset", "login"],
        "category": "account"
    },
    {
        "question": "what are your working hours",
        "answer": "We are open 9 AM to 5 PM.",
        "keywords": ["hours", "timing", "open", "close"],
        "category": "general"
    },
    {
        "question": "how can i pay the fee",
        "answer": "You can pay via UPI, card, or net banking.",
        "keywords": ["pay", "payment", "upi", "fee"],
        "category": "billing"
    }
]

personalized_entries = [
    {
        "question": "how do I update my registered mobile number",
        "answer": "Visit your account settings to update your registered mobile number.",
        "keywords": ["mobile", "number", "update", "phone"],
        "category": category2
    },
    {
        "question": "how do I check my account details",
        "answer": "Log in to your account to view your account details.",
        "keywords": ["account", "details", "profile", "check"],
        "category": category3
    }
]

# Combine all entries into a DataFrame
all_entries = fixed_entries + personalized_entries

df = pd.DataFrame(all_entries)

print("\n========== Q1: PERSONALIZED KNOWLEDGE BASE ==========")
print(df.to_string(index=False))


# ============================================================
# Q2. GENERATE AND SCORE A HYPOTHESIS
# ============================================================

def score_query(query, knowledge_base=df):
    """
    Match query words against FAQ keywords.
    Return all matching entries ranked by score.
    """
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    results = []

    for index, row in knowledge_base.iterrows():
        keywords = row["keywords"]

        # Count how many distinct FAQ keywords occur in query
        matched_keywords = sorted(query_words.intersection(keywords))
        score = len(matched_keywords)

        if score > 0:
            results.append({
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "score": score,
                "matched_keywords": matched_keywords
            })

    # Return every match, including entries tied for the highest score
    results.sort(key=lambda item: item["score"], reverse=True)

    return results


print("\n========== Q2: QUERY SCORING ==========")

query = input("\nEnter your FAQ query: ")
results = score_query(query)

if results:
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Category: {result['category']}")
        print(f"Score: {result['score']}")
        print(f"Matched keywords: {result['matched_keywords']}")
else:
    print("No matching FAQ found.")


# ============================================================
# Q3. SAME_CATEGORY FUNCTION
# ============================================================

def same_category(category_name, knowledge_base=df):
    """
    Return all questions belonging to a given category.
    """
    matching_rows = knowledge_base[
        knowledge_base["category"].str.lower()
        == category_name.lower()
    ]

    return matching_rows["question"].tolist()


print("\n========== Q3: CATEGORY FILTERING ==========")

category_input = input(
    "Enter category (billing/account/general): "
).strip()

category_questions = same_category(category_input)

if category_questions:
    print(f"\nQuestions in category '{category_input}':")
    for question in category_questions:
        print("-", question)
else:
    print("No questions found in this category.")


# ============================================================
# Q4. ADD A NEW KEYWORD AND SAVE UPDATED KNOWLEDGE BASE
# ============================================================

print("\n========== Q4: UPDATE KNOWLEDGE BASE ==========")

print("\nAvailable FAQ entries:")
for i, question in enumerate(df["question"], start=1):
    print(f"{i}. {question}")

try:
    entry_number = int(input("Select an FAQ entry (1-6): "))

    if not 1 <= entry_number <= len(df):
        raise ValueError("Please select a number from 1 to 6.")

    selected_index = entry_number - 1

    new_keyword = input("Enter the new keyword: ").strip().lower()

    if not new_keyword:
        print("Empty keyword not added.")
    elif new_keyword not in df.at[selected_index, "keywords"]:
        df.at[selected_index, "keywords"].append(new_keyword)
        print("Keyword added successfully.")
    else:
        print("Keyword already exists.")

    # Save the updated DataFrame as CSV
    filename = f"{roll_number}_faq_data.csv"
    df.to_csv(filename, index=False)

    print(f"Updated knowledge base saved to {filename}")

except ValueError as error:
    print("Invalid input:", error)


# ============================================================
# Q5. GROUPBY AND PRINT FAQ COUNT PER CATEGORY
# ============================================================

print("\n========== Q5: FAQ COUNT PER CATEGORY ==========")

category_counts = df.groupby("category").size()

print(category_counts)


# ============================================================
# Q6. HANDLE TIES IN QUERY SCORING
# ============================================================

def score_query_with_ties(query, knowledge_base=df):
    """
    Return all entries tied for the highest positive score.
    Never select only one entry when multiple entries tie.
    """
    results = score_query(query, knowledge_base)

    if not results:
        return []

    highest_score = results[0]["score"]

    return [
        result for result in results
        if result["score"] == highest_score
    ]


print("\n========== Q6: TIE-HANDLING DEMONSTRATION ==========")

# Test 1: Query matches the keyword 'fee' in two fixed FAQs
test_query_1 = "fee"
print(f"\nTest query: '{test_query_1}'")

tied_results = score_query_with_ties(test_query_1)

if tied_results:
    print("All highest-scoring matching entries:")
    for result in tied_results:
        print(
            f"Question: {result['question']} | "
            f"Score: {result['score']}"
        )
else:
    print("No matching FAQ found.")

# Test 2: Query that should not match any FAQ keyword
test_query_2 = "zebra"
print(f"\nTest query: '{test_query_2}'")

tied_results = score_query_with_ties(test_query_2)

if tied_results:
    for result in tied_results:
        print(result)
else:
    print("No matching FAQ found.")

print("\n========== ALL SIX QUESTIONS COMPLETED ==========")
