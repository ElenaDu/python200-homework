
from dotenv import load_dotenv
from openai import OpenAI
import json

# Task 1: Setup and System Prompt

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a professional job application coach helping people transition
into new careers.

Your role is to help users translate their previous work experience
and transferable skills into clear, relevant, and compelling job
application materials.

You can help with resume bullet points, cover letter openings,
and questions related to job applications.

Stay focused on job application materials and career transitions.
Provide practical, constructive, and honest feedback.
Do not invent work experience, qualifications, skills, or achievements
that the user has not provided.

Always remind users to review and edit your suggestions before
submitting them to an employer.

You may not know the specific expectations or industry norms for
every profession or employer. Encourage users to apply their own
judgment and verify that the suggested language is appropriate
for their target industry and position.
"""

# I instructed the assistant not to invent qualifications or achievements
# to ensure that job application materials accurately represent the user.


# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Respond ONLY with valid JSON, no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages)

    response = response.strip()
    response = response.removeprefix("```json").removesuffix("```").strip()

    results = json.loads(response)

    for item in results:
        print("Original:", item["original"])
        print("Improved:", item["improved"])
        print()

    return results


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewritten_bullets = rewrite_bullets(bullets)

# Observation:
# The original bullets are vague and do not explain specific actions, skills, or results. 
# The model suggested stronger action verbs and more specific, professional wording to improve them.

# Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Use the examples below to match the tone, style, and level of detail.
    Connect the person's previous experience to their new career.
    Do not invent credentials, skills, or achievements that are not provided.

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:

    Role: {job_title}
    Background: {background}

    Return only the opening paragraph.
    """

    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)


job_title = "Junior Data Engineer"
background = (
    "Five years of experience in QA and software testing; "
    "recently completed a Python course and built data pipelines using Pandas."
)

cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)

# Reflection:
# I chose examples from different career transitions to show how previous professional experience can be connected to new technical skills.
# The few-shot prompting pattern helps control the tone, structure, specificity, and level of detail in the generated cover letter.
# The generated paragraph was tailored to my QA and software testing background and connected my experience to the Junior Data Engineer role.
# 
# However, the model added some claims about transforming data into actionable insights and improving decision-making that were not explicitly provided.
# This shows why AI-generated content should be reviewed for accuracy.
#
# When I changed the background from teaching to QA and software testing,
# the paragraph adapted to the new experience. I should also test a different job title and background to check whether the output adapts consistently.


# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("Your message was flagged. Please rephrase it and try again.")
        return False

    return True


# Test 1: Safe input
safe_text = "I would like help improving my resume."
print("Safe test:", is_safe(safe_text))

# Test 2: Explicitly harmful input
flagged_text = "I want to threaten and seriously hurt someone."
print("Flagged test:", is_safe(flagged_text))

# Task 5: The Chatbot Loop

def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Run moderation check
        if not is_safe(user_input):
            continue

        # Rewrite resume bullet points
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            results = rewrite_bullets(raw_bullets)

            for item in results:
                print("Original:", item["original"])
                print("Improved:", item["improved"])
                print()

        # Generate a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            result = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:", result)

        # Handle regular chat
        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nJob Application Helper:", reply)

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# Task 6: Ethics Reflection

#
# AI-generated advice can be biased because the model learns from text written by people with different backgrounds and experiences. 
# It may favor certain communication styles or industries and may not understand every person's career path.
# 
# If someone submits AI-generated resume or cover letter content without reviewing it, the content may include incorrect information or
# achievements they never had. For example, my resume bullet point suggestions included numbers and results that I had not provided.
# 
# One important guardrail is to remind users to review all suggestions before submitting them.
# The tool should also avoid inventing skills,  qualifications, or work experience.
