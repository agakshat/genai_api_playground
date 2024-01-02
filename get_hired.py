"""
Conduct simple text conversations with recruiters based on your resume (in latex).
Usage: python get_hired.py -- <path to resume text or latex>
"""

import os
import argparse
from openai import OpenAI
from cost_util import APICostMonitor
from tutor_completion import should_end

MODEL = "gpt-3.5-turbo"

def chat(client, resume):
    CANDIDATE_INFORMATION = """
        Current location: \n
        Willing to relocate: \n
        Willing to work fully remote: \n
        Desired compensation: \n
        Visa Sponsorship: \n
        Start Date: \n
    """
    messages = [{
        "role": "system",
        "content": f"""You assist people who are looking for jobs, by taking the pre-interview
         screening calls for the candidate. This means that you will chat with the recruiter, and act as the candidate,
          and answer the recruiter's questions as well as ask them relevant questions about the role they are hiring for.
        Please make sure you do not act like the recruiter's assistant in this call, since you are the candidate's representative.
        Some information about the nature of these calls: These calls are typically held between a recruiter from the company,
          and the candidate, and is used by the recruiter to gauge prior experience and skills of the candidate,
           as well as interest and enthusiasm for the company and the role.\n
        Make sure you ask questions to find out about: the company, what it does, what the founder's backgrounds are,
        what the role they are hiring for, what ongoing projects that team is working on, what is the size of the team
        that they are hiring for, who will the manager be and what is their background, what the compensation structure
        will look like, what seniority they are hiring for.\n
        Additionally, the recruiter will give you information about the company, and the role. At the end of the conversation, 
        please summarize the information the recruiter has given you about the company, and prepend this summary with the word "Summary".
        More information about the candidate you are helping:\n\n
           {CANDIDATE_INFORMATION}\n
           Resume: {resume} \n\n
        """,
        }]
    cost_monitor = APICostMonitor(model=MODEL)
    recruiter_begin = input("Recruiter: ")
    messages.append({
        "role": "user",
        "content": recruiter_begin,
        "name": "recruiter"
    })
    while (True):
        response = client.chat.completions.create(model=MODEL, messages=messages)
        cost_monitor.update_usage(response)
        if should_end(client, response.choices[0].message.content):
            print("Candidate wants to end the conversation.")
            break
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        user_response = input("\nUser: " + response.choices[0].message.content+'\n\nRecruiter: ')
        if should_end(client, user_response):
            print("Recruiter wants to end the conversation.")
            break
        messages.append({"role": "user", "content": user_response, "name": "recruiter"})
    
    messages.append({"name": "Akshat", "role": "user", "content": "Please summarize the conversation between you and the recruiter now."})
    response = client.chat.completions.create(model=MODEL,messages=messages)
    print(f"\n\n{response.choices[0].message.content}")
    cost_monitor.print_cost()


def main():
    parser = argparse.ArgumentParser(description='Get Hired Bot')
    parser.add_argument('filepath', type=str, help='Absolute path to the resume')

    args = parser.parse_args()

    # Ensure the file path is absolute
    if not os.path.isabs(args.filepath):
        raise ValueError(f"Filepath {args.filepath} must be absolute")

    contents = ""
    # Read the resume's contents.
    with open(args.filepath, 'r') as f:
        contents += f.read()

    client = OpenAI()

    chat(client, contents)

if __name__ == "__main__":
    main()