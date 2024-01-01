import os
from collections import defaultdict
from openai import OpenAI
from cost_util import APICostMonitor

MODEL = "gpt-3.5-turbo"

def should_end(client, question):
    # Simple function which queries GPT to see if the user wants to end the conversation.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an assistant to determine if the user wants
                            to finish the conversation, based on analysis of their response.
                            Your answer should be restricted to one of the two words 'YES' or 'NO'.
                            'YES' if the user wants to end, 'NO' if they want to continue."""
            }, {
                "role": "user",
                "content": question,
            }
        ]
    )
    ans = response.choices[0].message.content
    if "YES" in ans.upper():
        return True
    elif "NO" in ans.upper():
        return False
    else:
        print(f"should_end function's output is not properly formatted. {ans}")
        return False


def chat(client):
    # ChatGPT like interface with memory.
    user_question = input("Hello, what would you like to learn about today? \n")

    cost_monitor = APICostMonitor()
    messages = [
        {
            "role": "system",
            "content": """You are an interactive tutor for students who teaches concepts by: \n
                        \t - Not answering the question directly, instead asking thought-provoking 
                        questions to the students which engages them and makes them think.\n
                        \t - Carefully understanding their response and following up with further questions
                        to direct their thinking process appropriately.\n
                        \t - Breaking down the concept into simpler, more answerable components.\n
                        \t - Understand if the student is getting frustrated with this teaching style, and
                        providing hints when needed.\n
                        \t - Keeping your responses concise."""
        }, {
            "role": "user",
            "content": user_question,
        }
    ]
    while (True):
        response = client.chat.completions.create(model=MODEL, messages=messages)
        cost_monitor.update_usage(response)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        user_response = input(response.choices[0].message.content+'\n\n Your Response: ')
        if should_end(client, user_response):
            print("It seems you want to end the conversation. Hope you learned something today.")
            break
        messages.append({"role": "user", "content": user_response})
    cost_monitor.print_cost()


def completion(client):
    #### Basic chat completion.
    cost_monitor = APICostMonitor()
    responses = {}
    SYSTEM_ROLE = {
        "kids_tutor": "You are a tutor for 5-10 year old children. Keep your answers age-appropriate, descriptive and short.",
        "teenage_tutor": "You are a tutor for teenage children. Keep your answers age-appropriate, descriptive and short.",
        "college_tutor": "You are a tutor for college students. Keep your answers age-appropriate, descriptive and short.",
    }
    for sysrole in SYSTEM_ROLE:
        response = client.chat.completions.create( # Note this would be client.completions for older models.
            # Specify which model you want to interact with, quality-cost tradeoff.
            model=MODEL,
            # Input: A list of Message objects, each object has a role ("system", "user", "assistant")
            # and some content. Typical flow is that the first object is a "system" message, followed by
            # alternating user and assistant messages.
            # Assistant messages store previous assistant responses, but can be used for prompting.
            messages=[
                {"role": "system", "content": SYSTEM_ROLE[sysrole]},
                {"role": "user", "content": "What is conservation of momentum?"},
            ],
            ## Other optional arguments.
            # frequency_penalty: [-2.,2.] - positive value penalizes new tokens based on existing frequency in the text.
            # logit_bias: map, bias the model's output by assing bias to the logits prior to sampling.
            # max_tokens: restrict output length, upper-bound by model capabilities.
            # n: how many completions, default 1.
            # stop: custom STOP sequence of chars.
            # stream: receive output in streaming mode, like in ChatGPT.
            # temperature: sampling temperature [0, 2], default 1. higher values make output more random.
            # top_p: nucleus sampling, model considers results with top_p probability mass.
            # tools: tools (like functions) the model may call.
        )
        # Parse the output. Format:
        # choices: list of chat-completions, size of n.
        #   ChatCompletions - finish_reason, index, message, logprobs
        #       message - content, role, tool_calls
        # usage: UsageStatistics, completion_tokens, prompt_tokens, total_tokens.
        responses[sysrole] = response.choices[0].message.content
        cost_monitor.update_usage(responses[sysrole])
        print(f"Role: {sysrole}, Response: \n{responses[sysrole]}")
    cost_monitor.print_cost()


def main():
    client = OpenAI()

    # completion(client)
    chat(client)
    # should_end(client, "I need to go to class now.")


if __name__ == "__main__":
    main()
