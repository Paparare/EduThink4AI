import os
from openai import OpenAI
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# Initialize

client = OpenAI()  # reads OPENAI_API_KEY from the environment

def get_all_synset_details(word_list):
    synsets_dict = {}
    for word in word_list:
        synsets_dict[word] = wn.synsets(word)
    return synsets_dict

##processor
def vocab_fetch_processor(user_input, topic):
    system_message =(f'''The topic of the user's request:{topic}.
    User now needs more vocabulary to start up with his writing, lookup on related vocabulary according to user request and the specific topic  
    If user ask for exact word, extract the exact word that user asked for.
    ONLY output individual tokens, nothing else.''')

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    classification_text = completion.choices[0].message.content.strip().lower()
    vocab_list = classification_text.split('\n')
    print(vocab_list)
    return vocab_list
##processor
def wordnet_interpreter_processor(word):
    wordnet_info = get_all_synset_details(word)
    system_message = (f'''Read user input. And provide explanation to each word in the list {word}. If there is no words, just output 'None'.''')
    try:
        message = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": (f"{wordnet_info}")},
                      {"role": "system", "content": system_message}])
        return message.content[0].text
    except Exception as e:
        return f"An error occurred with Claude: {e}"

##generator
def vocab_chat_with_model_generator(user_input,cus_prompt, word_list, word_info):

    system_message = (f"You are a teacher to support learning vocabulary related to user's writing. Please see the user input. Here is the a list of extracted vocabularies: {word_list}"
                      "Generate the definition of each word on the word list"
                      f"If these words have affix, paying particular attention to affixes (prefixes and suffixes) and roots, and integrate the information about the vocabulary from wordnet: {word_info}"
                      f"If the words not have affix, use information from wordnet: {word_info}"
                      "Explain how understanding these components can help in deciphering the meanings of unfamiliar words. "
                      "Provide examples for each word to demonstrate how affixes and roots alter the meaning of base words. "
                      "Encourage the user to create sentences with the new vocabulary to reinforce their learning. "
                      "Aim to make this an engaging and informative experience that promotes the user's vocabulary expansion in certain writing topics and further helps user to start their writing according the provided information of words.")

    if cus_prompt is None:
        system_message = system_message
    else:
        cus_prompt = str(cus_prompt)
        system_message = 'User information:' + cus_prompt + 'Task for you:' + system_message
    try:
        messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        vocab_info = completion.choices[0].message.content.strip().lower()
        return vocab_info
    except Exception as e:
        return f"An error occurred with Gpt-4o: {e}"
