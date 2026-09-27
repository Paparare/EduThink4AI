import os
import streamlit as st
from dataclasses import dataclass
from typing import Union

from bots import reasoning, TBLT, vocab, image


@dataclass
class Message:
    actor: str
    payload: Union[str, int, bytes]


def redirect_to_google():
    """End-of-session notice. The study's survey links have been removed from the public release."""
    st.markdown("Thank you for using EduThink4AI!")


# Optional access control for deployments: set APP_USERS (comma-separated) and/or
# APP_PASSWORD in the environment. When neither is set, any non-empty username works.
ALLOWED_USERS = {u.strip() for u in os.environ.get("APP_USERS", "").split(",") if u.strip()}
APP_PASSWORD = os.environ.get("APP_PASSWORD")

if 'login' not in st.session_state or st.session_state['login'] != True:
    st.title("EduThink4AI - Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button('Login'):
        if (username
                and (not ALLOWED_USERS or username in ALLOWED_USERS)
                and (APP_PASSWORD is None or password == APP_PASSWORD)):
            st.session_state['username'] = username
            st.session_state['login'] = True

            # per-user temporary directory for uploaded images
            data_base = "./temp"
            path = os.path.join(data_base, username)
            if os.path.exists(path):
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
            else:
                os.makedirs(path)
            st.session_state['path'] = path

            st.rerun()
        else:
            st.error("Login failed.")


if 'login' in st.session_state and st.session_state['login'] == True:
    st.title('EduThink4AI')
    with st.sidebar:
        st.header('This application is currently implemented with:')
        st.write('- Writing Assistance ✍️')
        st.write('- Vocabulary Building 📚')
        st.write('- Writing Assessment/feedback 📝')
        st.write('------------------------------')
        uploaded_file = st.file_uploader("Upload only", type=['png', 'jpg', 'jpeg'],
                                         key="file_uploader")

    user_input = st.chat_input('Enter `exit` to end this conversation.')
    st.info("If you are stuck or want to start a new conversation, please try to refresh the page and log in again.")

    if 'reset_uploader' not in st.session_state:
        st.session_state['reset_uploader'] = False

    if 'message' not in st.session_state:
        st.session_state['stage'] = 0  # 0
        st.session_state['model_type'] = 'gpt-4'  # 0
        st.session_state['history'] = []
        st.session_state['message'] = [Message(actor='ai', payload='Welcome to the TBLT writing class! Please enter your personal information and intends.😊')]
        st.session_state['cus_prompt'] = ''
        st.session_state['input_history'] = []
        st.session_state["upload"] = False
        st.session_state['step'] = 0

        for message in st.session_state['message']:
            if type(message.payload) == str:
                st.chat_message(message.actor).write(message.payload)
            else:
                st.chat_message(message.actor).audio(message.payload)

    if user_input:
        for message in st.session_state['message']:
                    if type(message.payload) == str:
                        st.chat_message(message.actor).write(message.payload)
                
        if st.session_state['step'] == 0:
            st.chat_message('user').write(user_input)
            st.session_state['message'].append(Message(actor='user', payload=user_input))
            cus_prompt = reasoning.cus_prompt_generator(user_input)
            with st.expander("Show Your Customized Prompt"):
                st.write(f'{cus_prompt}')
            st.session_state['cus_prompt'] = cus_prompt
            ai_message = Message(actor='ai', payload='Please enter your request or writing😊')
            st.chat_message('ai').write(ai_message.payload)
            st.session_state['message'].append(ai_message)
            st.session_state['step'] = 1
            
        


        elif user_input.lower() == 'exit':
            st.session_state['stage'] = 0  # 0
            st.session_state['model_type'] = 'gpt-4'  # 0
            st.session_state['history'] = []
            st.session_state['cus_prompt'] = ''
            st.session_state['input_history'] = []
            st.session_state["upload"] = False
            bot_response = ('All records have been deleted. If you need anything else, please let me know! 😊')
            st.balloons()
            redirect_to_google()





        elif st.session_state['step'] == 1 :
            st.chat_message('user').write(user_input)
            st.session_state['message'].append(Message(actor='user', payload=user_input))
            cus_prompt = st.session_state['cus_prompt']
            with st.expander("Show Your Customized Prompt"):
                st.write(f'{cus_prompt}')
            stage = TBLT.stage_classification(user_input)
            st.session_state['stage'] = stage
            reason = reasoning.reasoning_check(user_input)
            topic = TBLT.topic_classify(user_input)


            if stage == 0 and reason == 1:
                generation_1 = reasoning.f1(user_input)
                generation_2 = reasoning.f2(user_input, generation_1)
                c = reasoning.f3(user_input, generation_1, generation_2)
                v = reasoning.validity(user_input, generation_1)
                v_c = c + v
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    bot_response = TBLT.final_generator_pre(history_string,cus_prompt,v_c,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    bot_response = TBLT.final_generator_pre(history_string,cus_prompt,v_c,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
            elif stage == 0 and reason == 0:
                v = reasoning.validity_only(user_input)
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    bot_response = TBLT.final_generator_pre(history_string,cus_prompt,v,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    bot_response = TBLT.final_generator_pre(history_string, cus_prompt,v,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
            elif stage == 1 and reason == 0:
                v = reasoning.validity_only(user_input)
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    with st.expander("Related WordNet Information"):
                        word_list = vocab.vocab_fetch_processor(user_input)
                        wordnet_info = vocab.wordnet_interpreter_processor(word_list)
                        st.write(f'{wordnet_info}')
                    vocabulary = vocab.vocab_chat_with_model_generator(user_input,
                                                                     st.session_state['cus_prompt'], word_list,
                                                                     wordnet_info)
                    bot_response = TBLT.final_generator_vocab(history_string, cus_prompt, v,vocabulary,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    with st.expander("Related WordNet Information"):
                        word_list = vocab.vocab_fetch_processor(user_input)
                        wordnet_info = vocab.wordnet_interpreter_processor(word_list)
                        st.write(f'{wordnet_info}')
                    vocabulary = vocab.vocab_chat_with_model_generator(user_input,
                                                                     st.session_state['cus_prompt'], word_list,
                                                                     wordnet_info)
                    bot_response = TBLT.final_generator_vocab(history_string,cus_prompt, v,vocabulary,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
            elif stage == 2 and reason == 0:
                v = reasoning.validity_only(user_input)
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_during(history_string, cus_prompt,v,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                    if "Score 4" or "Score 5" in assessment:
                        st.session_state['stage']=3
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_during(history_string, cus_prompt,v,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                    if "Score 4" or "Score 5" in assessment:
                        st.session_state['stage']=3
            elif stage == 2 and reason == 1:
                generation_1 = reasoning.f1(user_input)
                generation_2 = reasoning.f2(user_input, generation_1)
                c = reasoning.f3(user_input, generation_1, generation_2)
                v = reasoning.validity(user_input, generation_1)
                v_c = c + v
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_during(history_string, cus_prompt,v_c,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                    if "Score 4" or "Score 5" in assessment:
                        st.session_state['stage']=3
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_during(history_string, cus_prompt,v_c,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                    if "Score 4" or "Score 5" in assessment:        
                        st.session_state['stage']=3
            elif stage == 3 and reason == 1:
                generation_1 = reasoning.f1(user_input)
                generation_2 = reasoning.f2(user_input, generation_1)
                c = reasoning.f3(user_input, generation_1, generation_2)
                v = reasoning.validity(user_input, generation_1)
                v_c = c + v
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_post(history_string, cus_prompt,v_c,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_post(history_string, cus_prompt,v_c,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
            else:
                v = reasoning.validity_only(user_input)
                if uploaded_file is not None and user_input.lower() == "uploaded":
                    file_path = image.compress_image(uploaded_file, st.session_state['path'])
                    gen_txt = image.orc_processor(file_path)
                    st.session_state['history'].append(gen_txt)
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_post(history_string, cus_prompt,v,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
                else:
                    st.session_state['history'].extend(["USER:", user_input])
                    history_string = " ".join(st.session_state['history'])
                    assessment = TBLT.chat_assessment_with_model_generator(history_string,cus_prompt)
                    bot_response = TBLT.final_generator_post(history_string, cus_prompt,v,assessment,topic)
                    st.session_state['history'].extend(["BOT_RESPONSE:", bot_response])
