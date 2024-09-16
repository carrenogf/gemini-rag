import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from pathlib import Path

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")

def iniciar_historial(carpeta_archivos):
    files = []
    for file in os.listdir(carpeta_archivos):
        try:
            if file.endswith(".pdf"):
                files.append(upload_to_gemini(os.path.join(carpeta_archivos, file), mime_type="application/pdf"))
            if file.endswith(".txt"):
                files.append(upload_to_gemini(os.path.join(carpeta_archivos, file), mime_type="text/plain"))
        except Exception as e:
            print(f"Error uploading file {file}: {e}")

    # Some files have a processing delay. Wait for them to be ready.
    wait_for_files_active(files)
    prompt_inicial = "Eres un asistente experto en normativa gubernamental de la provincia de tucumán, en base a estos archivos que tienen relación al gobierno de la provincia de tucumán y su administración financiera, contesta mis preguntas intentando ajustarte lo máximo posible al contenido de los mismos"
    files.append(prompt_inicial)
    history = [{"role": "user","parts": files,},{"role": "model",
        "parts": [
        "¡Claro que sí! Estoy listo para responder a tus preguntas sobre el gobierno de la provincia de Tucumán y su administración financiera, basándome en los archivos que me has proporcionado. \n\nPor favor, hazme tus preguntas. 😊 \n",
        ],
        }]
    return history
    


# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
parent_folder = Path(os.getcwd()).resolve()

data = os.makedirs('data',exist_ok = True)

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

st.write('# Chat with Gemini')

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    

    with st.spinner('Procesando...'):
        # inicial_history = iniciar_historial(carpeta_archivos)
        #st.session_state.gemini_history = []
        print('new_cache made')
st.session_state.model = genai.GenerativeModel('gemini-1.5-flash')

if 'initialized' not in st.session_state:
    carpeta_archivos = os.path.join(os.getcwd(), "files")
    st.session_state.gemini_history = iniciar_historial(carpeta_archivos)
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history)
    st.session_state.initialized = True
else:
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
    )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    ## Send message to AI
    response = st.session_state.chat.send_message(
        prompt,
        stream=True,
    )
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        assistant_response = response
        # Streams in a chunk at a time
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                # Rewrites with a cursor at end
                message_placeholder.write(full_response + '▌')
        # Write full message with placeholder
        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )

    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )