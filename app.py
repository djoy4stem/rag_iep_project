import streamlit as st
from collections import namedtuple
from iep_goal_generator import My_IEP_Goal_Generator
from rag_utils import StudentProfile

import sys





open_ai_key = "MY_OPEN_AI_API_KEY"  # Replace with your actual API key

@st.cache_resource
def load_agent():
    """
    This function caches the agent across reruns.
    We assume our agent is deterministic and does not change wth input
    """
    agent = My_IEP_Goal_Generator(model="gpt-4", open_ai_key=open_ai_key)
    agent.create_rag_pipeline()
    return agent

# if "agent" not in st.session_state:
#     # Load or create the IEP agent
#     # This checks whether the agent is in memory for the duration of the session.
#     agent = My_IEP_Goal_Generator(model="gpt-4", open_ai_key=open_ai_key)
#     agent.create_rag_pipeline()


# Set up the page
st.set_page_config(page_title="IEP Goal Generator", layout="centered")



agent = load_agent()


# Set up tabs
tab1, tab2 = st.tabs(["🎯 IEP Goal Generator", "💬 Live Conversation"])

with tab1:

    st.title("🎯 IEP Goal Generator")

    # st.write(f"Python path: {sys.executable}")

    # Student Profile Input
    st.subheader("Enter Student Information")


    grade_options = ["Freshman / 9th grade", "Sophomore / 10th grade", "Junior / 11th grade", "Senior / 12th grade"]

    # Initialize session state defaults if not set
    if "name" not in st.session_state:
        st.session_state.name = ""
    if "age" not in st.session_state:
        st.session_state.age = 10
    if "grade" not in st.session_state:
        st.session_state.grade = "Sophomore / 10th grade"
    if "career_interests_or_category" not in st.session_state:
        st.session_state.career_interests_or_category = ""
    if "learning_preferences" not in st.session_state:
        st.session_state.learning_preferences = ""
    if "onnet_results" not in st.session_state:
        st.session_state.onnet_results = ""
    if "career_suggestions" not in st.session_state:
        st.session_state.career_suggestions = ""
    if "preferred_employers" not in st.session_state:
        st.session_state.preferred_employers = ""

    # Button to load example profile. Here, we enter the profile of a student named Clarence
    if st.button("Use Example Profile"):
        st.session_state.name = "Clarence"
        st.session_state.age = 15
        st.session_state.grade = "Sophomore / 10th grade"
        st.session_state.career_interests_or_category = "Retail Salesperson, Driver/Sales Worker"
        st.session_state.learning_preferences = "Hands-on instruction"
        st.session_state.onnet_results = "Strength in Enterprising activities"
        st.session_state.career_suggestions = "Retail Sales, Driver/Sales Worker"
        st.session_state.preferred_employers = "Walmart"

    grade_index = grade_options.index(st.session_state.grade) if st.session_state.grade in grade_options else 0

    with st.form("Student Information"):
        st.session_state.name = st.text_input("Student Name", value=st.session_state.name)
        st.session_state.age = st.number_input("Age", min_value=10, max_value=22, step=1, value=st.session_state.age)
        st.session_state.grade = st.selectbox("Grade", grade_options, index=grade_index)
        st.session_state.career_interests_or_category = st.text_area(
            "Career Interests or Category", help="e.g., Retail Sales, Driver/Sales Worker", value=st.session_state.career_interests_or_category
        )
        st.session_state.learning_preferences = st.text_input(
            "Learning Preferences (optional)", help="e.g., Hands-on instruction", value=st.session_state.learning_preferences
        )
        st.subheader("Assessment Results")
        st.session_state.onnet_results = st.text_input(
            "O*Net Interest Profiler Result", help="e.g., Strength in Enterprising activities", value=st.session_state.onnet_results
        )
        st.session_state.career_suggestions = st.text_area(
            "Career Suggestions", help="e.g., Retail Salesperson, Driver/Sales Worker", value=st.session_state.career_suggestions
        )
        st.session_state.preferred_employers = st.text_input(
            "Preferred Employer(s)", help="e.g., Walmart", value=st.session_state.preferred_employers
        )

        submitted = st.form_submit_button("Generate IEP Goals")


    if submitted:
        if st.session_state.age is None:
            st.error("Age is required!")   

        if not st.session_state.grade.strip():
            st.error("Grade is required!")

        if not st.session_state.onnet_results.strip():
            st.error("O*Net Interest Profiler results are required!")

        if not st.session_state.career_suggestions.strip():
            st.error("Career suggestions are required!")


        with st.spinner("Generating goals..."):

            student_profile = StudentProfile(
                name=st.session_state.name,
                age=st.session_state.age,
                grade=st.session_state.grade,
                career_interest_or_category=st.session_state.career_interests_or_category,
                # career_interests=career_interests,
                learning_preferences=st.session_state.learning_preferences,
                onnet_results=st.session_state.onnet_results,
                career_suggestions=st.session_state.career_suggestions,
                preferred_employers=st.session_state.preferred_employers
            )


            # Generate the IEP goals
            iep_output, relevant_docs = agent.generate_iep_goals(student_profile, k=10)

            # Display results
            st.success("✅ IEP Goals Generated!")
            print("✅ IEP Goals Generated!")
            st.markdown("---")
            st.markdown(f"### IEP Goals and Transition Plan for **{st.session_state.name}**")
            st.markdown(iep_output.content)


# -------------------------
# Live conversation tab
# -------------------------

## This version uses a regular st.text_area instead() of the chat_input() function.
# An advantage is that you can easily set it so that it always stays on top.
# On the contrary, st.chat_input() always renders at the bottom of the app, and
# streamlit does not allow placing it elsewhere.
# Here, we make sure the  question * aswer pairs are displayed in reversed order, 
# i.e. the latest pair is placed on top.

with tab2:
    st.title("💬 Talk with the IEP Goal Assistant")

    # Input at the top
    with st.form(key="chat_form"):
        user_prompt = st.text_area("Type your question:", height=100)
        submitted = st.form_submit_button("Send")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if submitted and user_prompt.strip():
        # Append user's message
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

        # Get response
        with st.spinner("Thinking..."):
            try:
                response = agent.generate_response(user_prompt)
            except Exception as e:
                response = f"❌ Error: {e}"

        # Append agent's response
        st.session_state.chat_messages.append({"role": "assistant", "content": response['result']})


    # Group messages in (user, assistant) pairs
    messages = st.session_state.chat_messages
    message_pairs = list(zip(messages[::2], messages[1::2]))  # step in pairs

    # Display latest pairs first
    for user_msg, assistant_msg in reversed(message_pairs):
        with st.chat_message(user_msg["role"]):
            st.markdown(user_msg["content"])
        with st.chat_message(assistant_msg["role"]):
            st.markdown(assistant_msg["content"])


### 
# everytime the the chat input window is displayed between the last two questions/answers.
###

# with tab2:

#     st.title("💬 Talk with the IEP Goal Assistant")

#     st.markdown("Use this chat to explore questions or get further personalized support from the IEP Assistant.")


#     # Initialize session state for chat
#     if "chat_messages" not in st.session_state:
#         st.session_state.chat_messages = []

#     # Display previous messages
#     for msg in st.session_state.chat_messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     # User input at the bottom
#     user_prompt = st.chat_input("Ask me anything about IEP goals or planning...")




#     if user_prompt:
#         # Display user message
#         st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
#         with st.chat_message("user"):
#             st.markdown(user_prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Replace with your actual response function
#                     response = agent.generate_response(user_prompt)
#                 except Exception as e:
#                     response = f"❌ Sorry, there was an error: `{e}`"

#                 st.markdown(response['result'])

#         # Save agent response
#         st.session_state.chat_messages.append({"role": "assistant", "content": response['result']})
