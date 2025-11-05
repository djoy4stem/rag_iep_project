import os
from collections import namedtuple
from pathlib import Path
from typing import List


from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema  import HumanMessage

import warnings



PARENT_DIR = Path(__file__).resolve().parent


class RAGUtils:


    @staticmethod
    def create_and_save_embeddings(documents, open_ai_key, store_path= None):
        """
        Converts text chunks into embeddings using OpenAI, then stores them in a FAISS index for fast similarity search.
        """
        if store_path is None:
            store_path = os.path.join(PARENT_DIR, "data/faiss_store")
        try:
            # Step 3: Create embeddings and store in vector database
            # Creates vector representations of text chunks for semantic search
            embeddings = OpenAIEmbeddings(
                api_key=open_ai_key  # Replace with your actual API key
            )
            # FAISS is an efficient similarity search library
            vectorstore = FAISS.from_documents(documents, embeddings)
            print("FAISS Vector database created successfully")
            return vectorstore

        except Exception as exp:
            raise Exception(exp)


    @staticmethod
    def load_vectorstore(path="faiss_store"):
        """
        Loads an existing FAISS vector store from local storage.
        """
        embedding_model = OpenAIEmbeddings()
        return FAISS.load_local(path, embedding_model)


    @staticmethod
    def retrieve_relevant_documents(vectorstore:FAISS
                , query:str= "IEP goals, IEP transition plan, disabilities act, academic standards, job profiles."               
                , k:int=5
                , min_sim_score = None
                # , info_categories:str=None
                # , all_categories=False
                
                ) -> List[Document]:
        "Retrieves the top-k most relevant documents to a specied query."

        results = vectorstore.similarity_search_with_score(query, k=k)

        if not min_sim_score is None:
            results = [res for res, score in results if score>=min_sim_score]
        else:
            results = [res for res, score in results]

        # if info_categories:
        #     results = [doc for doc in results if doc.metadata.get("info_category") in info_categories]

        # # if all_categories:
        # #     for k in info_categories:
        # #         r_ = [doc for doc in results if doc.metadata.get("info_category") == k]
        # #         if len(r_) == 0:
        # #             warnings.warn(f"No relevant document was found with info category: '{k}'.\n\tReturning None.")
        # #             return None
        
        return results



    @staticmethod
    def generate_iep_goals(chat_model:ChatOpenAI, student_info:dict):

        ## Format the prompt by adding the student's profile and relevant documents
        ## From the vector store
        formatted_prompt = IEP_RAG_PROMT.format(**student_info)

        # Generate the response
        # The chat model expect a specific format. We will use HumanMessage
        response = chat_model.invoke([HumanMessage(content=formatted_prompt)])

        return response





# Define the structure of the student info
# This is to make sure we provide the correct types of information
StudentProfile = namedtuple("StudentProfile", [
    "name",
    "age",
    "grade",
    "career_interest_or_category",
    "learning_preferences",
    "onnet_results",
    "career_suggestions",
    "preferred_employers"
])

IEP_RAG_PROMT = PromptTemplate(
    input_variables=[
        "student_name",
        "student_age",
        "student_grade",
        "career_interest_or_category",
        "learning_preferences",
        "onnet_results",
        "career_suggestions",
        "preferred_employers",
        "retrieved_docs"
    ],
    template="""
You are an IEP transition planning expert. Based on the provided student profile and assessment data, use the retrieved documents 
(including standards from the Occupational Outlook Handbook, 21st Century Skills, and IEP templates) to generate a compliant, 
well-structured Individualized Education Program (IEP) goals for students with disabilities that align with industry standards 
and educational frameworks..

---

## Student Information:
- Name: {student_name}
- Age: {student_age}
- Grade: {student_grade}
- Career Interest/Category: {career_interest_or_category}
- Learning Preferences: {learning_preferences}

## Assessment Results:
- O*Net Interest Profiler: {onnet_results}
- Career Suggestions: {career_suggestions}
- Preferred Employers: {preferred_employers}

## Retrieved Context Documents:
{retrieved_docs}

## Instructions for Handling Missing or Ambiguous Information:

- If any student information (such as learning preferences or preferred employers) is missing or unclear, use general but relevant career-category aligned goals based on the retrieved standards.
- If career interests or assessment results are ambiguous or conflicting, prioritize measurable goals focusing on broad transferable skills (e.g., communication, workplace behavior).
- Clearly note any assumptions you make in your output.
- Avoid creating fictituous information and names for employers
- When in doubt, generate goals that reflect best practices for the specified career category and align with industry standards and state educational content standards.
- If no relevant documents were found for the suggested careers (i.e., nor relevant documents with info category 'career profile'), return a message to the user that says so.


---

## Tasks:

Using the student profile and retrieved documents, perform the following:

1. **Postsecondary Goals**
   - Create one measurable Employment Goal
   - Create one measurable Education or Training Goal

2. **Measurable annual goal aligned with standards:**
   - Must support the stated career path
   - Focus on a skill relevant to the career (e.g., communication, driving skills, punctuality, customer service)
   - Use language and expectations from the retrieved standards

3. **2–3 Short-Term Objectives**
   - That build toward the annual goal
   - Must be measurable and observable
   - Use common IEP formatting as seen in the retrieved documents

4. **Alignment to Standards**
   - Explain how the goals align with:
     - Career expectations from the Occupational Outlook Handbook (OOH)
     - Workplace readiness or academic standards (e.g., 21st Century Skills)
   - Refer to specific phrases, duties, or performance expectations found in the retrieved content

---

## Output Format:

**Postsecondary Goal:**  

1. Employment
[Insert]


2. Education/Training
[Insert]

**Annual IEP Goal:**  
[Insert]

**Short-Term Objectives:**  
- Objective 1: [Insert]  
- Objective 2: [Insert]  
- Objective 3 (optional): [Insert]

**Alignment to Standards:**  
1. Career Standards:    [Summarize alignment with OOH for the occupation]  
2. Education Standards: [Summarize alignment with 21st Century Skills or state frameworks]
"""
)


IEP_CHAT_PROMPT = PromptTemplate(
                        input_variables=["context", "question"], 
                        template= """
                        Answer the question based only on the following context:

                        {context}

                        Question: {question}

                        If the information to answer the question is not present in the context, 
                        respond with "I don't have enough information to answer this question."

                        Answer:
                        """

                    )

