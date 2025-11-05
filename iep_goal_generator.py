
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


from rag_utils import RAGUtils, IEP_RAG_PROMT, IEP_CHAT_PROMPT, StudentProfile
from data_utils import DataProcessor

import warnings




class My_IEP_Goal_Generator:
    def __init__(self, open_ai_key:str, model:str = "gpt-4", vstore_path:str=None):
        # Initialize the language model
        self.open_ai_key = open_ai_key
        self.chat_model = ChatOpenAI(model=model)
        if vstore_path is None:
            ## retrieve and process all documents
            docs = DataProcessor.collect_and_process_documents()
            docs = [di[-1] for di in docs.items()]
            docs = [item for sublist in docs for item in sublist]

            # print(set(doc.metadata['info_category'] for doc in docs))

            ## create and save a vector store
            self.vectorstore = RAGUtils.create_and_save_embeddings(documents=docs, open_ai_key=self.open_ai_key)
        else:
            self.vectorstore = RAGUtils.load_vectorstore(vstore_path)
        
    
    def generate_iep_goals(self, student_profile: StudentProfile, k:int=5, min_sim_score=None):
        assert student_profile.career_interest_or_category is not None, "Please provide a non-null occupation" 

        ## We must make sure to have documents relevant to these categories.
        must_info_categories = ['career_profile', 'state_standards']

        ### Formulate a qurey to retrieve documents relevant to career suggetions, and IEP planning
        query = f"IEP goals, IEP transition plan, disabilities act, academic standards, career profiles for {student_profile.career_suggestions}."
        print(f"query = {query}")
        relevant_docs = RAGUtils.retrieve_relevant_documents(vectorstore=self.vectorstore
                                                                , query=query
                                                                , k=k
                                                                , min_sim_score = min_sim_score                                                               
                                                            )




        if relevant_docs is None or len(relevant_docs) == 0:
            ## The AI Agent returns a response that indicates no relevant document were found for the specified career interest or suggestion.
            return SystemMessage(content=f"No relevant document were found for the specified career interest(s) or suggestion(s): {student_profile.career_suggestions}.")
        
        else:
            
            ## To limit hallucinations, we can make sure that docuemnts are retrieved for at number of categories.
            doc_info_categories = [doc.metadata.get('info_category') for doc in relevant_docs]
            print(f"doc_info_categories =          {set(doc_info_categories)}")
            print(f"Number of retrieved documents: {len( relevant_docs)}\n\n")
            # print("\n".join([doc.metadata['source'] + "\n\t" + doc.page_content[:150] for doc in relevant_docs]))
            # print("\n".join([doc.metadata['source'] for doc in relevant_docs]))
            missing_categories = [mic for mic in must_info_categories if not mic in doc_info_categories]
            # print(missing_categories)


            if len(missing_categories)>0:
                return SystemMessage(content=f"No relevant document could be found for:" + \
                     f"\n- the specific career interest(s) or suggestion(s): {student_profile.career_suggestions} \n- in the categories: {'; '.join(missing_categories)}\n"), relevant_docs

            if not 'career_profile' in doc_info_categories:
                ## The AI Agent returns a response that indicates no relevant document were found for the specified career interest or suggestion.
                # warnings.warn("No relevant document were found for the specified career interest or suggestion.")
                return SystemMessage(content=f"No relevant document were found for the specified career interest(s) or suggestion(s): {student_profile.career_suggestions}."), relevant_docs
            else:

                prompt_inputs = {
                    "student_name": student_profile.name,
                    "student_age": student_profile.age,
                    "student_grade": student_profile.grade,
                    "career_interest_or_category": student_profile.career_interest_or_category,
                    "learning_preferences": student_profile.learning_preferences,
                    "onnet_results": student_profile.onnet_results,
                    "career_suggestions": student_profile.career_suggestions,
                    "preferred_employers": student_profile.preferred_employers,
                    "retrieved_docs": relevant_docs
                }


                response = RAGUtils.generate_iep_goals(chat_model=self.chat_model, student_info=prompt_inputs)

                return response, relevant_docs

    def create_rag_pipeline(self, k:int=3):
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # Use semantic similarity for search
            search_kwargs={"k": k}  # Return top k most relevant chunks
        )

        self.qa_chain = RetrievalQA.from_chain_type(
                                llm=self.chat_model,
                                chain_type="stuff",  # "stuff" method: simply stuffs all retrieved documents into prompt
                                retriever=self.retriever,
                                chain_type_kwargs={"prompt": IEP_CHAT_PROMPT}
                            )

    def generate_response(self, message:str):
        message =  message + "Provide an answer in bullet point format, when applicable."
        response = self.qa_chain.invoke({"query": message})
        return response


    def launch_interactive_convo(self, print_source:bool=False, source_len:int=100):
        print("\n--- RAG Interactive Demo ---")
        print("Type 'exit' to end the demo")
        
        while True:
            user_query = input("\nEnter your question: ")
            if user_query.lower() == 'exit':
                break

            print(f"\n\nQuestion: { user_query}")  

            ## Adding some instructions tor format the reponse
            user_query = user_query + "Provide an answer in bullet point format, when applicable."

            response = self.qa_chain.invoke({"query": user_query})

            print(f"\nAnswer:\n--------\n{response['result']}")
            
            if print_source:
                # Optional: Display source chunks
                print("\nSource chunks (for reference):")
                docs = retriever.get_relevant_documents(user_query)
                for i, doc in enumerate(docs):
                    print(f"Chunk {i+1}: {doc.page_content[:source_len]}...")

    


import re

class GoalAssessment:
    @staticmethod
    def evaluate_iep_goal(goal_as_text, student_profile, retrieved_docs):
        evaluation = {
            "Specific": False,
            "Measurable": False,
            "Achievable": False,
            "Relevant": False,
            "Time-bound": False,
            "Aligned with Career Interest": False,
            "Aligned with Standards": False
        }
        
        ## Searches for sequences such as 90%, 90 percent, liker scale, etc.
        re_pattern = r'([1-9]*[0-9][%|\spercent]|\bpercent\b|\d+\s*out\s*of\s*\d+|likert scale|as\smeasured\sby)'

        if not (goal_as_text is None or len(goal_as_text) ==0 
                    or 'No relevant document could be found for' in goal_as_text):
            goal_as_text = goal_as_text.lower()

            short_term_goals = goal_as_text.lower().split('short-term objectives:')[-1].split('alignment to standards:')[0]
            # print(f"short_term_goals ({len(short_term_goals)}) :", short_term_goals)
            annual_iep_goal = goal_as_text.lower().split('annual iep goal:')[-1].split('short-term objectives:')[0]

            # 1. SMART Criteria
            evaluation["Specific"]   = student_profile.name.lower() in goal_as_text
            evaluation["Measurable"] = any(word in goal_as_text for word in ["demonstrate", "perform", "complete", "respond"])


            evaluation["Achievable"] = bool(re.search(re_pattern, annual_iep_goal + short_term_goals))
            
            # print(f"\n\n{[word for word in student_profile.career_interest_or_category.lower().split(',')]}\n\n")
            evaluation["Relevant"]   = any(word in goal_as_text for word in student_profile.career_interest_or_category.lower().split(','))

            evaluation["Time-bound"] = any(word in short_term_goals for word in ["weeks", "months", "by", "after high school", "semester", "trimester", "by the end of"])

            # 2. Student interest alignment
            evaluation["Aligned with Career Interest"] = any(
                term.lower() in goal_as_text for term in [
                    student_profile.career_suggestions, 
                    student_profile.preferred_employers,
                    student_profile.onnet_results
                ]
            )

            # 3. Standards alignment based on retrieved docs
            standards_doc = [doc for doc in  retrieved_docs if doc.metadata['info_category']=='state_standards']
            standards = " ".join(doc.page_content for doc in standards_doc).lower()


            evaluation["Aligned with Standards"] = any(
                keyword in standards
                for keyword in ["customer service", "21st century skills", "occupational outlook", "transition planning"]
            )

            # Score summary
            total_score = sum(evaluation.values())
            print(f"Total Score: {total_score}/7")


            return evaluation
        else:
            return None