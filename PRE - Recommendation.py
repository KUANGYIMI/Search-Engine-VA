import pandas as pd
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma



class EmployeeRecommendationEngine:

    def __init__(self,
                 participants = [],
                 start_time = None,
                 end_time = None,
                 location = [],
                 preference = None):
        
        self.participants = participants # Participants = ['Software Engineer with web service experience', 'HR', 'mark']
        self.start_time = start_time 
        self.end_time = end_time
        self.location = location
        self.preference = preference
        
        self.calendars, self.employees, self.collaboration, self.distance, self.vectordb = self._load_data()
        


    def _load_data(self):
        """
        Load the Employees and Calendars Data here
        """
        calendars = pd.read_csv('Inpixon Gen 30-employee Calendar.csv')
        employees = pd.read_csv('Inpixon Gen 30-employee data.csv')

        collaboration_matrix = pd.read_csv('collaboration_measure.csv', index_col=0)
        distance_matrix = pd.read_csv('distance_measure.csv', index_col=0)

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="./vector_employee/", embedding_function = embedding_function)

        return calendars, employees, collaboration_matrix, distance_matrix, vectordb



    def employee_coarse_match(self, participants, k = 5):
        """
        If the element in participants is Name -> Match Names in Employees Data
        If the element in participants is not names -> Match employees in Chroma DB (Large Number)
        """
        matched_employees = {}

        for participant in participants:
            # Initialize the list for this participant
            matched_employees[participant] = []

            # Check if participant is a full name in the employees DataFrame
            employee_match = self.employees[self.employees['FULLNAME'].str.lower() == participant.lower()]

            if not employee_match.empty:
                # If a matching employee is found, add their ID to the dictionary
                matched_employees[participant].append(employee_match.iloc[0]['ID'])
            else:
                # If no match, perform a similarity search in vectordb
                similar_employees = self.vectordb.similarity_search_with_relevance_scores(participant, k=k)

                # Extract and add employee IDs from the search results
                for document, _ in similar_employees:
                    employee_id = document.metadata['ID']
                    matched_employees[participant].append(employee_id)

                employee_scores = {document.metadata['ID']: score for document, score in similar_employees}


        return matched_employees, employee_scores 



    def employee_availability_filter(self, start_time, end_time):
        # real-time data (how to min the query # to connect to the PostGreSQL -> limit API calls to top 3) 
        """
        If User give time range: Keep employees who are available for the time range.
            If No employee satisfies the condition: Messages
        If User doesn't give time range: Pass
        """
        matched_employees, employee_scores = self.employee_coarse_match(self.participants)


        if (start_time is None) and (end_time is None):
            available_employees = matched_employees
            return available_employees, employee_scores
        

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)


        self.calendars['meeting_start_time'] = pd.to_datetime(self.calendars['meeting_start_time'])
        self.calendars['meeting_end_time'] = pd.to_datetime(self.calendars['meeting_end_time'])
        self.calendars['invited_members'] = self.calendars['invited_members'].apply(eval)

        # Filter for records where invited members overlap with the employee_id_list
        available_employees = {}

        for participant, employee_ids in matched_employees.items():
            available_employees[participant] = employee_ids.copy()

            # Filter for meetings that involve any of the employee IDs
            relevant_meetings = self.calendars[self.calendars['invited_members'].apply(lambda x: any(emp_id in x for emp_id in employee_ids))]

            for _, row in relevant_meetings.iterrows():
                if not (row['meeting_end_time'] < start_time or row['meeting_start_time'] > end_time):
                    for emp_id in row['invited_members']:
                        if emp_id in available_employees[participant]:
                            available_employees[participant].remove(emp_id)

        return available_employees, employee_scores




    def get_collaboration_score(self, matrix, user_id, emp_id):
        # Summing up the collaboration scores with other available employees
        return matrix.loc[user_id, emp_id]

    def get_distance_score(self, matrix, user_id, emp_id):
        # Summing up the distance scores with other available employees
        return matrix.loc[user_id, emp_id]

    def get_expertise_score(self, emp_id):
        employee_scores = self.employee_coarse_match(self.participants)[1]
        return employee_scores.get(emp_id, 0)


    def employee_score_lists(self, user_id, available_employees):
        """
        For the remaining employees:
            1. Add the Expertise Similarity Score
            2. Add the Distance Score
            3. Add the Collaboration Score
        """

        scored_available_employees = {}

        for participant, employee_ids in available_employees.items():
            scored_available_employees[participant] = {}

            for emp_id in employee_ids:
                collaboration_score = self.get_collaboration_score(self.collaboration_matrix, user_id, emp_id)
                distance_score = self.get_distance_score(self.distance_matrix, user_id, emp_id)
                expertise_score = self.get_expertise_score(emp_id)  
                # Assuming these function is defined 

                scored_available_employees[participant][emp_id] = {
                    'collaboration_score': collaboration_score,
                    'distance_score': distance_score,
                    'expertise_score': expertise_score
                }
            # Logic to calculate proximity factor

        return scored_available_employees
        


    def recommend_employees(self, scored_available_employees):
        """
        Check if user have employee recommendation preference:
            Prefer Location:
            Prefer Collaboration:
            Prefer Expertise:
            None:
        """

        recommendations = {}

        for role, employees in scored_available_employees.items():
            # Convert dictionary to a list of (employee_id, scores) for sorting
            employee_list = [(emp_id, scores) for emp_id, scores in employees.items()]

            if self.preference == 'Location':
                # Sort by distance score, higher is better
                sorted_employees = sorted(employee_list, key=lambda x: x[1]['distance_score'], reverse=True)
            elif self.preference == 'Collaboration':
                # Sort by collaboration score, higher is better
                sorted_employees = sorted(employee_list, key=lambda x: x[1]['collaboration_score'], reverse=True)
            elif self.preference == 'Expertise':
                # Sort by expertise score, higher is better
                sorted_employees = sorted(employee_list, key=lambda x: x[1]['expertise_score'], reverse=True)
            else:  # Default case or if preference is None
                # Sort by expertise, then distance, then collaboration
                sorted_employees = sorted(employee_list, key=lambda x: (x[1]['expertise_score'], x[1]['distance_score'], x[1]['collaboration_score']), reverse=True)

            # Recommend top 3 or all if fewer than 3
            recommendations[role] = [emp_id for emp_id, _ in sorted_employees[:3]]

        return recommendations