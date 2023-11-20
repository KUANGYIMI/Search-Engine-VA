from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np

# Load the Data
employee_data = pd.read_csv('Inpixon Gen 30-employee data')
booking_logs = pd.read_csv('Inpixon Gen 30-employee Calendar.csv')


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r


def generate_distance_matrix(employee_data, default_distance=9999):
    """
    Generate a distance matrix from employee location data.
    Follows the specified rules for distance calculations.
    If two employees are in the same building (as indicated by the 'Campus' column), their distance is set to 0.
    If two employees are not in the same campus, we'll use their coordinates to calculate their distance.
    If an employee's coordinates are 0,0, their distance to everyone else is set to 9999 km.
    """
    
    # Create an empty DataFrame for the distance matrix
    n = len(employee_data)
    distance_matrix = pd.DataFrame(default_distance, index=employee_data['ID'], columns=employee_data['ID'])

    # Calculate pairwise distances
    for i, loc1 in employee_data.iterrows():
        for j, loc2 in employee_data.iterrows():
            if i != j:
                # Check if in the same campus
                if loc1['Campus'] == loc2['Campus']:
                    distance_matrix.loc[loc1['ID'], loc2['ID']] = 0
                    distance_matrix.loc[loc2['ID'], loc1['ID']] = 0
                # Check if any employee has coordinates (0,0) or missing coordinates
                elif (loc1['Latitude'] == 0 and loc1['Longitude'] == 0) or \
                     (loc2['Latitude'] == 0 and loc2['Longitude'] == 0) or \
                     pd.isna(loc1['Latitude']) or pd.isna(loc1['Longitude']) or \
                     pd.isna(loc2['Latitude']) or pd.isna(loc2['Longitude']):
                    continue  # Keep the default distance
                else:
                    # Calculate distance using coordinates
                    dist = haversine(loc1['Latitude'], loc1['Longitude'], loc2['Latitude'], loc2['Longitude'])
                    distance_matrix.loc[loc1['ID'], loc2['ID']] = dist
                    distance_matrix.loc[loc2['ID'], loc1['ID']] = dist

    return distance_matrix


# One Time Generation
def generate_collaboration_matrix(booking_logs):
    """
    Generate a collaboration matrix from booking logs.
    """
    # Parse the 'invited_members' column to get a list of employee IDs for each meeting
    booking_logs['invited_members'] = booking_logs['invited_members'].apply(eval)

    # Get a list of unique employee IDs
    unique_ids = set()
    for members in booking_logs['invited_members']:
        unique_ids.update(members)
    unique_ids = list(unique_ids)

    # Initialize an empty matrix or a DataFrame to store collaboration counts
    collaboration_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Count collaborations
    for members in booking_logs['invited_members']:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                collaboration_matrix.loc[members[i], members[j]] += 1
                collaboration_matrix.loc[members[j], members[i]] += 1

    return collaboration_matrix