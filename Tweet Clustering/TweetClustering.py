# Import necessary libraries
# Hector Lopez github
from tabulate import tabulate
import pandas as pd
import random as rd
import re
import math

# Preprocessing for data removing characters, and setting to all lowercase for Data
def preProcessing(df):
      df = re.sub('@[^\s]+','',df)
      df = re.sub('http[^\s]+','',df)
      df = re.sub('\W', ' ', df)
      df = df.lower() 
      
      return df

# Process K_Means with tweets Max Iterations set at 100
def k_means(tweets, num_clusters, max_iterations=100):
    centroids = []

    count = 0
    cluster_index_map = dict()
    
    # Assign random tweets as centroids
    while count < num_clusters:
        random_tweet_index = rd.randint(0, len(tweets) - 1)
        if random_tweet_index not in cluster_index_map:
            count += 1
            cluster_index_map[random_tweet_index] = True
            centroids.append(tweets[random_tweet_index])

    iteration_count = 0
    previous_centroids = []

    # Keep iterating until it's not converged or until Max is reached
    while (has_not_converged(previous_centroids, centroids)) and (iteration_count < max_iterations):
        # Assign tweets to closest centroid tweet 
        clusters = assign_clusters_to_tweets(tweets, centroids)
        
        # Used to keep track of previous centroid for k-means covergence
        previous_centroids = centroids
        
        # Update the centroid from current clusters
        centroids = update_centroids(clusters)
        iteration_count = iteration_count + 1

    # Print Warning when Max iterations are reached
    if (iteration_count == max_iterations):
        print("!! Maximum iterations reached !!")

    sse = find_SSE(clusters)
    return clusters, sse

# Check Convergence of centroids
def has_not_converged(prev, new):

    # Check if lengths are not equal 
    if len(prev) != len(new):
        return True
    
    # Check through each cluster and verify similarity
    for i in range(len(new)):
        if " ".join(new[i]) != " ".join(prev[i]):
            return True

    return False

# Cluster assingment for tweets
def assign_clusters_to_tweets(tweets, centroids):
    clusters = dict()

    # Assign a centroid to each tweet based on nearest centroid
    for x in range(len(tweets)):
        minimum_distance = math.inf
        cluster_index = -1

        # Search for nearest centroid of the current tweet
        for i in range(len(centroids)):
            distance = find_distance(centroids[i], tweets[x])
        
            if centroids[i] == tweets[x]:
                cluster_index = i
                minimum_distance = 0
                break

            if distance < minimum_distance:
                cluster_index = i
                minimum_distance = distance

        # If no common centroid randomize assingment
        if minimum_distance == 1:
            cluster_index = rd.randint(0, len(centroids) - 1)

        # Centroid assignment to tweet
        clusters.setdefault(cluster_index, []).append([tweets[x]])
        # Find tweet distance from nearest centroid for SSE
        last_tweet_index = len(clusters.setdefault(cluster_index, [])) - 1
        clusters.setdefault(cluster_index, [])[last_tweet_index].append(minimum_distance)

    return clusters


# Updating centroids based on clusters
def update_centroids(clusters):
    centroids = []

    # Search for tweet with closest distance sum with all other tweets in cluster
    for i in range(len(clusters)):
        minimum_distance_sum = math.inf
        centroid_index = -1

        # To stop unecessary calculations
        minimum_distance_dp = []

        for x1 in range(len(clusters[i])):
            minimum_distance_dp.append([])
            sum_distance = 0

            # Retrieve distance sum for every tweet in the same cluster
            for x2 in range(len(clusters[i])):
                if x1 != x2:
                    if x2 < x1:
                        distance = minimum_distance_dp[x2][x1]
                    else:
                        distance = find_distance(clusters[i][x1][0], clusters[i][x2][0])

                    minimum_distance_dp[x1].append(distance)
                    sum_distance += distance
                else:
                    minimum_distance_dp[x1].append(0)

            # Assign tweet with minimum distance sum to be centroid for cluster
            if sum_distance < minimum_distance_sum:
                minimum_distance_sum = sum_distance
                centroid_index = x1

        # Add selected tweet to centroids
        centroids.append(clusters[i][centroid_index][0])

    return centroids

def find_distance(tweet1, tweet2):
    intersection = set(tweet1).intersection(tweet2)
    union = set().union(tweet1, tweet2)

    return 1 - (len(intersection) / len(union))

# Calculate SSE of clusters
def find_SSE(clusters):
    sse = 0
    
    # Calculate SSE of distances of the tweet from its centroid, for all clusters
    for x in range(len(clusters)):
        for i in range(len(clusters[x])):
            sse = sse + (clusters[x][i][1] * clusters[x][i][1])

    return sse

# Print out resuluts in table
def format_print(results):
	print(tabulate(results, headers=['Value of K', 'SSE', 'Size of each cluster'], tablefmt='fancy_grid'))

# Run data through K_Means Clustering algorithm
def runData(practice_data, k_values):
    
    results = []

    # Run through K-means with data and Cluster Size(k)
    for k in k_values:
        clusters, sse = k_means(practice_data['Tweets'], k)
        cluster_size = ""

        # Format Cluster coutn and Size of Cluster
        for c in range(len(clusters)):
            cluster_size += str(c+1) + ": "+ str(len(clusters[c])) + " tweets \n"
        results.append([k, sse, cluster_size])

    return results

# Retrieve data from Github repo
data = pd.read_csv("https://raw.githubusercontent.com/Hlopez490/ML01/main/usnewshealth.txt", sep='|', header=None)

# Data formatting
data = data.drop(0, axis=1)
data = data.drop(1, axis=1)
data.rename(columns = {2:'Tweets'}, inplace = True)

# Preproccess data
data['Tweets'] = data['Tweets'].apply(preProcessing)

# Run Data through K-Means Clustering
practice_data = data
k_values = [2, 5, 8, 10, 15]
results = runData(practice_data, k_values)

# Format results with Tabulate
format_print(results)

