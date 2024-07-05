import pandas as pd
import pickle

# Load the data and network graphs
anchor_sim_path = r'B:/6254_HW/second/wd_anchor_similarities.csv'
networks_path = r'C:/Users/11792/Desktop/data_similarity/wd/networks'
anchor_sim_data = pd.read_csv(anchor_sim_path)

# Create a set of tuples for anchor pairs and a dictionary for similarities
anchor_pairs_set = set(zip(anchor_sim_data['First Index'], anchor_sim_data['Second Index']))
similarity_dict = { (row['First Index'], row['Second Index']): row['Similarity'] for index, row in anchor_sim_data.iterrows() }

with open(networks_path, 'rb') as file:
    g1, g2 = pickle.load(file)

def find_matching_neighbors(g1, g2, anchor_pairs, similarity_dict):
    matching_pairs = set()
    details = []
    for a, a_star in anchor_pairs:
        b_set = set()
        b_star_set = set()
        neighbors_a = set(g1.neighbors(a))
        neighbors_a_star = set(g2.neighbors(a_star))
        matching_count = 0
        sim = similarity_dict.get((a, a_star), 0)  # Default to 0 if not found

        for b in neighbors_a:
            for b_star in neighbors_a_star:
                if (b, b_star) in anchor_pairs_set:
                    matching_pairs.add((b, b_star))
                    b_set.add(b)
                    b_star_set.add(b_star)
                    matching_count += 1

        total_neighbors_a = len(neighbors_a)
        total_neighbors_a_star = len(neighbors_a_star)
        total_b = len(b_set)
        total_b_star = len(b_star_set)
        proportion_1 = round(total_b / total_neighbors_a, 7)
        proportion_2 = round(total_b_star / total_neighbors_a_star, 7)
        details.append({
            'First Index': a,
            'Second Index': a_star,
            'Similarity': sim,
            'Total Neighbors a': total_neighbors_a,
            'Total Neighbors a_star': total_neighbors_a_star,
            'Matching Neighbor Pairs': matching_count,
            'total_b': total_b,
            'total_b_star': total_b_star,
            'b / a_neigh': proportion_1,
            'b_star / a_star_neigh': proportion_2
        })

    return matching_pairs, details

# Use the function and get results
matching_pairs, anchor_details = find_matching_neighbors(g1, g2, anchor_pairs_set, similarity_dict)

# Convert details to a DataFrame and save to CSV
details_df = pd.DataFrame(anchor_details)
details_df.to_csv('enhanced_anchor_matching_details.csv', index=False)
