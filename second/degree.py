import pickle
import networkx
with open("foursquare_following", "rb") as f:
    G1 = pickle.load(f)

with open("twitter_following", "rb") as f:
    G2 = pickle.load(f)

degree_sum = sum(dict(G1.degree()).values()) + sum(dict(G2.degree()).values())
average_degree = degree_sum / (G1.number_of_nodes()+G2.number_of_nodes())
print(f"图的平均度数: {average_degree}")