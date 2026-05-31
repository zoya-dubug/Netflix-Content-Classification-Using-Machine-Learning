import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_titles.csv")
print(df.describe(include = "all"))

#Handling Missing Values
df["director"] = df["director"].fillna("Unknown")
df["country"] = df["country"].fillna("Not Available")
df["cast"] = df["cast"].fillna("No Cast")

#Movies vs TV Show Distribution Plot
sns.countplot(x = df["type"])
plt.title("Movies vs TV Shows Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

#Rating Distribution Plot
plt.figure(figsize = (10,5))
plt.xticks(rotation = 45)
sns.countplot(data = df, x = "rating")
plt.title("Rating Distribution")
plt.tight_layout()
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

#Release Year Distribution Plot
plt.figure(figsize=(12,5))
sns.histplot(df["release_year"], bins=20)
plt.title("Netflix Content Release Distribution")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

#Top 10 Genres Plot
genre_df = df.assign(Genre=df["listed_in"].str.split(", ")).explode("Genre")
top_genre = genre_df["Genre"].value_counts().head(10)
sns.barplot(x = top_genre.values, y = top_genre.index)
plt.title("Top 10 Netflix Genres")
plt.xlabel("No. of Titles")
plt.ylabel("Genre")
plt.show()
