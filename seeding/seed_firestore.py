from google.cloud import firestore
import json
import os

# Use the same credentials as your main app
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../firebase-key.json"
db = firestore.Client()

def seed_modules():
    # Your seed data
    seed_data = {
        "modules/1": {
            "id": 1,
            "title": "AI and ML Around Us",
            "description": "Explore how artificial intelligence and machine learning are already shaping your everyday life, from social media to shopping apps.",
            "icon": "Brain",
            "lessons": [
                {
                "id": "l1",
                "title": "What is AI and ML? Everyday examples",
                "content": "AI (Artificial Intelligence) and ML (Machine Learning) are technologies that enable machines to perform tasks that normally require human intelligence. From voice assistants and self-driving cars to Netflix and TikTok recommendations — they’re all powered by AI/ML. This lesson breaks down the difference between the two and shows where you encounter them in daily life.",
                "order": 1,
                "type": "video",
                "videoUrl": ""
                },
                {
                "id": "l2",
                "title": "How recommender systems work",
                "content": "Understand the basic logic behind recommender systems: content-based filtering, collaborative filtering, and how platforms decide what to show you.",
                "order": 2
                },
                {
                "id": "l3",
                "title": "Python mini-project – Movie Recommender",
                "content": "Build a simple rule-based recommender system in Python using `if/else` statements and lists. Practice writing your own logic to suggest movies to a user.",
                "order": 3
                }
            ],
            "project": "Build a simple rule-based movie recommender in Python."
            }

        }
    
    # Upload to Firestore
    for path, data in seed_data.items():
        collection, doc_id = path.split('/')
        db.collection(collection).document(doc_id).set(data)
    
    print("Seed data uploaded successfully!")

if __name__ == "__main__":
    seed_modules()